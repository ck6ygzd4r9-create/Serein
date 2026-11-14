//! Desktop ownership for watching one other participant's screen share. The choice is an
//! explicit click; frames are decoded off the render thread and never written to disk.
use client_core::{Command, Event, State, screen, voice};
use discord_voice::{Identity, Status};
use eframe::egui;
use model::Id;
use std::{
	sync::{Arc, Mutex, atomic::Ordering},
	time::{Duration, Instant},
};
use tokio::{runtime::Runtime, sync::watch, task::JoinHandle};
use zeroize::Zeroizing;

const SIGNAL_TIMEOUT: Duration = Duration::from_secs(30);
// Eight UHD YUV pictures plus the displayed picture fit without increasing the item bound.
const DECODED_BUDGET: usize = 128 * 1024 * 1024;

/// Latest stream picture and whether the UI still has to upload it.
type Slot = Option<(platform::video::Picture, bool)>;

/// Replace the latest picture in place: a plain copy of the decoder's planes into one reused
/// allocation. Colour conversion happens on the GPU when the UI uploads it.
fn store_frame(slot: &mut Slot, frame: discord_voice::RemoteFrame<'_>) -> bool {
	let picture = frame.picture;
	if !picture.is_valid() {
		return false;
	}
	let len = picture.data.len();
	let mut data = slot.take().map(|(old, _)| old.data).unwrap_or_default();
	if data.capacity() > 1024 * 1024 && data.capacity() > len.saturating_mul(4) {
		// Release a previous large resolution while keeping ordinary resize reuse.
		data = Vec::new();
	}
	data.clear();
	// Exact growth: never double a previous resolution's allocation beyond the frame.
	data.reserve_exact(len);
	data.extend_from_slice(picture.data);
	*slot = Some((picture.with_data(data), true));
	true
}

// Decode early, display on the audio/device clock. Only decoded pictures may be
// skipped here: the H.264 decoder has already consumed their reference state.
#[derive(Default)]
struct Scheduled {
	latest: Slot,
	pending: std::collections::VecDeque<(platform::video::Picture, discord_voice::Presentation)>,
	bytes: usize,
}
impl Scheduled {
	fn store(&mut self, frame: discord_voice::RemoteFrame<'_>) -> bool {
		let Some(presentation) = frame.presentation.clone() else {
			return store_frame(&mut self.latest, frame);
		};
		if !presentation.is_current() || !frame.picture.is_valid() {
			return false;
		}
		let length = frame.picture.data.len();
		if length > DECODED_BUDGET {
			return false;
		}
		let old = if self.latest.as_ref().is_some_and(|(_, fresh)| !*fresh) {
			self.latest.take()
		} else {
			None
		};
		let mut data = old.map(|(picture, _)| picture.data).unwrap_or_default();
		if data.capacity() > length.saturating_mul(4) {
			data = Vec::new();
		}
		let required = data.capacity().max(length);
		while self.pending.len() >= 8
			|| self.bytes
				+ required + self
				.latest
				.as_ref()
				.map_or(0, |(picture, _)| picture.data.capacity())
				> DECODED_BUDGET
		{
			let Some((old, _)) = self.pending.pop_front() else {
				return false;
			};
			self.bytes -= old.data.capacity();
		}
		data.clear();
		data.reserve_exact(length);
		data.extend_from_slice(frame.picture.data);
		self.bytes += data.capacity();
		self.pending
			.push_back((frame.picture.with_data(data), presentation));
		true
	}
	fn advance(&mut self, now: Instant) -> Option<Instant> {
		while let Some((_, stamp)) = self.pending.front() {
			if stamp.is_current() && stamp.deadline(now) > now {
				return Some(stamp.deadline(now));
			}
			let (picture, stamp) = self.pending.pop_front().expect("front exists");
			self.bytes -= picture.data.capacity();
			if stamp.is_current() {
				self.latest = Some((picture, true));
			}
		}
		None
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Context {
	generation: u64,
	channel: Id,
	request: u64,
	stream_request: u64,
	streamer: Id,
}
struct Pending {
	context: Context,
	user: Id,
	peer: Option<Id>,
	session: Zeroizing<String>,
	identity: Arc<Identity>,
	audio: Option<std::sync::mpsc::SyncSender<discord_voice::PlaybackFrame>>,
	rtc: Option<(Id, Id)>,
	server: Option<(voice::Secret, String)>,
	started: Instant,
}
#[derive(Clone, Copy)]
enum Notice {
	Status(&'static str),
	Failed(&'static str),
}
struct Live {
	context: Context,
	task: JoinHandle<()>,
	events: watch::Receiver<Option<Notice>>,
	frames: Arc<Mutex<Scheduled>>,
}

/// One watched stream: its negotiation, connection, decoder queue and GPU picture. Views are
/// independent, so ending, failing or closing one never touches another.
struct View {
	streamer: Id,
	pending: Option<Pending>,
	live: Option<Live>,
	/// The streamer stopped or Discord failed the view; the state choice is cleared next poll.
	ended: Option<(ui::design::Level, &'static str)>,
	status: &'static str,
	/// GPU texture behind this stream's tile.
	texture: crate::gpu_video::VideoTexture,
	/// Only the listened-to view feeds the one playback device; the rest stay video-only.
	audible: discord_voice::Audible,
}
/// What one view asks of the manager after a poll.
struct Polled {
	command: Option<Command>,
	/// False once the view is finished or no longer chosen, so the manager drops it.
	keep: bool,
}
impl View {
	fn new(streamer: Id) -> Self {
		Self {
			streamer,
			pending: None,
			live: None,
			ended: None,
			status: "",
			texture: Default::default(),
			audible: Default::default(),
		}
	}
	fn stop(&mut self) {
		self.pending = None;
		self.ended = None;
		if let Some(live) = self.live.take() {
			live.task.abort();
		}
		// Logout stops polling, so release the GPU picture here; disposal waits a frame.
		self.texture = Default::default();
	}
	fn context(&self) -> Option<Context> {
		self.pending
			.as_ref()
			.map(|pending| pending.context)
			.or_else(|| self.live.as_ref().map(|live| live.context))
	}
	/// Take negotiation secrets before the UI reduces the event. Nothing is persisted.
	fn observe(&mut self, state: &State, event: &mut Event) {
		let Event::Voice(voice::Event::Watch {
			channel,
			request,
			stream_request,
			streamer,
			event,
		}) = event
		else {
			return;
		};
		let Some(context) = self.context() else {
			return;
		};
		if context.generation != state.generation
			|| (
				context.channel,
				context.request,
				context.stream_request,
				context.streamer,
			) != (*channel, *request, *stream_request, *streamer)
		{
			return;
		}
		match event {
			screen::Event::Created {
				rtc_server,
				rtc_channel,
			} => {
				if let Some(pending) = &mut self.pending {
					pending.rtc = Some((*rtc_server, *rtc_channel));
				}
			}
			screen::Event::Server { token, endpoint } => {
				if let Some(pending) = &mut self.pending {
					match (token.take(), endpoint.take()) {
						(Some(token), Some(endpoint)) => pending.server = Some((token, endpoint)),
						// A null endpoint means Discord is still allocating the stream server.
						(_, None) => pending.server = None,
						(None, Some(_)) => {
							self.ended = Some((
								ui::design::Level::Warning,
								"Discord omitted the stream connection token",
							));
						}
					}
				}
			}
			screen::Event::Deleted { reason } => {
				self.ended = Some((
					ui::design::Level::Info,
					reason.unwrap_or("The stream ended"),
				));
			}
			screen::Event::Failed(message) => {
				self.ended = Some((ui::design::Level::Warning, message));
			}
			screen::Event::Viewers(_) => {}
		}
	}
	/// `wanted` is the call this view should belong to while its streamer stays chosen.
	#[allow(clippy::too_many_arguments)] // The per-view inputs the manager derives once per poll.
	fn poll(
		&mut self,
		runtime: &Runtime,
		state: &mut State,
		ui: &mut ui::MessagingUi,
		ctx: &egui::Context,
		call: Option<super::screen::Call<'_>>,
		audio: Option<std::sync::mpsc::SyncSender<discord_voice::PlaybackFrame>>,
		wanted: Option<(u64, Id, u64)>,
		audible: bool,
		visible: bool,
		sequence: &mut u64,
	) -> Polled {
		self.audible.store(audible, Ordering::Relaxed);
		let wanted = wanted
			.map(|(generation, channel, request)| (generation, channel, request, self.streamer));
		let mut command = None;
		if let Some(context) = self.context()
			&& wanted
				!= Some((
					context.generation,
					context.channel,
					context.request,
					context.streamer,
				)) {
			self.stop();
			if context.generation == state.generation {
				command = Some(Command::Voice(voice::Command::StopWatching {
					channel: context.channel,
					request: context.request,
					stream_request: context.stream_request,
				}));
			}
			return Polled {
				command,
				keep: false,
			};
		}
		if let Some((level, message)) = self.ended.take() {
			let context = self.context();
			self.stop();
			self.status = message;
			// Said once; the stage keeps no line for a view that already ended.
			ui.toasts.push(level, message);
			state.stop_watching(self.streamer);
			let command = context
				.filter(|context| context.generation == state.generation)
				.map(|context| {
					Command::Voice(voice::Command::StopWatching {
						channel: context.channel,
						request: context.request,
						stream_request: context.stream_request,
					})
				});
			return Polled {
				command,
				keep: false,
			};
		}
		let Some((generation, channel, request, streamer)) = wanted else {
			return Polled {
				command: None,
				keep: false,
			};
		};
		if self.context().is_none() {
			let Some(call) = call.filter(|call| {
				call.generation == generation && call.channel == channel && call.request == request
			}) else {
				self.status = "Connect the call before watching a stream";
				return Polled {
					command: None,
					keep: true,
				};
			};
			*sequence = sequence.wrapping_add(1);
			let context = Context {
				generation,
				channel,
				request,
				stream_request: *sequence,
				streamer,
			};
			self.pending = Some(Pending {
				context,
				user: call.user,
				peer: call.peer,
				session: Zeroizing::new(call.session.to_owned()),
				identity: call.identity,
				audio,
				rtc: None,
				server: None,
				started: Instant::now(),
			});
			self.status = "Requesting the stream…";
			return Polled {
				command: Some(Command::Voice(voice::Command::WatchStream {
					channel,
					request,
					stream_request: context.stream_request,
					streamer,
				})),
				keep: true,
			};
		}
		if let Some(pending) = &self.pending {
			if pending.started.elapsed() >= SIGNAL_TIMEOUT {
				self.ended = Some((
					ui::design::Level::Warning,
					"Discord did not provide the stream connection; try again",
				));
				ctx.request_repaint();
			} else {
				ctx.request_repaint_after(SIGNAL_TIMEOUT.saturating_sub(pending.started.elapsed()));
			}
		}
		if self
			.pending
			.as_ref()
			.is_some_and(|pending| pending.rtc.is_some() && pending.server.is_some())
		{
			let pending = self.pending.take().expect("complete stream negotiation");
			if let Err(error) = self.start(runtime, pending, ctx) {
				self.ended = Some((ui::design::Level::Warning, error));
			}
		}
		if let Some(live) = &mut self.live {
			match *live.events.borrow_and_update() {
				Some(Notice::Status(status)) => self.status = status,
				Some(Notice::Failed(error)) => {
					self.ended = Some((ui::design::Level::Warning, error));
				}
				None => {}
			}
			if live.task.is_finished() && self.ended.is_none() {
				self.ended = Some((ui::design::Level::Warning, "The stream connection ended"));
			}
			// Uploading copies the planes into GPU staging, so it runs under the slot lock
			// and the decoder keeps refilling the same allocation. A stream whose tile is not
			// on screen keeps decoding but skips the upload; the picture stays flagged fresh.
			if let Ok(mut slot) = live.frames.try_lock() {
				if let Some(deadline) = slot.advance(Instant::now()) {
					ctx.request_repaint_after(deadline.saturating_duration_since(Instant::now()));
				}
				if visible
					&& let Some((picture, fresh)) = slot.latest.as_mut()
					&& std::mem::take(fresh)
					&& self
						.texture
						.upload(ctx, "remote-stream", picture.borrow())
						.is_some()
				{
					self.status = "Watching the stream";
				}
			}
		} else {
			self.texture = Default::default();
		}
		Polled {
			command: None,
			keep: true,
		}
	}
	fn start(
		&mut self,
		runtime: &Runtime,
		mut pending: Pending,
		ctx: &egui::Context,
	) -> Result<(), &'static str> {
		let (rtc_server, rtc_channel) = pending.rtc.take().ok_or("Missing stream RTC identity")?;
		let (token, endpoint) = pending.server.take().ok_or("Missing stream server")?;
		let session = voice::Secret::new(pending.session.to_string())
			.map_err(|_| "Invalid stream voice session")?;
		let credentials = voice::VoiceConnection {
			channel: rtc_channel,
			guild: Some(rtc_server),
			user: pending.user,
			peer: pending.peer,
			session,
			token,
			endpoint,
			request: pending.context.stream_request,
		};
		let frames = Arc::new(Mutex::new(Scheduled::default()));
		let slot = frames.clone();
		let wake = ctx.clone();
		let streamer = pending.context.streamer.0;
		let sink: discord_voice::VideoSink = Arc::new(move |frame: discord_voice::RemoteFrame| {
			if frame.user != streamer {
				return;
			}
			if slot.lock().is_ok_and(|mut slot| slot.store(frame)) {
				wake.request_repaint();
			}
		});
		let (send, events) = watch::channel(None);
		let wake = ctx.clone();
		let identity = pending.identity;
		let audio = pending.audio;
		let audible = self.audible.clone();
		let task = runtime.spawn(async move {
			let (status_send, status_wake) = (send.clone(), wake.clone());
			let result = discord_voice::watch_stream(
				credentials,
				identity,
				sink,
				audio,
				audible,
				move |event| {
					let status = match event {
						Status::Connecting => "Connecting to the stream…",
						Status::Discovering => "Checking the stream network…",
						Status::TransportReady | Status::Securing => "Securing the stream…",
						Status::WaitingForPeer => "Waiting for the streamer…",
						Status::Ready { .. } => "Stream secured · waiting for video",
						Status::RemoteAudio
						| Status::Speaking(_)
						| Status::CameraAvailable(_)
						| Status::Latency(_) => {
							return Ok(());
						}
					};
					status_send.send_replace(Some(Notice::Status(status)));
					status_wake.request_repaint();
					Ok(())
				},
			)
			.await;
			if let Err(error) = result {
				send.send_replace(Some(Notice::Failed(error)));
			}
			wake.request_repaint();
		});
		self.status = "Connecting to the stream…";
		self.live = Some(Live {
			context: pending.context,
			task,
			events,
			frames,
		});
		Ok(())
	}
}
impl Drop for View {
	fn drop(&mut self) {
		self.stop();
	}
}

/// Desktop ownership for watching other participants' screen shares: one independent
/// [`View`] per chosen stream, up to `voice::MAX_WATCHED`.
#[derive(Default)]
pub(super) struct Watch {
	views: Vec<View>,
	/// Unique per view, so Discord's replies and our stops always name one connection.
	sequence: u64,
}
impl Watch {
	pub fn stop(&mut self) {
		for view in &mut self.views {
			view.stop();
		}
		self.views.clear();
	}
	/// Take negotiation secrets before the UI reduces the event. Nothing is persisted.
	pub fn observe(&mut self, state: &State, event: &mut Event) {
		for view in &mut self.views {
			view.observe(state, event);
		}
	}
	/// Poll every view and publish their pictures to the stage. At most one command goes out
	/// per call; the remaining work finishes on the next frame.
	pub fn poll(
		&mut self,
		runtime: &Runtime,
		state: &mut State,
		ui: &mut ui::MessagingUi,
		ctx: &egui::Context,
		call: Option<super::screen::Call<'_>>,
		audio: Option<std::sync::mpsc::SyncSender<discord_voice::PlaybackFrame>>,
	) -> Option<Command> {
		if state.demo {
			// Offline previews are synthetic: the demo setup owns `ui.voice_streams`.
			self.stop();
			return None;
		}
		let active = state
			.voice
			.active
			.as_ref()
			.filter(|active| active.phase != voice::Phase::Failed);
		// The separate stream transports survive call rekeys and voice resumption.
		let key = active.map(|active| (state.generation, active.channel, active.request));
		let watching = active.map_or_else(Vec::new, |active| active.watching.clone());
		for &streamer in &watching {
			if self.views.len() < voice::MAX_WATCHED
				&& !self.views.iter().any(|view| view.streamer == streamer)
			{
				self.views.push(View::new(streamer));
			}
		}
		let listening = ui.voice_audible_stream(state);
		let frame = ctx.cumulative_frame_nr();
		let mut command = None;
		let mut index = 0;
		while index < self.views.len() && command.is_none() {
			let streamer = self.views[index].streamer;
			let visible = ui
				.voice_streams
				.iter()
				.find(|shown| shown.user == streamer)
				.is_none_or(|shown| frame.saturating_sub(shown.painted) <= 3);
			let polled = self.views[index].poll(
				runtime,
				state,
				ui,
				ctx,
				call.clone(),
				audio.clone(),
				key.filter(|_| watching.contains(&streamer)),
				listening == Some(streamer),
				visible,
				&mut self.sequence,
			);
			command = polled.command;
			if polled.keep {
				index += 1;
			} else {
				self.views.remove(index);
			}
		}
		if command.is_some() || index < self.views.len() {
			// Views after the one that issued a command are polled next frame.
			ctx.request_repaint();
		}
		let previous = std::mem::take(&mut ui.voice_streams);
		ui.voice_streams = self
			.views
			.iter()
			.map(|view| ui::StreamView {
				user: view.streamer,
				texture: view.texture.texture(),
				status: view.status,
				painted: previous
					.iter()
					.find(|shown| shown.user == view.streamer)
					.map_or(frame, |shown| shown.painted),
			})
			.collect();
		command
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn view_with_live(
		runtime: &Runtime,
		state: &State,
		streamer: Id,
		stream_request: u64,
	) -> (View, Context) {
		let call = state.voice.active.as_ref().unwrap();
		let context = Context {
			generation: state.generation,
			channel: call.channel,
			request: call.request,
			stream_request,
			streamer,
		};
		let (_send, events) = watch::channel(None);
		let mut view = View::new(streamer);
		view.live = Some(Live {
			context,
			task: runtime.spawn(std::future::pending()),
			events,
			frames: Arc::new(Mutex::new(Scheduled::default())),
		});
		(view, context)
	}

	fn poll(
		watch: &mut Watch,
		runtime: &Runtime,
		state: &mut State,
		ui: &mut ui::MessagingUi,
		ctx: &egui::Context,
	) -> Option<Command> {
		watch.poll(runtime, state, ui, ctx, None, None)
	}

	#[test]
	fn watch_survives_call_recovery_until_its_selection_ends() {
		let runtime = tokio::runtime::Builder::new_current_thread()
			.enable_all()
			.build()
			.unwrap();
		let mut state = test_support::voice_demo_state();
		state.demo = false;
		state.watch_stream(Id(3)).unwrap();
		let mut ui = ui::MessagingUi::default();
		let ctx = egui::Context::default();
		for failed in [false, true] {
			state.voice.active.as_mut().unwrap().watching = vec![Id(3)];
			let (view, context) = view_with_live(&runtime, &state, Id(3), 2);
			let mut watch = Watch {
				views: vec![view],
				sequence: 0,
			};
			for phase in [
				voice::Phase::Connected,
				voice::Phase::Securing,
				voice::Phase::ConnectingTransport,
				voice::Phase::OpeningAudio,
				voice::Phase::Waiting,
				voice::Phase::Connected,
			] {
				state.voice.active.as_mut().unwrap().phase = phase;
				assert!(poll(&mut watch, &runtime, &mut state, &mut ui, &ctx).is_none());
				assert!(watch.views[0].live.is_some());
			}
			if failed {
				state.voice.active.as_mut().unwrap().phase = voice::Phase::Failed;
			} else {
				state.stop_watching(Id(3));
			}
			assert!(matches!(
				poll(&mut watch, &runtime, &mut state, &mut ui, &ctx),
				Some(Command::Voice(voice::Command::StopWatching {
					channel, request, stream_request: 2,
				})) if channel == context.channel && request == context.request
			));
			assert!(watch.views.is_empty());
			assert!(ui.voice_streams.is_empty());
		}
	}

	#[test]
	fn closing_or_losing_one_stream_leaves_the_other_views_connected() {
		let runtime = tokio::runtime::Builder::new_current_thread()
			.enable_all()
			.build()
			.unwrap();
		let mut state = test_support::voice_demo_state();
		state.demo = false;
		state.voice.active.as_mut().unwrap().watching = vec![Id(3), Id(4), Id(5)];
		let mut watch = Watch::default();
		for (streamer, request) in [(3, 7), (4, 8), (5, 9)] {
			watch
				.views
				.push(view_with_live(&runtime, &state, Id(streamer), request).0);
		}
		let mut ui = ui::MessagingUi::default();
		let ctx = egui::Context::default();
		assert!(poll(&mut watch, &runtime, &mut state, &mut ui, &ctx).is_none());
		assert_eq!(
			ui.voice_streams
				.iter()
				.map(|view| view.user)
				.collect::<Vec<_>>(),
			[Id(3), Id(4), Id(5)]
		);
		// Closing the middle stream stops only that connection, by its own request number.
		state.stop_watching(Id(4));
		assert!(matches!(
			poll(&mut watch, &runtime, &mut state, &mut ui, &ctx),
			Some(Command::Voice(voice::Command::StopWatching {
				stream_request: 8,
				..
			}))
		));
		assert_eq!(
			watch
				.views
				.iter()
				.map(|view| view.streamer)
				.collect::<Vec<_>>(),
			[Id(3), Id(5)]
		);
		assert!(watch.views.iter().all(|view| view.live.is_some()));
		// A stream that fails ends only itself; its neighbours keep their connections.
		watch.views[0].ended = Some((ui::design::Level::Warning, "The stream ended"));
		assert!(matches!(
			poll(&mut watch, &runtime, &mut state, &mut ui, &ctx),
			Some(Command::Voice(voice::Command::StopWatching {
				stream_request: 7,
				..
			}))
		));
		assert_eq!(state.voice.active.as_ref().unwrap().watching, [Id(5)]);
		assert_eq!(watch.views.len(), 1);
		assert!(watch.views[0].live.is_some());
	}

	#[test]
	fn only_the_listened_to_stream_may_feed_playback() {
		let runtime = tokio::runtime::Builder::new_current_thread()
			.enable_all()
			.build()
			.unwrap();
		let mut state = test_support::voice_demo_state();
		state.demo = false;
		state.voice.active.as_mut().unwrap().watching = vec![Id(3), Id(4)];
		let mut watch = Watch::default();
		for (streamer, request) in [(3, 7), (4, 8)] {
			watch
				.views
				.push(view_with_live(&runtime, &state, Id(streamer), request).0);
		}
		let mut ui = ui::MessagingUi::default();
		let ctx = egui::Context::default();
		let flags = |watch: &Watch| {
			watch
				.views
				.iter()
				.map(|view| view.audible.load(Ordering::Relaxed))
				.collect::<Vec<_>>()
		};
		poll(&mut watch, &runtime, &mut state, &mut ui, &ctx);
		assert_eq!(
			flags(&watch),
			[true, false],
			"the oldest stream plays by default"
		);
		ui.voice_stream_listen = ui::StreamListen::Stream(Id(4));
		poll(&mut watch, &runtime, &mut state, &mut ui, &ctx);
		assert_eq!(flags(&watch), [false, true]);
		ui.voice_stream_listen = ui::StreamListen::Silent;
		poll(&mut watch, &runtime, &mut state, &mut ui, &ctx);
		assert_eq!(flags(&watch), [false, false]);
	}

	fn frame(width: u32, height: u32, data: &[u8]) -> discord_voice::RemoteFrame<'_> {
		discord_voice::RemoteFrame {
			presentation: None,
			user: 7,
			picture: platform::video::Picture {
				width,
				height,
				layout: platform::video::Layout::I420,
				color: platform::video::Color::WEBRTC,
				rotation: 0,
				data,
			},
		}
	}

	fn i420(width: u32, height: u32, value: u8) -> Vec<u8> {
		vec![value; platform::video::Layout::I420.len(width, height).unwrap()]
	}

	#[test]
	fn watch_reuses_the_frame_buffer_and_flags_only_updates() {
		let mut slot = None;
		assert!(store_frame(&mut slot, frame(2, 2, &i420(2, 2, 1))));
		let planes = slot.as_ref().unwrap().0.data.as_ptr();
		for value in 0..20 {
			slot.as_mut().unwrap().1 = false;
			assert!(store_frame(&mut slot, frame(2, 2, &i420(2, 2, value))));
			let (picture, fresh) = slot.as_ref().unwrap();
			assert!(*fresh);
			assert_eq!(picture.data.as_ptr(), planes);
			assert_eq!(picture.data, i420(2, 2, value));
		}
		// A smaller picture reuses the allocation too.
		assert!(store_frame(&mut slot, frame(1, 1, &i420(1, 1, 9))));
		let (picture, _) = slot.as_ref().unwrap();
		assert_eq!(picture.data.as_ptr(), planes);
		assert_eq!((picture.width, picture.height), (1, 1));
	}

	#[test]
	fn watch_invalid_frames_preserve_the_last_valid_picture() {
		let mut slot = None;
		assert!(store_frame(&mut slot, frame(1, 1, &i420(1, 1, 4))));
		let planes = slot.as_ref().unwrap().0.data.as_ptr();
		for invalid in [
			frame(0, 1, &[]),
			frame(1, 0, &[]),
			frame(1, 1, &[0; 2]),
			frame(u32::MAX, u32::MAX, &[]),
		] {
			assert!(!store_frame(&mut slot, invalid));
			let (picture, _) = slot.as_ref().unwrap();
			assert_eq!(picture.data.as_ptr(), planes);
			assert_eq!((picture.width, picture.height), (1, 1));
		}
	}

	#[test]
	fn watch_releases_large_capacity_after_a_resolution_drop() {
		let mut slot = None;
		assert!(store_frame(
			&mut slot,
			frame(1920, 1080, &i420(1920, 1080, 0))
		));
		assert!(store_frame(&mut slot, frame(16, 16, &i420(16, 16, 0))));
		let (picture, _) = slot.take().unwrap();
		assert_eq!(picture.data.len(), 16 * 16 * 3 / 2);
		assert!(picture.data.capacity() <= 16 * 16 * 6);
	}

	#[test]
	fn watch_resolution_growth_does_not_double_the_capacity() {
		let mut slot = None;
		assert!(store_frame(
			&mut slot,
			frame(1600, 900, &i420(1600, 900, 0))
		));
		assert!(store_frame(
			&mut slot,
			frame(1920, 1080, &i420(1920, 1080, 0))
		));
		let (picture, _) = slot.take().unwrap();
		assert_eq!(picture.data.len(), 1920 * 1080 * 3 / 2);
		assert!(picture.data.capacity() <= 1920 * 1080 * 3 / 2);
	}
	#[test]
	fn decoded_stream_pictures_wait_for_presentation_and_pressure_skips_only_decoded_frames() {
		let now = Instant::now();
		let mut queue = Scheduled::default();
		let pixels = i420(3840, 2160, 4);
		for n in 0..16 {
			let mut picture = frame(3840, 2160, &pixels);
			picture.presentation = Some(discord_voice::Presentation::new(
				now + Duration::from_millis(100 + n * 16),
			));
			assert!(queue.store(picture));
		}
		assert_eq!(queue.pending.len(), 8);
		assert!(queue.bytes <= DECODED_BUDGET);
		assert!(queue.latest.is_none());
		assert_eq!(queue.advance(now), Some(now + Duration::from_millis(228)));
		queue.advance(now + Duration::from_millis(260));
		assert_eq!(queue.pending.len(), 5);
		assert!(queue.latest.as_ref().unwrap().1);
		assert!(queue.bytes + queue.latest.as_ref().unwrap().0.data.capacity() <= DECODED_BUDGET);
		drop(queue);
		// A backend's RGBA fallback reaches the byte bound before the eight-picture bound.
		let mut queue = Scheduled::default();
		let pixels = vec![4; 3840 * 2160 * 4];
		for n in 0..16 {
			let mut picture = frame(3840, 2160, &pixels);
			picture.picture.layout = platform::video::Layout::Rgba;
			picture.presentation = Some(discord_voice::Presentation::new(
				now + Duration::from_millis(100 + n * 16),
			));
			assert!(queue.store(picture));
		}
		assert_eq!(queue.pending.len(), 4);
		assert!(queue.bytes <= DECODED_BUDGET);
		queue.advance(now + Duration::from_secs(1));
		assert!(queue.pending.is_empty());
		assert_eq!(queue.bytes, 0);
		assert!(queue.latest.as_ref().unwrap().1);
	}
}
