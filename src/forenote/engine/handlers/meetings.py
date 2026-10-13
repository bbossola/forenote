"""RPC handlers for meeting.* and transcript.* methods."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from forenote.audio.devices import (
    Defaults,
    DefaultsChanged,
    OutputAdded,
    OutputRemoved,
    WatcherState,
    monitor_of,
    query_defaults,
)
from forenote.audio.sources import FileAudioSource, PulseSource
from forenote.meeting.recorder import Recorder, RecorderError
from forenote.models.manifest import ManifestError, for_tier
from forenote.rpc.protocol import APP_ERROR, RpcError
from forenote.scheduler import Priority
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

ENGINE_STOP_DRAIN_S = 10.0


def pulse_device(defaults: Defaults | None, speaker: str) -> str:
    """The device that records a speaker. Use the server defaults when a name is not known."""
    if speaker == "me":
        return defaults.source if defaults and defaults.source else "@DEFAULT_SOURCE@"
    return monitor_of(defaults.sink) if defaults and defaults.sink else "@DEFAULT_MONITOR@"


class MeetingsHandlers:
    # Recording

    async def _meeting_start(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_not_closed()
        if self._recorder is not None or self._starting:
            raise RpcError(APP_ERROR, "A recording already runs.")
        debug_file = params.get("debug_file")
        if debug_file is not None:
            if not self.debug_sources:
                raise RpcError(APP_ERROR, "Debug sources are off. Start the engine with --debug-sources.")
            if not Path(debug_file).is_absolute() or not Path(debug_file).is_file():
                raise RpcError(APP_ERROR, "The debug file must be an absolute path to a WAV file.")
        elif self.hardware.audio_server is None:
            raise RpcError(APP_ERROR, "Recording is not available: no audio server.")
        # From here to the end, a second start gets an error, and Engine.stop() waits.
        self._starting = True
        self._start_idle.clear()
        try:
            return await self._start_recording(params, debug_file)
        finally:
            self._starting = False
            self._start_idle.set()

    def _check_not_closed(self) -> None:
        if self._closed:
            raise RpcError(APP_ERROR, "The engine is stopping.")

    async def _start_recording(self, params: dict[str, Any], debug_file: str | None) -> dict[str, Any]:
        transcriber = await self._speech_transcriber()
        self._check_not_closed()
        if debug_file is None and self._audio_defaults is None:
            try:
                self._audio_defaults = await run_in_daemon_thread(query_defaults)
            except Exception as e:  # noqa: BLE001 - no server, or libpulse missing
                raise RpcError(APP_ERROR, f"Recording is not available: {type(e).__name__}.") from None
            self._check_not_closed()

        def factory(speaker: str):
            if debug_file is not None:
                return FileAudioSource(speaker, Path(debug_file), realtime=self.debug_realtime)
            return PulseSource(speaker, pulse_device(self._audio_defaults, speaker))

        recorder = Recorder(
            self.meetings,
            transcriber,
            factory,
            ("others",) if debug_file is not None else ("me", "others"),
            self._emit_event,
            self.settings.get("silence_timeout_min") * 60,
            on_finished=self._recording_finished,
            final_status=self._final_status,
        )
        try:
            lost = await recorder.open()
        except RecorderError:
            raise RpcError(APP_ERROR, "No audio source could be opened.") from None
        if self._closed:
            await recorder.close()
            self._check_not_closed()
        start_reason = params.get("start_reason", "manual")
        app = params.get("app") if start_reason == "detected" else None
        title = params.get("title") or self._default_title(start_reason, app)
        meeting_id = None
        try:
            meeting_id = self.meetings.create(title, start_reason)
            self._set_detector_recording(True)
            await recorder.begin(meeting_id, lost)
        except Exception as e:  # noqa: BLE001 - report the type, keep no half recording
            log.error("the recording cannot start: %s", type(e).__name__)
            await self._undo_start(recorder, meeting_id)
            raise RpcError(APP_ERROR, f"The recording cannot start: {type(e).__name__}.") from None
        # begin() has no await on success today. This guard stays for a future await in begin().
        if self._closed:  # the engine stops: do not leave a recording behind
            await self._undo_start(recorder, meeting_id)
            self._check_not_closed()
        self._recorder = recorder
        self.scheduler.pause(Priority.NOTES)
        self._start_live_reading(recorder)
        self._recording_debug = debug_file is not None
        return {"meeting_id": meeting_id, "started_at": self.meetings.get(meeting_id)["started_at"]}

    async def _undo_start(self, recorder: Recorder, meeting_id: int | None) -> None:
        """Clean up a start that failed after open(). Spec 8.1: no meeting record stays."""
        self._stop_live_reading()  # a guard: today the live start is the last step of a start
        if meeting_id is None:
            await recorder.close()  # begin() did not run
        else:
            # stop() works after a failed begin(): it stops what begin() started.
            await recorder.stop("error", drain_s=ENGINE_STOP_DRAIN_S)
            try:
                self.meetings.delete(meeting_id)
            except Exception as e:  # noqa: BLE001 - keep the first error for the caller
                log.error("cannot delete the meeting: %s", type(e).__name__)
        self._recorder = None
        self._set_detector_recording(False)
        if not recorder.worker_stopped:
            log.warning("the ASR thread did not stop; the speech model loads again next time")
            self._transcriber = None

    async def _meeting_stop(self, params: dict[str, Any]) -> dict[str, Any]:
        recorder = self._recorder
        if recorder is None:
            if self._starting:
                raise RpcError(APP_ERROR, "A recording is starting. Try again in a moment.")
            raise RpcError(APP_ERROR, "No recording runs.")
        status = await recorder.stop("manual")
        return {"meeting_id": recorder.meeting_id, "status": status}

    async def _meeting_keep_recording(self, params: dict[str, Any]) -> dict[str, bool]:
        return {"reset": self._recorder.keep() if self._recorder is not None else False}

    async def _meeting_dismiss_detection(self, params: dict[str, Any]) -> None:
        self._detector.dismiss(params["app"])
        return None

    def _recording_finished(self, status: str) -> None:
        self._stop_live_reading()
        recorder, self._recorder = self._recorder, None
        self._recording_debug = False
        self._set_detector_recording(False)
        if recorder is not None and not recorder.worker_stopped:
            log.warning("the ASR thread did not stop; the speech model loads again next time")
            self._transcriber = None
        self.scheduler.resume(Priority.NOTES)
        if status == "processing" and recorder is not None:
            mid = recorder.meeting_id
            try:
                self._queue_notes(mid, False)
            except Exception as e:  # noqa: BLE001 - for example the scheduler stops: no job can run now
                log.error("the notes job cannot start: %s", type(e).__name__)
                # Send the events of a failed job, so that a waiting client does not wait forever.
                if self.meetings.end_processing(mid):
                    self._emit_event("meeting.state", {"meeting_id": mid, "status": "done"})
                    self._emit_event("notes.done", {"meeting_id": mid, "status": "failed"})

    async def _stop_recording_for_shutdown(self) -> None:
        if self._recorder is not None:
            await self._recorder.stop("manual", drain_s=ENGINE_STOP_DRAIN_S, error_if_not_drained=True)

    async def _speech_transcriber(self):
        tier = self.tier()
        try:
            entry = next((e for e in for_tier(self.manifest, tier) if e.role == "speech"), None) if tier else None
        except ManifestError:
            entry = None
        if entry is None or not self.models.is_installed(entry):
            raise RpcError(APP_ERROR, "The speech model is missing. Run: forenote-cli models download.")
        beam = self.settings.get("asr_final_beam_size")
        key = (entry.id, beam)
        if self._transcriber is None or self._transcriber_key != key:
            model_dir, threads = self.models.path(entry), self.hardware.asr_threads
            try:
                self._transcriber = await run_in_daemon_thread(lambda: self.transcriber_factory(model_dir, threads, beam))
            except Exception as e:  # noqa: BLE001 - out of memory, broken files
                raise RpcError(APP_ERROR, f"The speech model cannot load: {type(e).__name__}.") from None
            self._transcriber_key = key
        return self._transcriber

    @staticmethod
    def _default_title(start_reason: str, app: str | None = None) -> str:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"{app} call, {stamp}" if app else f"Meeting, {stamp}"

    # Stored meetings

    async def _meeting_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self.meetings.list(params.get("limit", 50), params.get("before_id"))

    async def _meeting_get(self, params: dict[str, Any]) -> dict[str, Any]:
        m = self.meetings.get(params["id"])
        if m is None:
            raise RpcError(APP_ERROR, "No such meeting.")
        return m

    async def _meeting_rename(self, params: dict[str, Any]) -> None:
        if not self.meetings.rename(params["id"], params["title"]):
            raise RpcError(APP_ERROR, "No such meeting.")
        return None

    async def _meeting_delete(self, params: dict[str, Any]) -> None:
        if self._recorder is not None and self._recorder.meeting_id == params["id"]:
            raise RpcError(APP_ERROR, "Stop the recording first.")
        if not self.meetings.delete(params["id"]):
            raise RpcError(APP_ERROR, "No such meeting.")
        return None

    async def _transcript_get(self, params: dict[str, Any]) -> dict[str, Any]:
        mid = params["meeting_id"]
        if self.meetings.get(mid) is None:
            raise RpcError(APP_ERROR, "No such meeting.")
        return {
            "lines": self.meetings.lines(mid, params.get("after_id"), params.get("limit", 1000)),
            "markers": self.meetings.markers(mid),
        }

    # Device events (loop thread)

    def _on_device_event(self, event) -> None:
        if self._closed:
            return  # a late event after the engine stopped
        # On WatcherState(connected=False), keep the last known defaults.
        # The watcher sends DefaultsChanged again after it reconnects.
        if isinstance(event, DefaultsChanged):
            old, self._audio_defaults = self._audio_defaults, event.defaults
            if self._recorder is not None and old is not None and not self._recording_debug:
                if old.source != event.defaults.source:
                    self._recorder.on_default_changed("me")
                if old.sink != event.defaults.sink:
                    self._recorder.on_default_changed("others")
        elif isinstance(event, OutputAdded):
            # The detector always tracks the streams: a closed call app shortens the
            # silence time also with detection off (parent spec 8.6).
            self._sync_detector()
            app = self._detector.output_added(event.index, event.app, event.binary)
            if app:
                self._spawn(self.rpc.broadcast("meeting.detected", {"app": app}))
        elif isinstance(event, OutputRemoved):
            if self._detector.output_removed(event.index) and self._recorder is not None:
                self._recorder.call_app_closed()
        elif isinstance(event, WatcherState):
            # Detection pauses while the watcher has no server. It resyncs the streams on reconnect.
            self._watcher_connected = event.connected
            self._sync_detector()

    def _set_detector_recording(self, recording: bool) -> None:
        self._detector_recording = recording
        self._sync_detector()

    def _detection_on(self) -> bool:
        return self._watcher_connected and self.settings.get("meeting_detection")

    def _sync_detector(self) -> None:
        """Mute the detector during a recording, when detection is off, and while the watcher has no server.

        A muted detector tracks streams but announces nothing and remembers no announcement.
        """
        self._detector.set_recording(self._detector_recording or not self._detection_on())

    def _detector_tick(self) -> None:
        # With detection off or paused, do not tick: a browser stream stays pending
        # and is announced when detection turns on again.
        if not self._detection_on():
            return
        self._sync_detector()
        app = self._detector.tick()
        if app:
            self._spawn(self.rpc.broadcast("meeting.detected", {"app": app}))

    def _on_setting_changed(self, key: str, value: Any) -> None:
        if key == "call_apps":
            self._detector.set_call_apps(value)
