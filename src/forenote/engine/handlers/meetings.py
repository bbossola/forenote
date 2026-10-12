"""RPC handlers for meeting.* and transcript.* methods."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from forenote.audio.devices import DefaultsChanged, OutputAdded, OutputRemoved, monitor_of, query_defaults
from forenote.audio.sources import FileAudioSource, PulseSource
from forenote.meeting.recorder import Recorder, RecorderError
from forenote.models.manifest import ManifestError, for_tier
from forenote.rpc.protocol import APP_ERROR, RpcError
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

ENGINE_STOP_DRAIN_S = 10.0


class MeetingsHandlers:
    # Recording

    async def _meeting_start(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._recorder is not None:
            raise RpcError(APP_ERROR, "A recording already runs.")
        debug_file = params.get("debug_file")
        if debug_file is not None:
            if not self.debug_sources:
                raise RpcError(APP_ERROR, "Debug sources are off. Start the engine with --debug-sources.")
            if not Path(debug_file).is_absolute() or not Path(debug_file).is_file():
                raise RpcError(APP_ERROR, "The debug file must be an absolute path to a WAV file.")
            speakers = ("others",)
        else:
            if self.hardware.audio_server is None:
                raise RpcError(APP_ERROR, "Recording is not available: no audio server.")
            speakers = ("me", "others")
        transcriber = await self._speech_transcriber()

        def factory(speaker: str):
            if debug_file is not None:
                return FileAudioSource(speaker, Path(debug_file), realtime=self.debug_realtime)
            defaults = self._audio_defaults
            if defaults is None:
                return PulseSource(speaker, "")  # "" opens the server default
            device = defaults.source if speaker == "me" else monitor_of(defaults.sink)
            return PulseSource(speaker, device)

        if debug_file is None and self._audio_defaults is None:
            try:
                self._audio_defaults = await run_in_daemon_thread(query_defaults)
            except Exception as e:  # noqa: BLE001 - no server, or libpulse missing
                raise RpcError(APP_ERROR, f"Recording is not available: {type(e).__name__}.") from None

        # A second start can come while this one waits. Check again.
        if self._recorder is not None or self._starting:
            raise RpcError(APP_ERROR, "A recording already runs.")
        self._starting = True
        try:
            return await self._open_and_begin(transcriber, factory, speakers, params)
        finally:
            self._starting = False

    async def _open_and_begin(self, transcriber, factory, speakers, params: dict[str, Any]) -> dict[str, Any]:
        recorder = Recorder(
            self.meetings,
            transcriber,
            factory,
            speakers,
            self._emit_event,
            self.settings.get("silence_timeout_min") * 60,
            on_finished=self._recording_finished,
        )
        try:
            lost = await recorder.open()
        except RecorderError:
            raise RpcError(APP_ERROR, "No audio source could be opened.") from None
        start_reason = params.get("start_reason", "manual")
        title = params.get("title") or self._default_title(start_reason)
        meeting_id = None
        begun = False
        try:
            meeting_id = self.meetings.create(title, start_reason)
            self._detector.set_recording(True)
            begun = True  # begin() cleans up after its own failure
            await recorder.begin(meeting_id, lost)
        except Exception as e:  # noqa: BLE001 - report the type, keep no half recording
            log.error("the recording cannot start: %s", type(e).__name__)
            if begun:
                # stop() works after a failed begin(): it finishes the row,
                # sends meeting.state "done" and calls on_finished.
                await recorder.stop("error")
            else:
                await recorder.close()
            if meeting_id is not None:
                # A no-op if stop() finished the row. begin() can fail before it sets meeting_id.
                try:
                    self.meetings.finish(meeting_id, "done", "error")
                except Exception as e2:  # noqa: BLE001 - keep the first error for the caller
                    log.error("cannot finish the meeting: %s", type(e2).__name__)
            self._recorder = None
            self._detector.set_recording(False)
            raise RpcError(APP_ERROR, f"The recording cannot start: {type(e).__name__}.") from None
        self._recorder = recorder
        return {"meeting_id": meeting_id, "started_at": self.meetings.get(meeting_id)["started_at"]}

    async def _meeting_stop(self, params: dict[str, Any]) -> dict[str, Any]:
        recorder = self._recorder
        if recorder is None:
            raise RpcError(APP_ERROR, "No recording runs.")
        status = await recorder.stop("manual")
        return {"meeting_id": recorder.meeting_id, "status": status}

    async def _meeting_keep_recording(self, params: dict[str, Any]) -> dict[str, bool]:
        return {"reset": self._recorder.keep() if self._recorder is not None else False}

    async def _meeting_dismiss_detection(self, params: dict[str, Any]) -> None:
        self._detector.dismiss(params["app"])
        return None

    def _recording_finished(self, status: str) -> None:
        recorder, self._recorder = self._recorder, None
        self._detector.set_recording(False)
        if recorder is not None and not recorder.worker_stopped:
            log.warning("the ASR thread did not stop; the speech model loads again next time")
            self._transcriber = None

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
        # On WatcherState(connected=False), keep the last known defaults.
        # The watcher sends DefaultsChanged again after it reconnects.
        if isinstance(event, DefaultsChanged):
            old, self._audio_defaults = self._audio_defaults, event.defaults
            if self._recorder is not None and old is not None:
                if old.source != event.defaults.source:
                    self._recorder.on_default_changed("me")
                if old.sink != event.defaults.sink:
                    self._recorder.on_default_changed("others")
        elif isinstance(event, OutputAdded):
            self._detector.set_call_apps(self.settings.get("call_apps"))
            app = self._detector.output_added(event.index, event.app, event.binary)
            if app and self.settings.get("meeting_detection"):
                self._spawn(self.rpc.broadcast("meeting.detected", {"app": app}))
        elif isinstance(event, OutputRemoved):
            if self._detector.output_removed(event.index) and self._recorder is not None:
                self._recorder.call_app_closed()
