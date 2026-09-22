import http
import http.server
import json
import pathlib
import typing

from neuro_os.domain import Intent
from neuro_os.sources.brainflow import BrainFlowSource
from neuro_os.trials import TrialRecorder


_MAX_JSON_BODY_BYTES = 16_384


def serve_stimulus(
    source: BrainFlowSource,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    static_dir: str | pathlib.Path = "apps/ssvep-stimulus",
    storage_dir: str | pathlib.Path = ".neuros/sessions",
) -> None:
    root = pathlib.Path(static_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"stimulus static directory does not exist: {root}")

    recorder = TrialRecorder(source, storage_dir=storage_dir)
    handler = _make_handler(recorder, root)

    source.open()
    try:
        server = http.server.ThreadingHTTPServer((host, port), handler)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        source.close()


def _make_handler(
    recorder: TrialRecorder,
    static_dir: pathlib.Path,
) -> type[http.server.SimpleHTTPRequestHandler]:
    class StimulusRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def do_GET(self) -> None:
            if self.path == "/api/status":
                self._send_json(
                    http.http.HTTPStatus.OK,
                    {
                        "connected": recorder.source.is_open,
                        "sample_rate_hz": recorder.source.sample_rate_hz,
                        "channels": recorder.source.channel_names,
                        "active_trial_id": (
                            recorder.active_trial.trial_id
                            if recorder.active_trial is not None
                            else None
                        ),
                    },
                )
                return
            super().do_GET()

        def do_POST(self) -> None:
            try:
                payload = self._read_json_body()
                if self.path == "/api/trial/start":
                    intent = Intent(str(payload["intent"]))
                    active = recorder.start(intent)
                    self._send_json(
                        http.HTTPStatus.CREATED,
                        {
                            "trial_id": active.trial_id,
                            "intent": active.intent.value,
                            "start_marker": active.start_marker,
                            "started_at": active.started_at,
                        },
                    )
                    return

                if self.path == "/api/trial/stop":
                    artifact = recorder.stop(
                        completion=str(payload.get("completion", "completed")),
                        client_metadata=dict(payload.get("client_metadata") or {}),
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "trial_id": artifact.trial_id,
                            "intent": artifact.intent,
                            "completion": artifact.completion,
                            "sample_count": artifact.sample_count,
                            "duration_seconds": artifact.duration_seconds,
                            "start_marker_indices": artifact.start_marker_indices,
                            "stop_marker_indices": artifact.stop_marker_indices,
                            "metadata_file": artifact.metadata_file,
                        },
                    )
                    return

                self._send_json(http.HTTPStatus.NOT_FOUND, {"error": "unknown API route"})
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json(http.HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except RuntimeError as exc:
                self._send_json(http.HTTPStatus.CONFLICT, {"error": str(exc)})

        def _read_json_body(self) -> dict[str, typing.Any]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if not 0 < length <= _MAX_JSON_BODY_BYTES:
                raise ValueError("invalid JSON body size")

            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError("JSON body must be an object")
            return value

        def _send_json(self, status: http.HTTPStatus, payload: dict[str, typing.Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

    return StimulusRequestHandler
