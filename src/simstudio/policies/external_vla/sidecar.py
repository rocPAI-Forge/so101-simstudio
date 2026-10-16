"""Versioned HTTP sidecar client and server for external VLA inference."""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

from .adapters.base import VLAAdapter, load_adapter
from .contract import PROTOCOL_VERSION, ActionChunk, Capabilities, ContractError, ObservationFrame
from .manifest import DeploymentManifest

logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 64 * 1024 * 1024


class SidecarError(RuntimeError):
    """Raised when a sidecar request fails or returns an invalid response."""


class SidecarClient:
    """Thin stdlib HTTP client suitable for a separate Python/CUDA environment."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 10.0,
        jpeg_quality: int = 90,
    ):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self.jpeg_quality = jpeg_quality
        self._capabilities = Capabilities.from_wire(self._request("GET", "/v1/capabilities"))

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=body,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-SimStudio-VLA-Protocol": PROTOCOL_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SidecarError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SidecarError(f"{method} {path} failed: {exc}") from exc
        if not isinstance(result, dict):
            raise SidecarError(f"{method} {path} returned a non-object JSON response")
        return result

    def health(self) -> dict[str, object]:
        return self._request("GET", "/v1/health")

    def reset_episode(self, episode_id: str | None = None) -> None:
        self._request(
            "POST",
            "/v1/reset",
            {"protocol_version": PROTOCOL_VERSION, "episode_id": episode_id},
        )

    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk:
        started = time.perf_counter()
        payload = self._request(
            "POST",
            "/v1/predict",
            observation.to_wire(jpeg_quality=self.jpeg_quality),
        )
        network_ms = (time.perf_counter() - started) * 1000.0
        received_at_ns = time.time_ns()
        chunk = ActionChunk.from_wire(payload)
        timings = dict(chunk.timings_ms)
        timings["round_trip"] = network_ms
        # Server and robot hosts may not share a synchronized wall clock.
        # Staleness is enforced from local receipt; request_timeout_s bounds
        # time spent before receipt.
        return ActionChunk(
            actions=chunk.actions,
            source_timestamp_ns=chunk.source_timestamp_ns,
            generated_at_ns=received_at_ns,
            dt_s=chunk.dt_s,
            joint_names=chunk.joint_names,
            protocol_version=chunk.protocol_version,
            timings_ms=timings,
        )

    def close(self) -> None:
        pass


class _SidecarHTTPServer(ThreadingHTTPServer):
    adapter: VLAAdapter
    adapter_lock: Lock


class _SidecarHandler(BaseHTTPRequestHandler):
    server: _SidecarHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-SimStudio-VLA-Protocol", PROTOCOL_VERSION)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            logger.warning("Client disconnected before the sidecar response was written")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > MAX_REQUEST_BYTES:
            raise ContractError(f"Request body is too large ({length} bytes > {MAX_REQUEST_BYTES} bytes)")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ContractError("Request body must be a JSON object")
        return payload

    def _check_protocol_header(self) -> None:
        version = self.headers.get("X-SimStudio-VLA-Protocol")
        if version != PROTOCOL_VERSION:
            raise ContractError(f"Incompatible protocol header {version!r}; expected {PROTOCOL_VERSION!r}")

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._check_protocol_header()
            if self.path == "/v1/health":
                self._send(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "protocol_version": PROTOCOL_VERSION,
                        "adapter": self.server.adapter.capabilities.adapter_name,
                        "time_ns": time.time_ns(),
                    },
                )
            elif self.path == "/v1/capabilities":
                self._send(HTTPStatus.OK, self.server.adapter.capabilities.to_wire())
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            self._send_error(exc)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._check_protocol_header()
            payload = self._read_json()
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                raise ContractError("Missing or incompatible protocol_version")
            if self.path == "/v1/reset":
                with self.server.adapter_lock:
                    self.server.adapter.reset_episode(payload.get("episode_id"))
                self._send(HTTPStatus.OK, {"ok": True, "protocol_version": PROTOCOL_VERSION})
            elif self.path == "/v1/predict":
                decode_started = time.perf_counter()
                observation = ObservationFrame.from_wire(payload)
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                model_started = time.perf_counter()
                with self.server.adapter_lock:
                    chunk = self.server.adapter.predict_chunk(observation)
                adapter_ms = (time.perf_counter() - model_started) * 1000.0
                chunk.validate(expected_horizon=self.server.adapter.capabilities.action_horizon)
                timings = dict(chunk.timings_ms)
                timings.setdefault("preprocess", decode_ms)
                timings.setdefault("model_and_postprocess", adapter_ms)
                response = ActionChunk(
                    actions=chunk.actions,
                    source_timestamp_ns=chunk.source_timestamp_ns,
                    generated_at_ns=chunk.generated_at_ns,
                    dt_s=chunk.dt_s,
                    joint_names=chunk.joint_names,
                    protocol_version=chunk.protocol_version,
                    timings_ms=timings,
                )
                self._send(HTTPStatus.OK, response.to_wire())
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            self._send_error(exc)

    def _send_error(self, exc: Exception) -> None:
        logger.exception("Sidecar request failed")
        status = (
            HTTPStatus.BAD_REQUEST
            if isinstance(exc, (ValueError, ContractError))
            else HTTPStatus.INTERNAL_SERVER_ERROR
        )
        self._send(status, {"error": type(exc).__name__, "detail": str(exc)})


def create_server(
    manifest: DeploymentManifest,
    *,
    host: str = "127.0.0.1",
    port: int = 10086,
    adapter: VLAAdapter | None = None,
) -> _SidecarHTTPServer:
    manifest.validate()
    loaded = adapter or load_adapter(manifest)
    if loaded.capabilities != manifest.capabilities():
        raise ValueError("Sidecar adapter capabilities do not match manifest")
    server = _SidecarHTTPServer((host, port), _SidecarHandler)
    server.adapter = loaded
    server.adapter_lock = Lock()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an external VLA deployment manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    manifest = DeploymentManifest.load(args.manifest)
    server = create_server(manifest, host=args.host, port=args.port)
    logger.info(
        "Serving %s at http://%s:%d (protocol %s)",
        manifest.adapter,
        args.host,
        args.port,
        PROTOCOL_VERSION,
    )
    try:
        server.serve_forever()
    finally:
        server.adapter.close()
        server.server_close()


if __name__ == "__main__":
    main()
