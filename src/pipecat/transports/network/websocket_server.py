#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import json
import time
import wave
from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class WebsocketServerParams(TransportParams):
    add_wav_header: bool = False
    serializer: FrameSerializer
    session_timeout: Optional[int] = None


class WebsocketServerCallbacks(BaseModel):
    on_client_connected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_client_disconnected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_session_timeout: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_websocket_ready: Callable[[], Awaitable[None]]


class WebsocketServerInputTransport(BaseInputTransport):
    def __init__(
        self,
        host: str,
        port: int,
        params: WebsocketServerParams,
        callbacks: WebsocketServerCallbacks,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        self._server_task = None

        # This task will monitor the websocket connection periodically.
        self._monitor_task = None

        self._stop_server_event = asyncio.Event()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._params.serializer.setup(frame)
        self._server_task = self.create_task(self._server_task_handler())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._stop_server_event.set()
        if self._monitor_task:
            await self.cancel_task(self._monitor_task)
        if self._server_task:
            await self.wait_for_task(self._server_task)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._monitor_task:
            await self.cancel_task(self._monitor_task)
        if self._server_task:
            await self.cancel_task(self._server_task)

    async def _server_task_handler(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}")
        async with websockets.serve(self._client_handler, self._host, self._port) as server:
            await self._callbacks.on_websocket_ready()
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket: websockets.WebSocketServerProtocol, path):
        logger.info(f"New client connection from {websocket.remote_address}")
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client connected, using new connection")

        self._websocket = websocket

        # Notify
        await self._callbacks.on_client_connected(websocket)

        # Create a task to monitor the websocket connection
        if self._params.session_timeout:
            self._monitor_task = self.create_task(
                self._monitor_websocket(websocket, self._params.session_timeout)
            )

        # Handle incoming messages
        try:
            async for message in websocket:
                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await self._websocket.close()
        self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")

    async def _monitor_websocket(
        self, websocket: websockets.WebSocketServerProtocol, session_timeout: int
    ):
        """Wait for session_timeout seconds, if the websocket is still open,
        trigger timeout event."""
        try:
            await asyncio.sleep(session_timeout)
            if not websocket.closed:
                await self._callbacks.on_session_timeout(websocket)
        except asyncio.CancelledError:
            logger.info(f"Monitoring task cancelled for: {websocket.remote_address}")
            raise


class WebsocketServerOutputTransport(BaseOutputTransport):
    def __init__(self, params: WebsocketServerParams, **kwargs):
        super().__init__(params, **kwargs)

        self._params = params

        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # write_raw_audio_frames() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0

    async def set_client_connection(self, websocket: Optional[websockets.WebSocketServerProtocol]):
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client allowed, using new connection")
        self._websocket = websocket

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._params.serializer.setup(frame)
        self._send_interval = (self._audio_chunk_size / self.sample_rate) / 2

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        await self._write_frame(frame)

    async def write_raw_audio_frames(self, frames: bytes):
        if not self._websocket:
            # Simulate audio playback with a sleep.
            await self._write_audio_sleep()
            return

        frame = OutputAudioRawFrame(
            audio=frames,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                wav_frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                frame = wav_frame

        await self._write_frame(frame)

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

    async def _write_frame(self, frame: Frame):
        try:
            payload = await self._params.serializer.serialize(frame)
            if payload and self._websocket:
                await self._websocket.send(payload)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

    async def _write_audio_sleep(self):
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class WebsocketServerTransport(BaseTransport):
    def __init__(
        self,
        params: WebsocketServerParams,
        host: str = "localhost",
        port: int = 8765,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
            on_websocket_ready=self._on_websocket_ready,
        )
        self._input: Optional[WebsocketServerInputTransport] = None
        self._output: Optional[WebsocketServerOutputTransport] = None
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")
        self._register_event_handler("on_websocket_ready")

    def input(self) -> WebsocketServerInputTransport:
        if not self._input:
            self._input = WebsocketServerInputTransport(
                self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> WebsocketServerOutputTransport:
        if not self._output:
            self._output = WebsocketServerOutputTransport(self._params, name=self._output_name)
        return self._output

    async def _on_client_connected(self, websocket):
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        if self._output:
            await self._output.set_client_connection(None)
            await self._call_event_handler("on_client_disconnected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_session_timeout(self, websocket):
        await self._call_event_handler("on_session_timeout", websocket)

    async def _on_websocket_ready(self):
        await self._call_event_handler("on_websocket_ready")
