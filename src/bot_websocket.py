#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.whisper import WhisperSTTService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.styletts2 import StyleTTS2Service
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
    FastAPIWebsocketCallbacks,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

from enum import Enum

class Model(Enum):
    """Class of basic Whisper model selection options"""

    TINY = "tiny"
    BASE = "base"
    MEDIUM = "medium"
    LARGE = "large-v3"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"


import datetime
import io
import wave
import aiofiles


async def save_audio(audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = f"conversation_recording{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        print(f"Merged audio saved to {filename}")
    else:
        print("No audio data to save")



async def main():
    async with aiohttp.ClientSession() as session:
        transport = WebsocketServerTransport(
            host='localhost',
            port=8765,
            params=WebsocketServerParams(
                audio_out_sample_rate=24000,
                audio_out_enabled=True,
                add_wav_header=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            )
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

        # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        stt = WhisperSTTService(
            model=Model.LARGE,
            sample_rate=24000,
            # audio_passthrough=False,
            )
        
        # tts = ElevenLabsTTSService(
        #     api_key=os.getenv("ELEVENLABS_API_KEY"),
        #     output_format="pcm_24000",
            
        #     # English
            
        #     voice_id="cgSgspJ2msm6clMCkdW9",
            
        #     # Spanish
            
        #     # model="eleven_multilingual_v2",
        #     # voice_id="gD1IexrzCvsXPHUuT0s3",
        # )

        tts = StyleTTS2Service(
            base_url='http://localhost:8014',
            aiohttp_session=session,
            sample_rate=24000,
        )

        messages = [
            {
                "role": "system",
                # English
                # "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
                # Korean
                "content": "당신은 Chatbot이며, 친절하고 도움을 주는 로봇입니다. 목표는 간결하게 능력을 보여주는 것입니다. 출력은 오디오로 변환되므로 특수 문자를 포함하지 마세요. 사용자의 말에 창의적이고 도움이 되도록 답해주세요. 먼저 자신을 소개하세요."

            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        audiobuffer = AudioBufferProcessor(
            buffer_size=480000,
            # sample_rate=24000,
            )

        pipeline = Pipeline(
            [
                transport.input(),  # Websocket input from client
                stt,  # Speech-To-Text
                context_aggregator.user(),
                llm,  # LLM
                tts,  # Text-To-Speech
                transport.output(),  # Websocket output to client
                # audiobuffer,
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            await save_audio(audio, sample_rate, num_channels)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
