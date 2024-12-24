#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiofiles
import asyncio
import io
import os
import sys

import aiohttp
import datetime
import wave
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.whisper import WhisperSTTService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.styletts2 import StyleTTS2Service
# from pipecat.services.dani_onnx import DaniONNXService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyTranscriptionSettings

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
    TURBO = "turbo"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"


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
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(),
                # transcription_enabled=True,
                #
                # Spanish
                #
                # transcription_settings=DailyTranscriptionSettings(
                #     language="ko",
                    # tier="nova",
                    # model="2-general"
                # )
            ),
        )

        stt = WhisperSTTService(
            model=Model.LARGE,
            audio_passthrough=True,
            )

        # tts = ElevenLabsTTSService(
        #     api_key=os.getenv("ELEVENLABS_API_KEY"),
        #     #
        #     # English
        #     #
        #     voice_id="cgSgspJ2msm6clMCkdW9",
        #     aiohttp_session=session,
        #     #
        #     # Spanish
        #     #
        #     # model="eleven_multilingual_v2",
        #     # voice_id="gD1IexrzCvsXPHUuT0s3",
        # )


        tts = StyleTTS2Service(
            base_url='http://localhost:8014',
            aiohttp_session=session,
            sample_rate=24000,
        )

        # tts = DaniONNXService()

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

        messages = [
            {
                "role": "system",
                #
                # English
                #
                # "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. Keep all your response to 12 words or fewer.",
                #
                # Korean
                #
                "content": "당신은 Chatbot이며, 친절하고 도움을 주는 로봇입니다. 목표는 간결하게 능력을 보여주는 것입니다. 출력은 오디오로 변환되므로 특수 문자를 포함하지 마세요. 사용자의 말에 창의적이고 도움이 되도록 답해주세요. 먼저 자신을 소개하세요."
                #
                # Spanish
                #
                # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Save audio every 10 seconds.
        audiobuffer = AudioBufferProcessor(buffer_size=480000)

        pipeline = Pipeline(
            [
                transport.input(),  # microphone
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                # audiobuffer,  # used to buffer the audio in the pipeline
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            await save_audio(audio, sample_rate, num_channels)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
