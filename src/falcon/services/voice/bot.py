"""Voice bot (docs/VOICE_PLAN.md): browser mic -> Silero VAD -> Deepgram Nova-3 -> Falcon runtime -> ElevenLabs -> browser speaker.

The agent is the Falcon runtime itself (/api/chat over SSE with the customer's session token): FalconAgentService is a Pipecat LLM
service that hands each finished user turn to call.CallSession, which decides what is spoken (the four rules). No model runs here.

    PYTHONUTF8=1 PYTHONPATH=src .venv-voice/Scripts/python -m falcon.services.voice.bot -t webrtc    # http://localhost:7860/client

Who is calling: the request body of the call (`{"token", "thread", "lang"}`, the app's Call button in phase 2) or, from Pipecat's own
client page, VOICE_DEMO_USER signed in through the demo switcher (VOICE_RESET=1 puts the demo data back first, like the sidebar).
VOICE_LANG (ar | en) is the call's one language: the STT model language and the voice (Nova-3 Arabic is monolingual).
"""

from __future__ import annotations

import dataclasses
import os
import time
import uuid

import httpx
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (Frame, InputAudioRawFrame, InterimTranscriptionFrame, LLMContextFrame, LLMFullResponseEndFrame,
                                   LLMFullResponseStartFrame, LLMTextFrame, TranscriptionFrame, TTSSpeakFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams, UserTurnStoppedMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from falcon.services.voice.call import PROGRESS_AFTER, STILL_AFTER, CallSession, RuntimeClient

load_dotenv(override=True)

RUNTIME_URL = os.getenv("VOICE_RUNTIME_URL", "http://localhost:8000")
# ElevenLabs voices (the listening test may change them; env wins). Defaults are premade voices: a library voice (e.g. "Sara",
# Modern Standard Arabic, XTa3iQyMA6f1qrI4F6kZ) works only after it is added to the account's voices, else the socket closes 1008.
VOICES = {"ar": os.getenv("ELEVENLABS_VOICE_AR", "EXAVITQu4vr4xnSDxMaL"), "en": os.getenv("ELEVENLABS_VOICE_EN", "XrExE9yKIg1WjnnlVkGX")}
TTS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")  # the low-latency multilingual model; Arabic is in its list
STT_LANG = {"ar": Language.AR, "en": Language.EN}
GREETING = {"ar": "مرحباً، معك فالكون. كيف أساعدك؟", "en": "Hello, this is Falcon. How can I help?"}


# Keyterm prompting (Nova-3): words the call is likely to contain, so "الحجز K3P7QZ" is not heard as "الحاجز الزامد فيدي" (20 Oct).
# The customer's own booking codes and flight numbers are added per call from /api/bookings.
KEYTERMS = {"ar": ["فالكون", "حجز", "الحجز", "إلغاء", "تغيير", "استرداد", "رصيد سفر", "أمتعة", "حقيبة", "رحلة", "الرحلة", "الموفرة",
                   "المرنة", "الخاصة", "دبي", "القاهرة", "الرياض", "الكويت", "لندن", "حامل", "نعم", "لا"],
            "en": ["Falcon", "booking", "cancel", "cancellation", "refund", "travel credit", "baggage", "Saver", "Flex", "Special",
                   "Dubai", "Cairo", "Riyadh", "Kuwait", "London", "pregnant", "yes", "no"]}


async def keyterms(token: str, lang: str) -> list[str]:
    terms = list(KEYTERMS[lang])
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(f"{RUNTIME_URL}/api/bookings", headers={"Authorization": f"Bearer {token}"})
            r.raise_for_status()
        for b in r.json().get("bookings") or []:
            terms.append(b.get("pnr") or "")
            terms += [seg.get("flight_number") or "" for seg in b.get("segments") or []]
    except Exception as exc:  # noqa: BLE001 - keyterms are a hint; the call works without them
        logger.warning(f"keyterms: could not read the bookings: {exc}")
    return list(dict.fromkeys(t for t in terms if t))


def confidence_of(result) -> tuple[float | None, float | None]:
    """(mean, lowest) word confidence of a Deepgram live result, or (None, None) when it carries none."""
    try:
        alt = result.channel.alternatives[0]
        words = [w.confidence for w in (alt.words or []) if getattr(w, "confidence", None) is not None]
        return (round(sum(words) / len(words), 3), round(min(words), 3)) if words else (round(alt.confidence, 3), None)
    except (AttributeError, IndexError, TypeError, ZeroDivisionError):
        return None, None


class ConfidenceLog(FrameProcessor):
    """Logs every final transcript with the service's confidence (Deepgram gives one per word), so a noise threshold is picked from
    real calls, not guessed (20 Oct), and two services can be compared on the same call (scripts/voice_stt_compare.py).
    Nothing is filtered yet. Line format: "voice transcript [<stt>]: confidence mean <m> lowest <l>: <text>"."""

    def __init__(self, stt_name: str):
        super().__init__()
        self._stt = stt_name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            mean, low = confidence_of(frame.result)
            logger.info(f"voice transcript [{self._stt}]: confidence mean {mean} lowest {low}: {frame.text}")
        await self.push_frame(frame, direction)


def make_stt(name: str, lang: str, terms: list[str], shadow: bool = False):
    """The speech-to-text service: VOICE_STT picks the one that drives the call (deepgram, the default; or elevenlabs), and
    VOICE_STT_SHADOW runs another one on the same audio for comparison only (decision #55)."""
    if name == "elevenlabs":
        # Scribe v2 Realtime (a WebSocket, interim and committed transcripts). Driving the call, Pipecat's VAD commits each segment
        # (manual); as a shadow there is no VAD in its pipeline, so ElevenLabs' own VAD commits.
        return ElevenLabsRealtimeSTTService(
            api_key=os.environ["ELEVENLABS_API_KEY"], commit_strategy=CommitStrategy.VAD if shadow else CommitStrategy.MANUAL,
            include_language_detection=True,
            settings=ElevenLabsRealtimeSTTService.Settings(language=STT_LANG[lang], keyterms=terms[:50]))
    return DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"],
                              settings=DeepgramSTTService.Settings(model="nova-3", language=STT_LANG[lang], smart_format=True, keyterm=terms))


class ShadowTee(FrameProcessor):
    """Copies every microphone frame into the shadow STT's own pipeline and passes the original on untouched: the shadow hears
    exactly what the main STT hears, and nothing it does reaches the call."""

    def __init__(self, shadow: PipelineWorker):
        super().__init__()
        self._shadow = shadow

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self._shadow.queue_frame(InputAudioRawFrame(audio=frame.audio, sample_rate=frame.sample_rate, num_channels=frame.num_channels))
        await self.push_frame(frame, direction)


class AudioProbe(FrameProcessor):
    """Diagnostics (VOICE_PROBE=1, FAILURE_DIARY #64): every 2 s the peak input level (0..1), and every transcript Deepgram returns."""

    def __init__(self, where: str):
        super().__init__()
        self._where, self._peak, self._frames, self._t = where, 0.0, 0, time.monotonic()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self._frames += 1
            self._peak = max(self._peak, float(np.abs(np.frombuffer(frame.audio, dtype=np.int16)).max(initial=0)) / 32768)
            if time.monotonic() - self._t >= 2:
                logger.info(f"probe[{self._where}] {self._frames} audio frames, peak level {self._peak:.3f}")
                self._peak, self._frames, self._t = 0.0, 0, time.monotonic()
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            logger.info(f"probe[{self._where}] {type(frame).__name__}: {frame.text}")
        await self.push_frame(frame, direction)


class FalconAgentService(LLMService):
    """The Falcon runtime as a Pipecat LLM service: takes the latest user transcript from the context and hands it to the call.

    Each turn runs in its own task, not in the frame-processing task, so a barge-in (InterruptionFrame) stops the speech but never
    cancels a runtime turn half-way (a cancelled SSE request would abandon a turn between a tool call and its result; decision #50).
    """

    def __init__(self, runtime: RuntimeClient, lang: str, **kwargs):
        # the runtime owns model and prompt: every field is declared empty so Pipecat does not log the settings as incomplete
        super().__init__(settings=LLMSettings(**{f.name: None for f in dataclasses.fields(LLMSettings) if f.name != "extra"}), **kwargs)
        self.call = CallSession(runtime, lang, self._speak, self._observe,
                                progress_after=float(os.getenv("VOICE_PROGRESS_AFTER", PROGRESS_AFTER)),
                                still_after=float(os.getenv("VOICE_STILL_AFTER", STILL_AFTER)))

    async def _speak(self, text: str, kind: str) -> None:
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _observe(self, kind: str, payload: dict) -> None:
        if kind == "spoken":  # the wait the caller sat through, per line (seconds since the turn was sent to the runtime)
            logger.info(f"voice timing: {payload['kind']} at +{payload.get('after_s')}s")
        # the transcript mirror for the app (phase 2): the client receives every user turn, done event and spoken line
        await self.push_frame(RTVIServerMessageFrame(data={"falcon": kind, **payload}))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        heard = next((m.get("content") for m in reversed(frame.context.get_messages()) if m.get("role") == "user"), "") or ""
        if isinstance(heard, list):
            heard = " ".join(p.get("text", "") for p in heard if isinstance(p, dict))
        logger.info(f"heard ({self.call.lang}): {heard}")
        self.create_task(self.call.handle(heard), name="falcon-turn")


async def sign_in(body: dict | None) -> tuple[str, str, str]:
    """(session token, thread, lang) for this call."""
    body = body or {}
    lang = body.get("lang") or os.getenv("VOICE_LANG", "ar")
    if body.get("token"):
        return body["token"], body.get("thread") or "main", lang
    user = os.getenv("VOICE_DEMO_USER", "u-fatima")
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(f"{RUNTIME_URL}/api/demo/switch", json={"user_id": user, "reset": os.getenv("VOICE_RESET", "1") == "1"})
        r.raise_for_status()
    logger.info(f"voice call signed in as {user} ({lang})")
    return r.json()["token"], f"voice-{uuid.uuid4().hex[:8]}", lang


async def open_card(token: str, thread: str) -> dict | None:
    """The confirmation card the conversation is waiting on, if any (the chat's /api/chat/history carries it)."""
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(f"{RUNTIME_URL}/api/chat/history", params={"thread": thread}, headers={"Authorization": f"Bearer {token}"})
            r.raise_for_status()
        return r.json().get("pending")
    except Exception as exc:  # noqa: BLE001 - a new thread has no history; the call starts without a card
        logger.debug(f"no history for the call's thread: {exc}")
        return None


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    token, thread, lang = await sign_in(runner_args.body if isinstance(runner_args.body, dict) else None)
    terms = await keyterms(token, lang)
    logger.info(f"keyterms ({len(terms)}): {', '.join(terms)}")
    stt_name = os.getenv("VOICE_STT", "deepgram")
    shadow_name = os.getenv("VOICE_STT_SHADOW", "")
    stt = make_stt(stt_name, lang, terms)
    logger.info(f"voice stt: {stt_name}" + (f", shadow {shadow_name}" if shadow_name else "") + f" ({lang})")
    shadow = (PipelineWorker(Pipeline([make_stt(shadow_name, lang, terms, shadow=True), ConfidenceLog(f"{shadow_name} shadow")]))
              if shadow_name and shadow_name != stt_name else None)
    tts = ElevenLabsTTSService(api_key=os.environ["ELEVENLABS_API_KEY"],
                               settings=ElevenLabsTTSService.Settings(voice=VOICES[lang], model=TTS_MODEL, language=STT_LANG[lang]))
    agent = FalconAgentService(RuntimeClient(RUNTIME_URL, token, thread, lang), lang)
    agent.call.pending = await open_card(token, thread)  # a card already on screen when Call is pressed is answered by voice

    context = LLMContext()
    # empty_user_turn=None: a turn that ends with no transcript (noise) is never answered (the spike replied to one).
    # The turn ends after VOICE_TURN_PAUSE seconds of silence, not when Smart Turn v3 judges the sentence complete: on Arabic it closed
    # the turn at nearly every pause and one request became four answers (FAILURE_DIARY #66).
    pause = float(os.getenv("VOICE_TURN_PAUSE", "1.5"))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(), empty_user_turn=None,
        user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=pause)])))
    probe = os.getenv("VOICE_PROBE") == "1"
    stages = [transport.input(), *([ShadowTee(shadow)] if shadow else []), *([AudioProbe("mic")] if probe else []), stt, ConfidenceLog(stt_name),
              *([AudioProbe("stt")] if probe else []),
              user_aggregator, agent, tts, transport.output(), assistant_aggregator]
    # End of the customer's speech to the bot's first audio, per turn, with Pipecat's breakdown (endpointing, STT, agent, TTS):
    # the number VOICE_PLAN §4 bars at 4 s, logged on every call instead of measured once.
    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        for line in breakdown.turn_contribution_lines():
            logger.info(f"voice latency: {line}")

    worker = PipelineWorker(Pipeline(stages), params=PipelineParams(enable_metrics=True, enable_usage_metrics=True), observers=[latency])
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker, *([shadow] if shadow else []))

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # a fixed line, not a model turn: the customer knows the line is open before saying anything
        await worker.queue_frames([TTSSpeakFrame(GREETING[lang]), RTVIServerMessageFrame(data={"falcon": "call", "thread": thread, "lang": lang})])

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        logger.debug(f"user turn stopped ({strategy}): {message.content}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)})
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
