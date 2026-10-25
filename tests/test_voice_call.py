"""VOICE_PLAN rules 1-3 against a fake runtime: only the reviewed reply is spoken, one progress phrase per turn, the spoken card."""

import httpx

from falcon.core.confirmation import is_bare_confirmation, is_bare_decline
from falcon.services.voice.call import NO_ANSWER_NOTE, PROGRESS, STILL, UNAVAILABLE, CallSession, card_text

PENDING = {"type": "confirm", "lang": "ar", "actions": [{"tool": "cancel_booking", "args": {"pnr": "K3P7QZ", "quote_id": "Q-1"}, "quote": {
    "kind": "cancel", "pnr": "K3P7QZ", "fee": "100.00", "refund": "1291.00", "refund_method": "credit", "currency": "AED",
    "flights": [{"seq": 1, "flight_number": "FA929", "origin": "DXB", "destination": "CAI", "departure": "2026-10-21 19:40 (Gulf time)"}]}}]}


class FakeRuntime:
    def __init__(self, *turns):
        self.turns, self.calls = list(turns), []

    async def _events(self, events):
        for e in events:
            if isinstance(e, Exception):
                raise e
            yield e

    def chat(self, message):
        self.calls.append(("chat", message))
        return self._events(self.turns.pop(0))

    def confirm(self, approved, note=None):
        self.calls.append(("confirm", approved, note))
        return self._events(self.turns.pop(0))


def session(runtime, lang="ar"):
    spoken, seen = [], []

    async def speak(text, kind):
        spoken.append((kind, text))

    async def observe(kind, payload):
        seen.append(kind)

    return CallSession(runtime, lang, speak, observe, progress_after=0.05, still_after=0.15), spoken, seen


async def test_only_the_done_message_is_spoken_never_the_token_stream():
    rt = FakeRuntime([("token", {"text": "draft that the reviewer rejected"}), ("review", {"status": "revise"}),
                      ("token", {"text": "Hello"}), ("done", {"message": "Hello Fatima.", "pending": None})])
    call, spoken, seen = session(rt, "en")
    await call.handle("hi")
    assert spoken == [("reply", "Hello Fatima.")]
    assert seen == ["user", "done", "spoken"]


class DelayedRuntime(FakeRuntime):
    """The reply arrives after `delay` seconds, whatever tool events come before it."""

    def __init__(self, delay, *turns):
        super().__init__(*turns)
        self.delay = delay

    async def _events(self, events):
        import asyncio
        for e in events:
            if e[0] == "done":
                await asyncio.sleep(self.delay)
            yield e


async def test_a_fast_reply_gets_no_progress_phrase():
    rt = DelayedRuntime(0.0, [("tool", {"tool": "get_booking"}), ("done", {"message": "ok", "pending": None})])
    call, spoken, _ = session(rt, "ar")
    await call.handle("ما ساعة رحلتي؟")
    assert [k for k, _ in spoken] == ["reply"]


async def test_a_slow_reply_gets_one_fixed_phrase_before_it():
    rt = DelayedRuntime(0.1, [("done", {"message": "ok", "pending": None})])
    call, spoken, _ = session(rt, "ar")
    await call.handle("أريد إلغاء حجزي")
    assert [k for k, _ in spoken] == ["progress", "reply"] and spoken[0][1] == PROGRESS["ar"]


async def test_a_very_slow_reply_gets_the_second_phrase_once_and_never_a_third():
    rt = DelayedRuntime(0.6, [("tool", {"tool": "search_policy"}), ("tool", {"tool": "check_travel_eligibility"}), ("done", {"message": "ok", "pending": None})])
    call, spoken, _ = session(rt, "ar")
    await call.handle("أنا حامل، هل يمكنني السفر؟")
    assert spoken == [("progress", PROGRESS["ar"]), ("progress", STILL["ar"]), ("reply", "ok.")]


async def test_the_opening_phrase_is_said_on_the_first_turn_only():
    rt = DelayedRuntime(0.1, *[[("done", {"message": "ok", "pending": None})]] * 2)
    call, spoken, _ = session(rt, "ar")
    await call.handle("أريد إلغاء حجزي")
    await call.handle("نعم، كرصيد سفر")
    assert [k for k, _ in spoken] == ["progress", "reply", "reply"] and spoken[0][1] == PROGRESS["ar"]  # the second slow turn gets no filler
    rt2 = DelayedRuntime(0.6, *[[("done", {"message": "ok", "pending": None})]] * 2)
    call2, spoken2, _ = session(rt2, "ar")
    await call2.handle("أريد إلغاء حجزي")
    await call2.handle("كم المبلغ؟")
    assert spoken2[-2:] == [("progress", STILL["ar"]), ("reply", "ok.")]  # a very slow later turn still gets the long-wait phrase, once


async def test_the_card_is_spoken_and_a_bare_yes_presses_confirm():
    rt = FakeRuntime([("done", {"message": "الرسوم 100.00 AED.", "pending": PENDING})], [("done", {"message": "تم إلغاء الحجز K3P7QZ.", "pending": None})])
    call, spoken, _ = session(rt)
    await call.handle("ألغِ حجزي")
    assert [k for k, _ in spoken] == ["reply", "card"]
    assert "كيه ثلاثة بي سبعة كيو زد" in spoken[1][1] and "ألف ومئتان وواحد وتسعون درهم" in spoken[1][1] and "قل نعم للتأكيد" in spoken[1][1]
    await call.handle("نعم")
    assert rt.calls[-1] == ("confirm", True, None)
    assert call.pending is None


async def test_a_bare_no_declines():
    rt = FakeRuntime([("done", {"message": "?", "pending": PENDING})], [("done", {"message": "لم يتغير شيء.", "pending": None})])
    call, _, _ = session(rt)
    await call.handle("ألغِ حجزي")
    await call.handle("لا شكرا")
    assert rt.calls[-1] == ("confirm", False, None)


async def test_anything_else_repeats_the_card_once_then_declines():
    rt = FakeRuntime([("done", {"message": "?", "pending": PENDING})], [("done", {"message": "لم يتغير شيء.", "pending": None})])
    call, spoken, _ = session(rt)
    await call.handle("ألغِ حجزي")
    await call.handle("كم الرسوم بالضبط؟")
    assert len(rt.calls) == 1 and spoken[-1][0] == "card" and spoken[-1][1].startswith("لم أسمع نعم أو لا.")
    await call.handle("وماذا عن التغيير؟")
    assert rt.calls[-1] == ("confirm", False, NO_ANSWER_NOTE)
    assert call.pending is None


async def test_an_empty_turn_is_never_sent_or_answered():
    rt = FakeRuntime()
    call, spoken, _ = session(rt)
    await call.handle("   ")
    assert rt.calls == [] and spoken == []


async def test_a_runtime_failure_is_spoken_as_unavailable():
    rt = FakeRuntime([httpx.ConnectError("refused")])
    call, spoken, _ = session(rt, "en")
    await call.handle("hello")
    assert spoken == [("error", UNAVAILABLE["en"])]


def test_card_text_english():
    text = card_text({**PENDING, "lang": "en"}, "en")
    assert text.startswith("To confirm: Cancel booking K3P7QZ. Flight FA929, DXB to CAI, 21 October 19:40.")
    assert "You get back 1291.00 AED as travel credit, fee 100.00 AED." in text


def test_yes_and_no_vocabulary():
    for t in ["لا", "لا شكرا", "no thanks", "No, leave it", "don't", "لا تلغيه"]:
        assert is_bare_decline(t) and not is_bare_confirmation(t), t
    for t in ["نعم", "yes please", "لا نعم", "no, but what about Flex?", "ما هذا", "كم الرسوم؟", "no, cancel the booking"]:
        assert not is_bare_decline(t), t


async def test_what_is_said_while_a_turn_runs_goes_to_the_agent_as_one_message():
    import asyncio

    release = asyncio.Event()

    class SlowRuntime(FakeRuntime):
        async def _events(self, events):
            if len(self.calls) == 1:
                await release.wait()
            for e in events:
                yield e

    rt = SlowRuntime([("done", {"message": "first", "pending": None})], [("done", {"message": "second", "pending": None})])
    call, spoken, _ = session(rt)
    first = asyncio.create_task(call.handle("بعتقد لازم"))
    await asyncio.sleep(0)
    await call.handle("الرحلة راح تكون مشغولة")
    await call.handle("فممكن تلغي الرحلة")
    release.set()
    await first
    assert rt.calls == [("chat", "بعتقد لازم"), ("chat", "الرحلة راح تكون مشغولة فممكن تلغي الرحلة")]
    assert [t for _, t in spoken] == ["first.", "second."]
