"""Tracing: spans exist where the agent's own logic lives, and ids are available for the audit log and the UI.

Runs with an in-memory exporter: no collector, no key. The LangChain instrumentation is switched on by setup_tracing, so
the scripted graph run below also produces the automatic LangGraph spans; we only assert on our own.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from falcon.core import telemetry
from falcon.services.runtime.graph import build_graph

EXPORTER = InMemorySpanExporter()


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


def _run(drafts: list[str], thread: str) -> dict:
    model = ScriptedModel(messages=iter([AIMessage(content=d) for d in drafts]))
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": thread, "tools": [], "llm": model, "default_lang": "en"}}
    initial = {"messages": [HumanMessage(content="what do I get back if I cancel?")], "user_name": "Test"}
    asyncio.run(graph.ainvoke(initial, config))
    return asyncio.run(graph.aget_state(config)).values


def test_setup_is_idempotent_and_ids_follow_the_current_span():
    telemetry.setup_tracing("falcon-test", exporter=EXPORTER)
    telemetry.setup_tracing("falcon-test", exporter=EXPORTER)  # second call: no second provider, no error
    assert telemetry.enabled()
    assert telemetry.current_ids() == (None, None)  # no span active
    with telemetry.tracer("t").start_as_current_span("chat.turn"):
        trace_id, span_id = telemetry.current_ids()
        assert trace_id and len(trace_id) == 32 and span_id and len(span_id) == 16
    assert telemetry.link_to(trace_id, span_id)[0].context.trace_id == int(trace_id, 16)
    assert telemetry.link_to(None, None) == [] and telemetry.link_to("zz", "zz") == []


def test_reviewer_rounds_and_holds_are_spans():
    telemetry.setup_tracing("falcon-test", exporter=EXPORTER)
    EXPORTER.clear()
    with telemetry.tracer("t").start_as_current_span("chat.turn"):
        values = _run(["You would get about 900.00 AED back.", "I can quote the exact refund: shall I check?"], thread="trace-1")
    assert values["verification"]["status"] == "retried"
    names = [s.name for s in EXPORTER.get_finished_spans()]
    assert "reviewer.round" in names and "chat.turn" in names
    rnd = next(s for s in EXPORTER.get_finished_spans() if s.name == "reviewer.round")
    assert rnd.attributes["falcon.round"] == 1 and rnd.attributes["falcon.problems"] == "unsupported_amounts"
    assert "900.00" in rnd.attributes["falcon.nudge"]
    turn = next(s for s in EXPORTER.get_finished_spans() if s.name == "chat.turn")
    assert rnd.context.trace_id == turn.context.trace_id  # the round is inside the turn

    EXPORTER.clear()
    _run(["Refund 900.00 AED."] * 5, thread="trace-2")
    names = [s.name for s in EXPORTER.get_finished_spans()]
    assert names.count("reviewer.round") == 3 and "reviewer.hold" in names


def test_otlp_endpoint_normalisation_and_headers():
    assert telemetry._otlp_traces_endpoint("http://phoenix:6006") == "http://phoenix:6006/v1/traces"
    assert telemetry._otlp_traces_endpoint("https://cloud.langfuse.com/api/public/otel/") == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert telemetry._otlp_traces_endpoint("http://x/v1/traces") == "http://x/v1/traces"
    assert telemetry._headers("Authorization=Basic abc=,x-y=1") == {"Authorization": "Basic abc=", "x-y": "1"}
    assert telemetry._headers("") == {}
