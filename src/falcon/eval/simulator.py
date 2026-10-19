"""The simulated customer: a second agent that plays the person with a hidden goal.

Real mode: an LLM (SIMULATOR_MODEL via OpenRouter) with a strict script: it knows only the facts in the
task, speaks only in the task language unless the task says to switch, answers one message at a time,
decides confirmations from its goal, and says ###STOP### when the goal is reached or clearly impossible.
Known weakness (τ-bench reports it too): simulators invent details; the facts list and the STOP rule
limit that, and the failure diary records any run where the simulator went off-script.

Fake mode: replays the task's script, used to test the harness itself without a key.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from falcon.core.config import settings

from .tasks import Task

STOP = "###STOP###"

SIM_PROMPT = """You are role-playing a customer (or an airline employee) talking to Falcon Airline's assistant in a chat. Stay in character; never say you are simulated or an AI.

Your situation and goal:
{goal}

Facts you know (use only these; do not invent booking references, dates or amounts):
{facts}

Rules:
- Write ONLY the next message you would send, in {lang_name}{switch_rule}. Keep it short and natural (one or two sentences). Flight numbers and booking references stay in Latin letters.
- Do not reveal your whole goal at once; answer the assistant's questions as a real person would.
- If the assistant asks you to confirm an action, answer according to your goal (for example "yes, go ahead" or "no, not at that price").
- When your goal is achieved, or the assistant has clearly explained it is impossible and offered alternatives you do not want, reply with exactly: {stop}
- Never write more than one message per turn."""

CONFIRM_PROMPT = """The assistant now asks you to confirm this action before it happens:
{action}
Given your goal, reply with exactly APPROVE or DECLINE."""

LANG_NAMES = {"en": "English", "ar": "Arabic (Modern Standard, written in Arabic script)"}


class ScriptedSimulator:
    def __init__(self, task: Task):
        self.script = list(task.script)
        self.approve = task.script_confirm

    async def next_message(self, assistant_text: str, history: list[tuple[str, str]]) -> str:
        return self.script.pop(0) if self.script else STOP

    async def decide(self, action: dict) -> bool:
        return self.approve


class LlmSimulator:
    def __init__(self, task: Task, model=None):
        self.task = task
        if model is None:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model=settings.simulator_model, base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key, temperature=0.7, timeout=60, max_retries=2)
        self.model = model
        switch = ""
        if task.lang == "mixed":
            switch = ". Start in English; after the assistant's first reply, continue in Arabic for the rest of the conversation"
        facts = "\n".join(f"- {k}: {v}" for k, v in task.facts.items()) or "- (none)"
        self.system = SIM_PROMPT.format(goal=task.goal, facts=facts, lang_name=LANG_NAMES.get("en" if task.lang == "mixed" else task.lang, task.lang), switch_rule=switch, stop=STOP)

    async def next_message(self, assistant_text: str, history: list[tuple[str, str]]) -> str:
        # The simulator sees the conversation from its own side: assistant messages are "human" input to it.
        msgs = [SystemMessage(content=self.system)]
        for role, text in history:
            msgs.append(AIMessage(content=text) if role == "user" else HumanMessage(content=text))
        if assistant_text:
            msgs.append(HumanMessage(content=assistant_text))
        else:
            msgs.append(HumanMessage(content="(The chat is open. Send your first message.)"))
        reply = await self.model.ainvoke(msgs)
        text = reply.content if isinstance(reply.content, str) else "".join(b.get("text", "") for b in reply.content if isinstance(b, dict))
        return text.strip()

    async def decide(self, action: dict) -> bool:
        msgs = [SystemMessage(content=self.system), HumanMessage(content=CONFIRM_PROMPT.format(action=action))]
        reply = await self.model.ainvoke(msgs)
        text = (reply.content if isinstance(reply.content, str) else "").strip().upper()
        return text.startswith("APPROVE")


def make_simulator(task: Task, fake: bool):
    return ScriptedSimulator(task) if fake else LlmSimulator(task)
