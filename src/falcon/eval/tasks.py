"""Task definitions for the evaluation harness, loaded from eval/tasks/*.yaml.

A task is a hidden goal for the simulated user plus the checks that decide success:
- write_actions: write tools that must have executed (name + argument subset)
- forbidden_actions: tools that must NOT have executed successfully
- must_mention / must_not_mention: regexes over the assistant's messages
- final_state: SQL checks on the database after the run
- language: 'follow' (reply in the language of the user's last message) or a fixed code
- grounding: when true, every clause id the assistant cites must come from a retrieval result in the conversation so far (that turn or an earlier one)
- expected_tools: the tool calls a competent agent makes for this task, in order ("a|b" = either); used for the
  trajectory metrics (sequence correctness, efficiency), not for pass/fail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from falcon.core.config import REPO_ROOT

TASK_DIR = REPO_ROOT / "eval" / "tasks"


@dataclass
class Task:
    id: str
    lang: str
    persona: str
    goal: str
    facts: dict = field(default_factory=dict)
    script: list[str] = field(default_factory=list)       # fake-mode user utterances
    script_confirm: bool = True                            # fake-mode: approve confirmations
    fake_ok: bool = False                                  # the scripted model can complete this task
    max_turns: int = 8
    write_actions: list[dict] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    final_state: list[dict] = field(default_factory=list)
    language: str = "follow"
    grounding: bool = False
    expected_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    attachments: list[dict] = field(default_factory=list)  # [{before_turn: n, file: <eval/fixtures/...>}]: uploaded by the runner before the n-th user message


def load_tasks(pattern: str | None = None, fake_only: bool = False) -> list[Task]:
    tasks = []
    for path in sorted(TASK_DIR.glob("*.yaml")):
        for raw in yaml.safe_load(path.read_text(encoding="utf-8")):
            task = Task(**raw)
            if pattern and not any(part.strip() and part.strip() in task.id for part in pattern.split(",")):
                continue  # one substring, or a comma-separated list of them (re-run the tasks that failed)
            if fake_only and not task.fake_ok:
                continue
            tasks.append(task)
    return tasks
