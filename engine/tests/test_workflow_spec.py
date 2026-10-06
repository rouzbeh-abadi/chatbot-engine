"""The workflow schema rejects graphs that could not run."""

from __future__ import annotations

import pytest

from chatbot_engine.models.workflow import WorkflowSpec

GOOD = {
    "start": "retrieve",
    "nodes": [
        {"id": "retrieve", "type": "retrieve"},
        {
            "id": "kind",
            "type": "condition",
            "question": "Is this about an order?",
            "branches": {"order": "lookup", "other": "answer"},
        },
        {
            "id": "lookup",
            "type": "tool",
            "tool": "get_booking_status",
            "arguments": {"reference": "{{message}}"},
            "var": "booking",
        },
        {"id": "answer", "type": "model", "prompt": "Booking: {{vars.booking}}"},
    ],
    "edges": [{"from": "retrieve", "to": "kind"}, {"from": "lookup", "to": "answer"}],
}


def test_a_well_formed_workflow_validates():
    spec = WorkflowSpec.model_validate(GOOD)
    assert spec.next_of("retrieve") == "kind"
    assert spec.next_of("answer") is None


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(start="nowhere"), "unknown node"),
        (
            lambda d: d["edges"].append({"from": "answer", "to": "ghost"}),
            "unknown node",
        ),
        (lambda d: d["nodes"].append({"id": "orphan", "type": "end"}), "unreachable"),
        (
            lambda d: d["edges"].append({"from": "retrieve", "to": "answer"}),
            "more than one outgoing",
        ),
        (
            lambda d: d["edges"].append({"from": "kind", "to": "answer"}),
            "routes by its branches",
        ),
        (lambda d: d["nodes"].append({"id": "retrieve", "type": "end"}), "unique"),
        (
            lambda d: (
                d["nodes"].append({"id": "stop", "type": "end"}),
                d["edges"].extend(
                    [{"from": "answer", "to": "stop"}, {"from": "stop", "to": "lookup"}]
                ),
            ),
            "no edge may leave it",
        ),
        (lambda d: d["nodes"].append({"id": "x", "type": "teleport"}), "teleport"),
    ],
)
def test_malformed_workflows_are_refused(change, message):
    import copy

    bad = copy.deepcopy(GOOD)
    change(bad)
    with pytest.raises(ValueError, match=message):
        WorkflowSpec.model_validate(bad)


def test_the_assistant_config_carries_a_workflow():
    from chatbot_engine.models.chat import AssistantConfig

    config = AssistantConfig(project_id="p", name="P", system_prompt=".", workflow=GOOD)
    assert config.workflow is not None and config.workflow.start == "retrieve"


def _chain(n: int) -> dict:
    """A workflow of n reply steps, each leading to the next."""
    return {
        "start": "s0",
        "nodes": [{"id": f"s{i}", "type": "reply", "text": "Hi"} for i in range(n)],
        "edges": [{"from": f"s{i}", "to": f"s{i + 1}"} for i in range(n - 1)],
    }


def test_a_workflow_takes_up_to_forty_steps():
    """A workflow with several ready-made paths (a booking with its questions,
    a call-back, a subscription) outgrew twenty steps."""
    assert len(WorkflowSpec.model_validate(_chain(40)).nodes) == 40
    with pytest.raises(ValueError, match="at most 40 items"):
        WorkflowSpec.model_validate(_chain(41))
