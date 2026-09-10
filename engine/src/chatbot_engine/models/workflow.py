"""A workflow: the turn as a graph of steps, described as data.

An assistant can carry one of these instead of relying on an agent's built-in
shape. Nodes come from a fixed library, so a caller composes steps but never
ships code; the `workflow` agent plugin turns the description into a LangGraph
at run time. The schema is the contract, validated at the API boundary.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_Id = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_-]*$")]


class _Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: _Id


class RetrieveNode(_Node):
    """Search the knowledge base for the message; later model steps see the passages."""

    type: Literal["retrieve"]


class ModelNode(_Node):
    """Call the assistant's model with the conversation and whatever retrieval found.

    Streams the reply as the answer unless `var` names a variable to hold it
    instead. Runs the assistant's tools when the model asks for them, up to the
    assistant's tool-iteration limit.
    """

    type: Literal["model"]
    #: Extra instructions for this step, appended to the assistant's prompt.
    prompt: str = Field(default="", max_length=4000)
    #: Store the reply here instead of speaking it. Templates read it as `{{vars.<var>}}`.
    var: str | None = Field(default=None, max_length=40, pattern=r"^[a-z][a-z0-9_]*$")
    #: Whether the model may call the assistant's tools in this step.
    tools: bool = True


class ConditionNode(_Node):
    """Ask the model one classification question and branch on the label it picks."""

    type: Literal["condition"]
    question: str = Field(min_length=1, max_length=2000)
    #: Label to the node that follows. The first label is the fallback when the
    #: model answers with none of them.
    branches: dict[str, _Id] = Field(min_length=2, max_length=8)


class ToolNode(_Node):
    """Call one named tool with templated arguments and keep the result in a variable."""

    type: Literal["tool"]
    tool: str = Field(min_length=1, max_length=80)
    #: Values are templates: `{{message}}`, `{{user_id}}`, `{{vars.<name>}}`.
    arguments: dict[str, str] = Field(default_factory=dict, max_length=20)
    var: str = Field(max_length=40, pattern=r"^[a-z][a-z0-9_]*$")


class ReplyNode(_Node):
    """Say a fixed text, rendered with the same templates as tool arguments."""

    type: Literal["reply"]
    text: str = Field(min_length=1, max_length=4000)


class HandoffNode(_Node):
    """Tell the visitor a person will follow up, and mark the turn handed off."""

    type: Literal["handoff"]
    message: str = Field(min_length=1, max_length=2000)
    #: A tool to call with the transcript, when the assistant has one for it.
    tool: str | None = Field(default=None, max_length=80)


class EndNode(_Node):
    """Finish the turn."""

    type: Literal["end"]


WorkflowNode = Annotated[
    RetrieveNode
    | ModelNode
    | ConditionNode
    | ToolNode
    | ReplyNode
    | HandoffNode
    | EndNode,
    Field(discriminator="type"),
]


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_: _Id = Field(alias="from")
    to: _Id


class WorkflowSpec(BaseModel):
    """The graph: nodes, edges, where to start, and how many steps a turn may take."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    nodes: list[WorkflowNode] = Field(min_length=1, max_length=20)
    #: A node with no outgoing edge and no branches ends the turn.
    edges: list[Edge] = Field(default_factory=list, max_length=40)
    start: _Id
    #: Node visits allowed in one turn, so a loop cannot run away.
    max_steps: int = Field(default=30, ge=1, le=100)

    @model_validator(mode="after")
    def _well_formed(self) -> WorkflowSpec:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        known = set(ids)
        if self.start not in known:
            raise ValueError(f"start names an unknown node {self.start!r}")
        for e in self.edges:
            for end in (e.from_, e.to):
                if end not in known:
                    raise ValueError(f"edge names an unknown node {end!r}")
        for n in self.nodes:
            if n.type == "condition":
                for label, target in n.branches.items():
                    if target not in known:
                        raise ValueError(
                            f"branch {label!r} of {n.id!r} names an unknown node {target!r}"
                        )
                if any(e.from_ == n.id for e in self.edges):
                    raise ValueError(
                        f"condition {n.id!r} routes by its branches, not by edges"
                    )
            elif n.type != "end" and sum(e.from_ == n.id for e in self.edges) > 1:
                raise ValueError(f"node {n.id!r} has more than one outgoing edge")
        # Everything must be reachable from the start, or it is a mistake.
        seen: set[str] = set()
        todo = [self.start]
        while todo:
            cur = todo.pop()
            if cur in seen:
                continue
            seen.add(cur)
            node = next(n for n in self.nodes if n.id == cur)
            todo.extend(e.to for e in self.edges if e.from_ == cur)
            if node.type == "condition":
                todo.extend(node.branches.values())
        unreachable = known - seen
        if unreachable:
            raise ValueError(f"unreachable nodes: {sorted(unreachable)}")
        return self

    def next_of(self, node_id: str) -> str | None:
        """The node an unconditional edge leads to, or None to end the turn."""
        return next((e.to for e in self.edges if e.from_ == node_id), None)
