"""Text the chatbot did not write: knowledge extracts, the person's files, tool results.

Any of it may carry an instruction meant for the model (a prompt injection):
a line in a crawled page, a sentence in an uploaded PDF, an order note in a
shop. The prompts say such text is data, and the engine puts it where the
model trusts it least, in the person's turn or a tool's result. Two more
things are done here, before any of it reaches a model:

- Characters a reader cannot see but a model reads are removed: the Unicode
  tag characters (U+E0000 to U+E007F), which spell ASCII invisibly and are
  the usual way to hide an instruction from a person; the bidirectional
  embedding, override and isolate controls; the zero-width space, the word
  joiner and the invisible operators; the variation selectors (U+FE00 to
  U+FE0F, U+E0100 to U+E01EF), which can carry bytes hidden behind an emoji;
  and a byte order mark inside the text. Joiners and direction marks (U+200C
  to U+200F) stay: Persian, Arabic and the Indic scripts need them, and so do
  emoji. An emoji that loses its presentation selector still reads the same.
- A closing tag for the frame the text travels in is shown as `[/tag]`, so
  the text cannot end its frame early and speak from outside it, however it
  is spelt: in either case, spaced, with the fullwidth less-than or
  greater-than sign (U+FF1C, U+FF1E), or with a slash that only looks like
  one.
- A line that starts the way a numbered extract does (`[3] other.md`) gets a
  backslash in front (`unnumbered`), so it cannot pass for another extract.

And when a document is indexed, `instruction_warnings` says what in it reads
like orders to an AI, so its owner can look. Nothing is refused for it: a
help page about prompt injection is a fair document, and the warning is a
cue to read, not a verdict.
"""

from __future__ import annotations

import re

#: Tag characters, the zero-width space, bidirectional controls, the word
#: joiner and invisible operators, variation selectors, a stray BOM.
_INVISIBLE = re.compile(
    "[\U000e0000-\U000e007f\u200b\u202a-\u202e\u2060-\u2064\u2066-\u2069"
    "\ufe00-\ufe0f\ufeff\U000e0100-\U000e01ef]"
)

#: What the owner is warned about: the invisible characters that hide or
#: reorder text, and a run of variation selectors, the way bytes are hidden
#: behind an emoji. One selector after an emoji, a zero-width space or a word
#: joiner is ordinary in text copied from the web, and is only removed.
_HIDING = re.compile(
    "[\U000e0000-\U000e007f\u202a-\u202e\u2061-\u2064\u2066-\u2069\ufeff]"
    "|[\ufe00-\ufe0f\U000e0100-\U000e01ef]{2,}"
)

#: The tag characters that mirror printable ASCII: U+E0020 to U+E007E.
_TAG_ASCII = re.compile("[\U000e0020-\U000e007e]+")

#: What a tag can open with besides `<`: the fullwidth less-than sign.
_OPENERS = "<\uff1c"
#: What a closing tag's slash can be besides `/`: the fullwidth solidus, the
#: fraction slash, the division slash and the big solidus.
_SLASHES = "/\uff0f\u2044\u2215\u29f8"
#: What a tag can end with besides `>`: the fullwidth greater-than sign.
_CLOSERS = ">\uff1e"

#: A line that starts the way a numbered extract does: `[3]`.
_NUMBERED_LINE = re.compile(r"^([ \t]*)\[(\d+)\]", re.MULTILINE)


def visible(text: str) -> str:
    """`text` without the characters a reader cannot see."""
    return _INVISIBLE.sub("", text)


#: How far past a closer's name its `>` is looked for. A closer whose `>` is
#: further away, or missing, still loses its opener, which is what ends a
#: frame; only what follows it stays as text.
_CLOSER_TAIL = 64

#: How much past a cut a text is framed, so a closer that crosses the cut is
#: still seen whole.
CLOSER_ROOM = _CLOSER_TAIL + 32


def closing_tag(tag: str) -> re.Pattern[str]:
    """A closing tag for `tag`, however it is spelt or spaced, or written with
    a fullwidth angle bracket or a slash that only looks like one; not the
    closer of another element whose name only starts with `tag`
    (`</file-list>`).

    The `>` is looked for at most `_CLOSER_TAIL` characters on. Read to the
    end of the text, as it was, every `</file ` without a `>` after it cost a
    scan of the rest, so a file of them held the event loop for seconds on
    every turn (docs/review-2026-10.md, TURN-2)."""
    return re.compile(
        rf"[{_OPENERS}][{_SLASHES}]\s*{re.escape(tag)}"
        rf"(?=[\s{_SLASHES}{_CLOSERS}]|$)(?:[^{_CLOSERS}]{{0,{_CLOSER_TAIL}}}[{_CLOSERS}])?",
        re.IGNORECASE,
    )


def framed(text: str, tag: str) -> str:
    """`text` made safe to put inside `<tag>…</tag>`: nothing invisible, no closer."""
    return closing_tag(tag).sub(f"[/{tag}]", visible(text))


def label(name: str, fallback: str = "file") -> str:
    """A name as a frame or a header may hold it: one line, nothing invisible,
    nothing that could open or close a tag."""
    return (
        " ".join(re.sub(rf'[{_OPENERS}{_CLOSERS}"]', " ", visible(name)).split())
        or fallback
    )


def unnumbered(text: str) -> str:
    """`text` with a backslash before each line that starts like a numbered
    extract (`[3] other.md` becomes `\\[3] other.md`).

    Extracts reach the model numbered, one header line each, and the model
    cites them by number. A line in a chunk that started the same way could
    pass the chunk's text off as another extract, or name a source nothing
    retrieved; escaped, it reads as what it is, a line of the chunk.
    """
    return _NUMBERED_LINE.sub(r"\1\\[\2]", text)


#: What reads like orders to an AI, by kind. Each pattern is narrow on
#: purpose: a warning nobody believes is worse than none.
_INSTRUCTIONS = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
    r"\b(?:previous|prior|above|earlier|all|any|your|these|those|system)\b[^.\n]{0,30}?"
    r"\b(?:instructions?|rules|prompts?|guidelines|directions)\b"
    r"|\b(?:new|updated|real|actual)\s+(?:system\s+)?instructions?\s*:"
    r"|\byou\s+are\s+now\s+(?:a|an|the|in|no\s+longer)\b"
    r"|\b(?:reveal|print|repeat|show|output|disclose)\b[^.\n]{0,30}?"
    r"\b(?:system\s+prompt|your\s+(?:instructions|prompt|rules))\b"
    r"|\bdo\s+not\s+(?:tell|inform|mention\s+(?:this\s+)?to|reveal\s+(?:this\s+)?to)\s+(?:the\s+)?(?:user|visitor|customer|human)\b",
    re.IGNORECASE,
)

#: Chat markup that imitates a model's roles.
_ROLE_MARKUP = re.compile(
    r"<\|(?:im_start|im_end|system|assistant|user|endoftext)\|>|\[/?INST\]|<<SYS>>",
    re.IGNORECASE,
)


def _excerpt(text: str, start: int, end: int, *, width: int = 100) -> str:
    """The match with a little around it, on one line, at most `width` characters."""
    left = max(0, start - 30)
    snippet = " ".join(visible(text[left : end + 50]).split())
    if len(snippet) > width:
        snippet = snippet[: width - 1].rstrip() + "…"
    return ("…" if left else "") + snippet


def _places(count: int) -> str:
    return "" if count == 1 else f" ({count} places)"


def instruction_warnings(text: str) -> list[str]:
    """What in `text` reads like orders to an AI, one sentence per kind, for
    the document's owner. Empty for an ordinary document."""
    warnings: list[str] = []

    hidden = _TAG_ASCII.findall(text)
    if hidden:
        # Each tag character is its ASCII twin moved up by 0xE0000.
        spelt = " ".join(
            " ".join(
                "".join(chr(ord(c) - 0xE0000) for c in run) for run in hidden
            ).split()
        )
        shown = spelt if len(spelt) <= 100 else spelt[:99].rstrip() + "…"
        warnings.append(
            f"Hidden characters that a reader cannot see spell out: “{shown}”{_places(len(hidden))}"
        )
    elif _HIDING.search(text):
        warnings.append(
            "Hidden control characters that can change or hide how text reads"
        )

    clean = visible(text)
    found = list(_INSTRUCTIONS.finditer(clean))
    if found:
        first = found[0]
        warnings.append(
            f"Text that reads like instructions to an AI: “{_excerpt(clean, first.start(), first.end())}”{_places(len(found))}"
        )

    markup = list(_ROLE_MARKUP.finditer(text))
    if markup:
        warnings.append(
            f"Chat markup that imitates an AI's roles: “{markup[0].group(0)}”{_places(len(markup))}"
        )

    return warnings
