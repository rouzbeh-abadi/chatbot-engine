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
  greater-than sign (U+FF1C, U+FF1E), with a slash that only looks like
  one, or with characters that show as nothing between its parts (a soft
  hyphen, a joiner or direction mark, a control character such as NUL).
- A line that starts the way a numbered extract does (`[3] other.md`) gets a
  backslash in front (`unnumbered`), so it cannot pass for another extract.

And when a document is indexed, `instruction_warnings` says what in it reads
like orders to an AI, so its owner can look. Nothing is refused for it: a
help page about prompt injection is a fair document, and the warning is a
cue to read, not a verdict.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

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

#: What a tag can open with besides `<`: the fullwidth and small-form
#: less-than signs, and the angle brackets that look like one (single
#: guillemet, CJK, mathematical and the older left-pointing angle bracket).
_OPENERS = "<\uff1c\ufe64\u2039\u3008\u27e8\u2329"
#: What a closing tag's slash can be besides `/`: the fullwidth solidus, the
#: fraction slash, the division slash and the big solidus.
_SLASHES = "/\uff0f\u2044\u2215\u29f8"
#: What a tag can end with besides `>`: the twins of the openers above.
_CLOSERS = ">\uff1e\ufe65\u203a\u3009\u27e9\u232a"
#: What shows as nothing, so a closer split by it still reads as one to a
#: model: the control characters (NUL among them) and every default-ignorable
#: code point, the ones `visible` keeps included (the soft hyphen, the
#: joiners and direction marks the scripts need, the combining grapheme
#: joiner, the Hangul fillers). Whitespace is matched apart, as `\s`.
_IGNORABLE = (
    "\x00-\x08\x0e-\x1f\x7f-\x9f\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5"
    "\u180b-\u180f\u200b-\u200f\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f"
    "\ufeff\uffa0\ufff0-\ufff8\U0001bca0-\U0001bca3\U0001d173-\U0001d17a"
    "\U000e0000-\U000e0fff"
)

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
    a fullwidth angle bracket or a slash that only looks like one, or split
    anywhere by characters that show as nothing (`_IGNORABLE`); not the
    closer of another element whose name only starts with `tag`
    (`</file-list>`).

    The `>` is looked for at most `_CLOSER_TAIL` characters on. Read to the
    end of the text, as it was, every `</file ` without a `>` after it cost a
    scan of the rest, so a file of them held the event loop for seconds on
    every turn (docs/review-2026-10.md, TURN-2)."""
    gap = rf"[\s{_IGNORABLE}]*"
    hidden = f"[{_IGNORABLE}]*"
    name = hidden.join(re.escape(letter) for letter in tag)
    return re.compile(
        rf"[{_OPENERS}]{gap}[{_SLASHES}]{gap}{name}"
        rf"(?={hidden}(?:[\s{_SLASHES}{_CLOSERS}]|$))"
        rf"(?:[^{_CLOSERS}]{{0,{_CLOSER_TAIL}}}[{_CLOSERS}])?",
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
#: purpose: a warning nobody believes is worse than none. Every gap in it is
#: bounded (at most 64 spaces between two words, 40 or 30 other characters),
#: so a match is never longer than `_LONGEST_MATCH`.
_INSTRUCTIONS = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
    r"\b(?:previous|prior|above|earlier|all|any|your|these|those|system)\b[^.\n]{0,30}?"
    r"\b(?:instructions?|rules|prompts?|guidelines|directions)\b"
    r"|\b(?:new|updated|real|actual)\s{1,64}(?:system\s{1,64})?instructions?\s{0,64}:"
    r"|\byou\s{1,64}are\s{1,64}now\s{1,64}(?:a|an|the|in|no\s{1,64}longer)\b"
    r"|\b(?:reveal|print|repeat|show|output|disclose)\b[^.\n]{0,30}?"
    r"\b(?:system\s{1,64}prompt|your\s{1,64}(?:instructions|prompt|rules))\b"
    r"|\bdo\s{1,64}not\s{1,64}(?:tell|inform|mention\s{1,64}(?:this\s{1,64})?to|reveal\s{1,64}(?:this\s{1,64})?to)"
    r"\s{1,64}(?:the\s{1,64})?(?:user|visitor|customer|human)\b",
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


#: How much of a text one call of a pattern reads in `instruction_warnings`.
#: A regex holds the interpreter for the whole of its call, and the scan runs
#: on an indexing thread beside the event loop: a 2 MB document matched in one
#: call kept every other request waiting for as long. A window at a time, the
#: loop gets its turn between them.
_WINDOW = 32_768

#: The longest a match of the patterns read a window at a time can be
#: (`_INSTRUCTIONS`, about 410 characters; `_ROLE_MARKUP`; `_HIDING`'s first
#: two characters, which are all a search needs), with room to spare.
_LONGEST_MATCH = 640


def _matches(pattern: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    """`pattern.finditer(text)`, a window of `_WINDOW` characters at a time.

    Each window is read on to `_LONGEST_MATCH` characters past its end, and
    one more for the word boundary after a match, so a match that starts in it is seen
    whole, and the character before it is the text's own (`pos` is not a
    slice). The next window starts where the last match ended, as one pass
    over the whole text goes on, so this finds exactly what that pass would,
    for a pattern no match of which is longer. A pattern of one repeated
    character class may have a match cut where the reading stops: `_runs`
    joins it up again."""
    pos, size = 0, len(text)
    while pos < size:
        end = min(size, pos + _WINDOW)
        reach = min(size, end + _LONGEST_MATCH + 1)
        after = end
        for match in pattern.finditer(text, pos, reach):
            if match.start() >= end:
                break
            yield match
            after = max(after, match.end())
        pos = after


def _runs(pattern: re.Pattern[str], text: str) -> list[str]:
    """`pattern.findall(text)` for a run of one character class
    (`_TAG_ASCII`), a window at a time, a run cut by a window joined up."""
    runs: list[str] = []
    last = -1
    for match in _matches(pattern, text):
        if match.start() == last:
            runs[-1] += match.group(0)
        else:
            runs.append(match.group(0))
        last = match.end()
    return runs


def _visible_in_windows(text: str) -> str:
    """`visible(text)`, a window at a time: it removes single characters, so
    any cut will do."""
    return "".join(
        visible(text[at : at + _WINDOW]) for at in range(0, len(text), _WINDOW)
    )


def instruction_warnings(text: str) -> list[str]:
    """What in `text` reads like orders to an AI, one sentence per kind, for
    the document's owner. Empty for an ordinary document. Read a window at a
    time (`_matches`), so a long document does not hold up the event loop."""
    warnings: list[str] = []

    hidden = _runs(_TAG_ASCII, text)
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
    elif next(_matches(_HIDING, text), None) is not None:
        warnings.append(
            "Hidden control characters that can change or hide how text reads"
        )

    clean = _visible_in_windows(text)
    found = list(_matches(_INSTRUCTIONS, clean))
    if found:
        first = found[0]
        warnings.append(
            f"Text that reads like instructions to an AI: “{_excerpt(clean, first.start(), first.end())}”{_places(len(found))}"
        )

    # On the text the model reads: a zero-width space inside `<|im_start|>`
    # hides it from a match on the raw text, and is removed before the model
    # sees it whole (INGEST-7).
    markup = [match.group(0) for match in _matches(_ROLE_MARKUP, clean)]
    if markup:
        warnings.append(
            f"Chat markup that imitates an AI's roles: “{markup[0]}”{_places(len(markup))}"
        )

    return warnings
