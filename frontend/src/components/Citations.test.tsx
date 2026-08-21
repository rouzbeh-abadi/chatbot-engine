/**
 * Citation markers come from model output, so the parsing has to cope with
 * whatever it writes -- including nothing, and including a `[2]` that is really
 * part of a quoted document rather than a citation.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ReactNode } from "react";

import { withCitations } from "./Citations";
import type { SourceRef } from "../api/types";

const source = (over: Partial<SourceRef> = {}): SourceRef => ({
  doc_id: "d1",
  source: "cancellations.md",
  score: 0.5,
  heading: null,
  excerpt: null,
  ...over,
});

function renderBlock(children: ReactNode[], sources: SourceRef[]) {
  return render(<p>{withCitations(children, sources)}</p>);
}

describe("withCitations", () => {
  it("turns a marker into a chip naming the document", () => {
    renderBlock(["Refunds take 7 days. [1]"], [source()]);

    expect(screen.getByText("cancellations.md")).toHaveClass("cite");
    expect(screen.getByText(/Refunds take 7 days\./)).toBeInTheDocument();
  });

  it("replaces a marker in the middle of a paragraph too", () => {
    // Several sentences on their own lines are one Markdown paragraph, so a
    // trailing-only rule left these as literal text.
    renderBlock(
      ["Fees are per segment. [1]\nSeats are per leg. [2]"],
      [source(), source({ source: "seat_selection.md" })],
    );

    expect(screen.getByText("cancellations.md")).toBeInTheDocument();
    expect(screen.getByText("seat_selection.md")).toBeInTheDocument();
  });

  it("leaves no marker text behind", () => {
    const { container } = renderBlock(["Done. [1]"], [source()]);

    expect(container.textContent).not.toContain("[1]");
  });

  it("collapses a run to one chip per document", () => {
    renderBlock(["Both say so. [1][2]"], [source(), source({ doc_id: "d2" })]);

    expect(screen.getAllByText("cancellations.md")).toHaveLength(1);
  });

  it("shows a chip each when a run cites two documents", () => {
    renderBlock(
      ["Two of them. [1][2]"],
      [source(), source({ source: "refunds.md" })],
    );

    expect(screen.getByText("cancellations.md")).toBeInTheDocument();
    expect(screen.getByText("refunds.md")).toBeInTheDocument();
  });

  it("leaves a number that names no source as text", () => {
    const { container } = renderBlock(["See clause [9] of the policy."], [source()]);

    expect(container.textContent).toContain("[9]");
    expect(screen.queryByText("cancellations.md")).not.toBeInTheDocument();
  });

  it("passes React elements through untouched", () => {
    renderBlock(
      ["bold: ", <strong key="b">yes</strong>, " [1]"],
      [source()],
    );

    expect(screen.getByText("yes").tagName).toBe("STRONG");
    expect(screen.getByText("cancellations.md")).toBeInTheDocument();
  });

  it("adds the excerpt as the chip's tooltip", () => {
    renderBlock(["Cited. [1]"], [source({ excerpt: "Refunds are processed..." })]);

    expect(screen.getByText("cancellations.md")).toHaveAttribute(
      "title",
      "Refunds are processed...",
    );
  });
});
