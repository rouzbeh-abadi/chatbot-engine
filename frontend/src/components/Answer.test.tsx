/**
 * Markdown rendering.
 *
 * The security assertion is the important one here: answer text is assembled
 * partly from retrieved documents and tool output, so anything that renders it
 * must not execute markup that arrives in that text.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Answer } from "./Answer";

describe("markdown", () => {
  it("renders emphasis as elements rather than literal asterisks", () => {
    render(<Answer text="Your booking **AB12CD** is confirmed." />);

    expect(screen.getByText("AB12CD").tagName).toBe("STRONG");
    expect(screen.queryByText(/\*\*/)).not.toBeInTheDocument();
  });

  it("renders a list, which is how a policy answer usually arrives", () => {
    render(<Answer text={"Allowance:\n\n- one cabin bag\n- 8 kg\n- 55 x 40 x 23 cm"} />);

    expect(screen.getAllByRole("listitem")).toHaveLength(3);
  });

  it("renders a GFM table", () => {
    render(
      <Answer
        text={"| Fare | Refundable |\n| --- | --- |\n| Flexible | yes |\n| Basic | no |"}
      />,
    );

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(3);
  });

  it("renders code without executing it", () => {
    render(<Answer text="Use `make seed` to load documents." />);

    expect(screen.getByText("make seed").tagName).toBe("CODE");
  });
});

describe("untrusted content", () => {
  it("does not render embedded HTML", () => {
    // Retrieved documents and tool results end up inside this text. If HTML were
    // rendered, a document could inject markup into the page.
    render(<Answer text={'Hello <img src="x" onerror="alert(1)"> world'} />);

    expect(document.querySelector("img")).toBeNull();
  });

  it("does not render a script tag", () => {
    render(<Answer text={"<script>window.__pwned = true</script>"} />);

    expect(document.querySelector("script")).toBeNull();
    expect((window as unknown as { __pwned?: boolean }).__pwned).toBeUndefined();
  });

  it("opens generated links in a new tab with the opener severed", () => {
    render(<Answer text="See [the policy](https://example.com/refunds)." />);

    const link = screen.getByRole("link", { name: "the policy" });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link.getAttribute("rel")).toContain("noopener");
  });
});

describe("streaming", () => {
  it("shows an unterminated marker literally until its pair arrives", () => {
    // Mid-stream the text is incomplete; it must not blank out or throw.
    const { rerender } = render(<Answer text="Your booking **AB12" />);
    expect(screen.getByText(/\*\*AB12/)).toBeInTheDocument();

    rerender(<Answer text="Your booking **AB12CD** is confirmed." />);
    expect(screen.getByText("AB12CD").tagName).toBe("STRONG");
  });

  it("renders an empty answer without complaining", () => {
    const { container } = render(<Answer text="" />);
    expect(container.querySelector(".answer")).toBeInTheDocument();
  });
});
