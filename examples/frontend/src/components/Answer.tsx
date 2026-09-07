import type { ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { withCitations } from "./Citations";
import type { SourceRef } from "../api/types";

/**
 * The assistant's answer, rendered as Markdown with its citations inline.
 *
 * Models reach for lists, emphasis and the occasional table when explaining a
 * policy, and the knowledge base is Markdown to begin with. Rendering it is
 * kinder than asking the model for plain prose, which produces walls of text.
 *
 * The model ends a cited sentence with `[2]`; those become chips naming the
 * file, right where the claim is, so the reader never has to match a claim
 * against a list somewhere else.
 *
 * **No raw HTML.** `react-markdown` ignores embedded HTML unless you add
 * `rehype-raw`, and this deliberately does not: answer text is built partly from
 * retrieved documents and tool output, neither of which is trusted enough to
 * execute in the page.
 *
 * Partially streamed Markdown is fine -- an unterminated `**` renders as literal
 * asterisks until the closing pair arrives, then reflows.
 */
export function Answer({
  text,
  sources = [],
}: {
  text: string;
  sources?: SourceRef[];
}) {
  /** Render a block with its citation markers turned into chips. */
  const cited = (Tag: "p" | "li") =>
    function Cited({ children }: { children?: ReactNode }) {
      const parts = Array.isArray(children) ? children : [children];

      return <Tag>{withCitations(parts, sources)}</Tag>;
    };

  return (
    <div className="answer">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: cited("p"),
          li: cited("li"),
          // Links in generated text point wherever the model decided. Open them
          // in a new tab, and sever the opener so the target cannot reach back.
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer nofollow">
              {children}
            </a>
          ),
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}
