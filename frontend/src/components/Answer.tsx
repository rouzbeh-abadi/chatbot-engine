import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * The assistant's answer, rendered as Markdown.
 *
 * Models reach for lists, emphasis and the occasional table when explaining a
 * policy, and the knowledge base is Markdown to begin with. Rendering it is
 * kinder than asking the model for plain prose, which produces walls of text.
 *
 * **No raw HTML.** `react-markdown` ignores embedded HTML unless you add
 * `rehype-raw`, and this deliberately does not: answer text is built partly from
 * retrieved documents and tool output, neither of which is trusted enough to
 * execute in the page.
 *
 * Partially streamed Markdown is fine -- an unterminated `**` renders as literal
 * asterisks until the closing pair arrives, then reflows.
 */
export function Answer({ text }: { text: string }) {
  return (
    <div className="answer">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
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
