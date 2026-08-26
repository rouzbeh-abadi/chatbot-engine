import { useState, type ReactNode } from "react";

/** The input. Enter sends, Shift+Enter adds a line. Controls sit in a bar
 * below the text: anything passed as children on the left, the send button
 * (an up arrow) on the right. */
export function Composer({
  onSend,
  onStop,
  busy,
  children,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
  children?: ReactNode;
}) {
  const [text, setText] = useState("");

  function send() {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    onSend(trimmed);
    setText("");
  }

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        send();
      }}
    >
      <textarea
        className="composer__input"
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            send();
          }
        }}
        placeholder="Ask about baggage, refunds, check-in, or a booking reference…"
        rows={1}
        aria-label="Your message"
      />

      <div className="composer__bar">
        <div className="composer__tools">{children}</div>

        {busy ? (
          <button
            type="button"
            className="composer__send composer__send--stop"
            onClick={onStop}
            aria-label="Stop"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor" />
            </svg>
          </button>
        ) : (
          <button
            type="submit"
            className="composer__send"
            disabled={!text.trim()}
            aria-label="Send"
          >
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path
                d="M12 20V6M6 12l6-6 6 6"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        )}
      </div>
    </form>
  );
}
