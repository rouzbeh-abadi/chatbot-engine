import { useState } from "react";

/** The input. Enter sends, Shift+Enter adds a line. */
export function Composer({
  onSend,
  onStop,
  busy,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
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
        rows={2}
        aria-label="Your message"
      />
      {busy ? (
        <button type="button" className="btn btn--stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="submit" className="btn" disabled={!text.trim()}>
          Send
        </button>
      )}
    </form>
  );
}
