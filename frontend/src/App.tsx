import { useCallback, useEffect, useRef, useState } from "react";
import { ModelPicker } from "./components/ModelPicker";
import { ApiError, streamChat } from "./api/client";
import { Composer } from "./components/Composer";
import { Knowledge } from "./components/Knowledge";
import { Message, type ChatMessage } from "./components/Message";
import type { ToolCall } from "./api/types";

/** Stable enough for React keys within one session. */
const nextId = (() => {
  let n = 0;
  return () => `m${++n}`;
})();

function describe(error: unknown): { title: string; detail: string } {
  if (error instanceof ApiError) {
    if (error.isNotImplemented) {
      return {
        title: "The engine has no agent yet",
        detail: error.message,
      };
    }
    if (error.isEngineDown) {
      return {
        title: "The engine service is not running",
        detail: `${error.message} Start it with \`make engine\`.`,
      };
    }
    return { title: `Backend error ${error.status}`, detail: error.message };
  }
  if (error instanceof Error && error.name === "AbortError") {
    return { title: "Stopped", detail: "You stopped this answer." };
  }
  return {
    title: "Could not reach the backend",
    detail: "Is it running on port 8000? Start it with `make backend`.",
  };
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  /** Null until someone picks -- the YAML's own model applies until then, and
      whatever is chosen applies to every following turn. */
  const [model, setModel] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /** Update one message in place without disturbing the others. */
  const patch = useCallback(
    (id: string, change: (message: ChatMessage) => ChatMessage) => {
      setMessages((all) =>
        all.map((message) => (message.id === id ? change(message) : message)),
      );
    },
    [],
  );

  const send = useCallback(
    async (text: string) => {
      const history = messages
        .filter((message) => !message.problem && message.text)
        .map((message) => ({ role: message.role, content: message.text }));

      const question: ChatMessage = {
        id: nextId(),
        role: "user",
        text,
        sources: [],
        toolCalls: [],
        streaming: false,
      };
      const answer: ChatMessage = {
        id: nextId(),
        role: "assistant",
        text: "",
        sources: [],
        toolCalls: [],
        streaming: true,
      };

      setMessages((all) => [...all, question, answer]);
      setBusy(true);

      const controller = new AbortController();
      abort.current = controller;

      try {
        const stream = streamChat(
          { message: text, history, model: model ?? undefined },
          controller.signal,
        );

        for await (const event of stream) {
          switch (event.type) {
            case "retrieval":
              patch(answer.id, (m) => ({ ...m, sources: event.sources }));
              break;

            case "token":
              patch(answer.id, (m) => ({ ...m, text: m.text + event.text }));
              break;

            case "tool_call_started": {
              const call: ToolCall = {
                call_id: event.call_id,
                tool: event.tool,
                server: event.server,
                arguments: event.arguments,
              };
              patch(answer.id, (m) => ({
                ...m,
                toolCalls: [...m.toolCalls, call],
              }));
              break;
            }

            case "tool_call_finished":
              patch(answer.id, (m) => ({
                ...m,
                toolCalls: m.toolCalls.map((call) =>
                  call.call_id === event.call_id
                    ? {
                        ...call,
                        ok: event.ok,
                        duration_ms: event.duration_ms,
                        error: event.error,
                      }
                    : call,
                ),
              }));
              break;

            case "usage":
              patch(answer.id, (m) => ({ ...m, usage: event }));
              break;

            case "error":
              patch(answer.id, (m) => ({
                ...m,
                problem: { title: "The answer failed", detail: event.message },
              }));
              break;

            case "done":
              break;

            default:
              // An event type this build does not know about. Ignoring it keeps
              // the conversation working against a newer backend.
              break;
          }
        }
      } catch (error) {
        patch(answer.id, (m) => ({ ...m, problem: describe(error) }));
      } finally {
        patch(answer.id, (m) => ({ ...m, streaming: false }));
        setBusy(false);
        abort.current = null;
      }
    },
    [messages, patch, model],
  );

  return (
    <div className="app">
      <header className="header">
        <h1 className="header__title">SkyDesk Support</h1>
        <p className="header__sub">
          Ask about baggage, refunds, check-in, or a booking reference such as
          <code>AB12CD</code>.
        </p>
      </header>

      <main className="main">
        <section className="chat">
          {messages.length === 0 && (
            <div className="empty">
              <p>Nothing asked yet. Try one of these:</p>
              <ul>
                <li>What is the cabin baggage allowance?</li>
                <li>Is my flight delayed? My booking is AB12CD.</li>
                <li>Can I get a refund on a Basic fare?</li>
              </ul>
            </div>
          )}

          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}
          <div ref={bottom} />
        </section>

        <Knowledge />
      </main>

      <footer className="footer">
        <ModelPicker value={model} onChange={setModel} disabled={busy} />
        <Composer
          onSend={send}
          onStop={() => abort.current?.abort()}
          busy={busy}
        />
      </footer>
    </div>
  );
}
