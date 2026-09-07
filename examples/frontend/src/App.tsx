import { useCallback, useEffect, useRef, useState } from "react";
import { AgentPicker } from "./components/AgentPicker";
import { MemoryPanel } from "./components/MemoryPanel";
import { ModelPicker } from "./components/ModelPicker";
import { ApiError, streamChat } from "./api/client";
import { Composer } from "./components/Composer";
import { Knowledge } from "./components/Knowledge";
import { Message, type ChatMessage } from "./components/Message";
import { Admin } from "./components/Admin";
import { ExportMenu } from "./components/ExportMenu";
import type { SessionUsage } from "./components/Knowledge";
import { exportConversation } from "./export";
import type { ToolCall } from "./api/types";

/** Total tokens and cost across every answered turn this session. */
function sessionUsage(messages: ChatMessage[]): SessionUsage {
  let tokens = 0;
  let cost = 0;
  let hasCost = false;
  let turns = 0;
  let model: string | undefined;

  for (const message of messages) {
    if (!message.usage) continue;
    turns += 1;
    tokens += message.usage.total_tokens;
    if (message.usage.cost_usd != null) {
      cost += message.usage.cost_usd;
      hasCost = true;
    }
    if (message.usage.model) model = message.usage.model;
  }

  return { turns, tokens, cost: hasCost ? cost : null, model };
}

/** Stable enough for React keys within one session. */
const nextId = (() => {
  let n = 0;
  return () => `m${++n}`;
})();

function describe(error: unknown): { title: string; detail: string } {
  if (error instanceof ApiError) {
    if (error.isNotImplemented) {
      return {
        title: "The engine is not configured",
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
  /** Null until someone picks -- the YAML's own agent applies until then. */
  const [agent, setAgent] = useState<string | null>(null);
  const [adminOpen, setAdminOpen] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);
  /** This conversation. A new chat is a new id; memory is keyed on the
   *  browser's client id instead, so it carries over. */
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  /** Bumped after each answer so the memory panel re-reads what the turn stored. */
  const [turn, setTurn] = useState(0);

  /**
   * Start a new conversation: a new thread id, and no history carried over.
   *
   * Memory is not cleared. It belongs to the person, not the thread, which is
   * what makes it long-term; use Forget everything in the memory panel to clear
   * it.
   */
  const newChat = () => {
    abort.current?.abort();
    setMessages([]);
    setSessionId(crypto.randomUUID());
    setTurn((n) => n + 1);
  };
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
          {
            message: text,
            history,
            model: model ?? undefined,
            agent: agent ?? undefined,
            session_id: sessionId,
          },
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
        // The assistant may have written a note mid-answer, through a tool the
        // browser never sees. Nudge the memory panel to re-read.
        setTurn((n) => n + 1);
        abort.current = null;
      }
    },
    [messages, patch, model, agent, sessionId],
  );

  return (
    <div className="app">
      <header className="header">
        <div className="header__row">
          <h1 className="header__title">SkyDesk Support</h1>
          <div className="header__actions">
            <button
              type="button"
              className="btn btn--ghost"
              onClick={() => setMemoryOpen(true)}
            >
              Memory
            </button>
            <button
              className="btn btn--ghost"
              onClick={newChat}
              disabled={busy || messages.length === 0}
            >
              New chat
            </button>
            <button
              className="btn btn--ghost"
              onClick={() => setAdminOpen(true)}
            >
              Admin dashboard
            </button>
            <ExportMenu
              onExport={(format) => exportConversation(messages, format)}
              disabled={messages.length === 0}
            />
          </div>
        </div>
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

        <Knowledge usage={sessionUsage(messages)} />
      </main>

      <footer className="footer">
        <Composer
          onSend={send}
          onStop={() => abort.current?.abort()}
          busy={busy}
        >
          <ModelPicker value={model} onChange={setModel} disabled={busy} />
          <AgentPicker value={agent} onChange={setAgent} disabled={busy} />
        </Composer>
      </footer>

      {adminOpen && <Admin onClose={() => setAdminOpen(false)} />}
      {memoryOpen && (
        <MemoryPanel
          turn={turn}
          onClose={() => setMemoryOpen(false)}
        />
      )}
    </div>
  );
}
