import type { SourceRef, ToolCall, UsageEvent } from "../api/types";
import { Answer } from "./Answer";
import { ToolCalls } from "./ToolCalls";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources: SourceRef[];
  toolCalls: ToolCall[];
  usage?: UsageEvent;
  /** Set when the turn failed. Rendered instead of an empty answer. */
  problem?: { title: string; detail: string };
  streaming: boolean;
}

export function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const empty = !message.text && !message.problem;

  return (
    <article className={`msg msg--${message.role}`}>
      <div className="msg__who">{isUser ? "You" : "Assistant"}</div>

      <div className="msg__body">
        {!isUser && <ToolCalls calls={message.toolCalls} />}

        {message.problem ? (
          <div className="problem" role="alert">
            <strong className="problem__title">{message.problem.title}</strong>
            <p className="problem__detail">{message.problem.detail}</p>
          </div>
        ) : isUser ? (
          <p className="msg__text">{message.text}</p>
        ) : (
          <div className="msg__answer">
            <Answer text={message.text} sources={message.sources} />
            {message.streaming && <span className="caret" aria-hidden="true" />}
          </div>
        )}

        {/* Before the first token there is nothing to show but the fact that
            something is happening. */}
        {empty && message.streaming && message.toolCalls.length === 0 && (
          <p className="thinking" aria-live="polite">
            Thinking<span className="thinking__dots" />
          </p>
        )}
      </div>
    </article>
  );
}
