import type { ToolCall } from "../api/types";

/**
 * Tool activity, as a row of chips.
 *
 * A call with no `ok` yet is still running -- that is the progress indicator for
 * the slowest part of a turn, so it renders as soon as `tool_call_started`
 * arrives rather than waiting for the result.
 */
export function ToolCalls({ calls }: { calls: ToolCall[] }) {
  if (calls.length === 0) return null;

  return (
    <ul className="tools" aria-label="Tools used">
      {calls.map((call) => {
        const running = call.ok === undefined;
        const state = running ? "running" : call.ok ? "ok" : "failed";
        return (
          <li key={call.call_id} className={`tool tool--${state}`}>
            <span className="tool__dot" aria-hidden="true" />
            <code className="tool__name">{call.tool}</code>
            {running && <span className="tool__meta">running…</span>}
            {!running && call.duration_ms != null && (
              <span className="tool__meta">{call.duration_ms} ms</span>
            )}
            {call.error && <span className="tool__error">{call.error}</span>}
          </li>
        );
      })}
    </ul>
  );
}
