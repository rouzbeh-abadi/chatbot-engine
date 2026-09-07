import { useEffect, useState } from "react";
import { listAgents } from "../api/client";

/**
 * Which agent runs the turn, for the rest of the conversation.
 *
 * The options come from the engine rather than a list held here: which agents
 * exist depends on what is installed there, so an adopter who adds their own
 * sees it offered without touching the frontend.
 *
 * `value` is null until someone picks: the first entry renders as selected and
 * the request omits the field, so the YAML's own `agent:` applies. A change
 * takes effect on the next turn, with the history intact.
 *
 * Hidden when the engine offers fewer than two, since a choice of one is not a
 * choice.
 */
export function AgentPicker({
  value,
  onChange,
  disabled,
}: {
  value: string | null;
  onChange: (id: string) => void;
  disabled: boolean;
}) {
  const [agents, setAgents] = useState<string[]>([]);

  useEffect(() => {
    let live = true;
    listAgents()
      .then((found) => live && setAgents(found))
      // A picker that cannot load is not worth an error bar: the composer still
      // works, and the backend still has a default.
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  if (agents.length < 2) return null;

  return (
    <label className="model" aria-label="Agent">
      <select
        className="model__select"
        value={value ?? agents[0]}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      >
        {agents.map((agent) => (
          <option key={agent} value={agent}>
            {agent}
          </option>
        ))}
      </select>
    </label>
  );
}
