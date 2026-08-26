import { useEffect, useState } from "react";
import { listModels } from "../api/client";

/**
 * Which model answers, for the rest of the conversation.
 *
 * Only the id travels, and the backend checks it against its own allowlist --
 * a browser that could name any model freely could spend on any model freely.
 * Everything else about the assistant stays in `projects/support.yaml`.
 *
 * `value` is null until someone picks: the first entry renders as selected and
 * the request omits the field, so the YAML's own `model:` applies. A change
 * takes effect on the next turn, with the history intact.
 */
export function ModelPicker({
  value,
  onChange,
  disabled,
}: {
  value: string | null;
  onChange: (id: string) => void;
  disabled: boolean;
}) {
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    let live = true;
    listModels()
      .then((found) => live && setModels(found))
      // A picker that cannot load is not worth an error bar: the composer still
      // works, and the backend still has a default.
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  const first = models[0];
  if (first === undefined) return null;

  return (
    <label className="model" aria-label="Model">
      <select
        className="model__select"
        value={value ?? first}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      >
        {models.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
    </label>
  );
}
