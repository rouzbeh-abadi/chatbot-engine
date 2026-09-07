import { useCallback, useEffect, useState } from "react";
import { forgetMemory, listMemory } from "../api/client";
import type { MemoryRow } from "../api/types";

/**
 * What the assistant has written down about you.
 *
 * A debug view, and the only place a person can see or erase it. Notes belong
 * to the browser's user rather than to the conversation, so they survive New
 * chat; this panel is what makes that visible rather than something you have to
 * take on trust.
 *
 * Refreshed when opened and after each turn, since the assistant writes through
 * a tool mid-answer and the panel has no other way to know.
 */
export function MemoryPanel({
  turn,
  onClose,
}: {
  /** Bumped after every answer, to re-read what the turn may have stored. */
  turn: number;
  onClose: () => void;
}) {
  const [rows, setRows] = useState<MemoryRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listMemory()
      .then(setRows)
      .catch(() => setError("Could not read memory."));
  }, []);

  useEffect(refresh, [refresh, turn]);

  const clear = async () => {
    try {
      await forgetMemory();
      setRows([]);
    } catch {
      setError("Could not clear memory.");
    }
  };

  return (
    <div className="admin" role="dialog" aria-label="Memory">
      <div className="admin__head">
        <h2 className="admin__title">Memory</h2>
        <div className="admin__tabs" style={{ border: 0, padding: 0 }}>
          <button className="btn btn--ghost" onClick={clear} disabled={!rows?.length}>
            Forget everything
          </button>
          <button className="btn btn--ghost" onClick={onClose}>
            Close
          </button>
        </div>
      </div>

      <div className="admin__body">
        <div className="admin__stack">
          <p className="admin__note">
            What the assistant has stored about you, through its
            <code> remember </code> tool. Notes follow you rather than the chat,
            so they survive starting a new one.
          </p>

          {error && <p className="admin__note admin__note--bad">{error}</p>}

          {rows === null && !error && <p className="admin__note">Loading…</p>}

          {rows?.length === 0 && (
            <p className="admin__note">
              Nothing stored yet. Tell the assistant a lasting preference, such
              as “I always book an aisle seat”, and it will appear here.
            </p>
          )}

          {rows && rows.length > 0 && (
            <div className="admin__scroll">
              <table className="admin__table">
                <thead>
                  <tr>
                    <th>Subject</th>
                    <th>Remembered</th>
                    <th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.subject}>
                      <td className="mono">{row.subject}</td>
                      <td>{row.content}</td>
                      <td className="mono">
                        {row.updated_at.slice(0, 16).replace("T", " ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
