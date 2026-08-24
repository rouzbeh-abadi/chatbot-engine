import { useEffect, useState } from "react";
import { ApiError, listDocuments } from "../api/client";
import type { DocumentRecord } from "../api/types";

/** Running totals across every answered turn this session. */
export interface SessionUsage {
  turns: number;
  tokens: number;
  cost: number | null;
  model?: string;
}

/**
 * What the assistant knows, plus this session's token usage.
 *
 * This calls the same ingestion path the engine owns, so it reports 501 the same
 * way the chat does -- showing the state honestly is more useful than an empty
 * list that looks like a working but empty knowledge base.
 */
export function Knowledge({ usage }: { usage: SessionUsage }) {
  const [docs, setDocs] = useState<DocumentRecord[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    listDocuments()
      .then((result) => live && setDocs(result))
      .catch((error: unknown) => {
        if (!live) return;
        setProblem(
          error instanceof ApiError
            ? error.isNotImplemented
              ? "The engine has no document registry registered yet."
              : error.isEngineDown
                ? "The engine service is not running."
                : error.message
            : "Could not reach the backend.",
        );
      });
    return () => {
      live = false;
    };
  }, []);

  return (
    <aside className="panel">
      <h2 className="panel__title">Knowledge base</h2>

      {problem && <p className="panel__note">{problem}</p>}

      {docs && docs.length === 0 && (
        <p className="panel__note">
          Nothing indexed yet. Run <code>make seed</code>.
        </p>
      )}

      {docs && docs.length > 0 && (
        <ul className="docs">
          {docs.map((doc) => (
            <li key={doc.doc_id} className="doc">
              <span className="doc__name">{doc.external_id}</span>
              <span className="doc__meta">
                {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"} ·{" "}
                {doc.status}
              </span>
            </li>
          ))}
        </ul>
      )}

      {!docs && !problem && <p className="panel__note">Loading…</p>}

      {usage.turns > 0 && (
        <div className="usage-total">
          <h3 className="usage-total__title">Session usage</h3>
          <p className="usage-total__line">
            {usage.tokens.toLocaleString()} tokens · {usage.turns} turn
            {usage.turns === 1 ? "" : "s"}
            {usage.cost != null && ` · $${usage.cost.toFixed(5)}`}
          </p>
          {usage.model && (
            <p className="usage-total__model">{usage.model}</p>
          )}
        </div>
      )}
    </aside>
  );
}
