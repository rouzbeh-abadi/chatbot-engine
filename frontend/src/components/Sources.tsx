import { useState } from "react";
import type { SourceRef } from "../api/types";

/**
 * The documents an answer was drawn from.
 *
 * Collapsed by default: sources matter for trusting an answer, but they should not
 * push the answer itself off the screen.
 */
export function Sources({ sources }: { sources: SourceRef[] }) {
  const [open, setOpen] = useState(false);

  if (sources.length === 0) return null;

  return (
    <div className="sources">
      <button
        type="button"
        className="sources__toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? "▾" : "▸"} {sources.length} source
        {sources.length === 1 ? "" : "s"}
      </button>

      {open && (
        <ol className="sources__list">
          {sources.map((source, index) => (
            <li key={`${source.doc_id}-${index}`} className="source">
              <div className="source__head">
                <span className="source__file">{source.source}</span>
                <span className="source__score">
                  {(source.score * 100).toFixed(0)}%
                </span>
              </div>
              {source.heading && (
                <div className="source__heading">{source.heading}</div>
              )}
              {source.excerpt && (
                <blockquote className="source__excerpt">
                  {source.excerpt}
                </blockquote>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
