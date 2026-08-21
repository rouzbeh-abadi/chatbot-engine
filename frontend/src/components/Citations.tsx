import type { ReactNode } from "react";
import type { SourceRef } from "../api/types";

/** A run of markers, `[2]` or `[1][3]`, with any space in front of it. */
const RUN = /\s*(?:\[\d+\])+/g;
const NUMBER = /\[(\d+)\]/g;

/**
 * Replace every citation marker in a block with a chip naming its document.
 *
 * Not just the trailing one: models put a marker at the end of each sentence,
 * and several sentences separated by single newlines are one Markdown paragraph.
 * Anything left un-replaced shows up as literal `[1]` in the answer.
 *
 * A run whose numbers name no source is left alone -- it is the document's own
 * text, not a citation.
 */
export function withCitations(
  children: ReactNode[],
  sources: SourceRef[],
): ReactNode[] {
  return children.flatMap((child, position) => {
    if (typeof child !== "string") return [child];

    const parts: ReactNode[] = [];
    let taken = 0;

    for (const run of child.matchAll(RUN)) {
      const cited = unique(
        [...run[0].matchAll(NUMBER)]
          .map((marker) => sources[Number(marker[1]) - 1])
          .filter((source): source is SourceRef => Boolean(source)),
      );

      if (cited.length === 0) continue;

      parts.push(child.slice(taken, run.index));
      parts.push(
        ...cited.map((source) => (
          <span
            key={`${position}-${run.index}-${source.source}`}
            className="cite"
            title={source.excerpt ?? undefined}
          >
            {source.source}
          </span>
        )),
      );
      taken = run.index + run[0].length;
    }

    parts.push(child.slice(taken));

    return parts;
  });
}

/** Two passages from one file are one citation to a reader. */
function unique(sources: SourceRef[]): SourceRef[] {
  const byFile = new Map<string, SourceRef>();

  for (const source of sources) {
    if (!byFile.has(source.source)) byFile.set(source.source, source);
  }

  return [...byFile.values()];
}
