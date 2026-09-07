/**
 * Export the conversation to a file.
 *
 * Only the conversation itself -- who said what -- is exported. Sources, tools
 * and token usage are shown in the UI but left out of the export, which is meant
 * to be a plain, portable transcript.
 */

import { jsPDF } from "jspdf";
import type { ChatMessage } from "./components/Message";

export type ExportFormat = "json" | "csv" | "pdf";

/** The formats the export menu offers, in order. */
export const EXPORT_FORMATS: ExportFormat[] = ["json", "csv", "pdf"];

interface Turn {
  role: "user" | "assistant";
  text: string;
}

/** The conversation as plain turns, dropping empty and failed messages. */
function turns(messages: ChatMessage[]): Turn[] {
  return messages
    .filter((message) => message.text && !message.problem)
    .map((message) => ({ role: message.role, text: message.text }));
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function toJson(rows: Turn[]): Blob {
  return new Blob([JSON.stringify({ messages: rows }, null, 2)], {
    type: "application/json",
  });
}

/** One `role,text` row per turn. Quotes are doubled so commas and newlines in
 *  the text stay inside the field. */
function toCsv(rows: Turn[]): Blob {
  const escape = (value: string) => `"${value.replace(/"/g, '""')}"`;
  const lines = ["role,text", ...rows.map((r) => `${r.role},${escape(r.text)}`)];
  return new Blob([lines.join("\n")], { type: "text/csv" });
}

/** jsPDF's built-in fonts only render Latin-1. Map smart punctuation to ASCII
 *  and drop anything above U+00FF (emoji, other scripts) -- otherwise a single
 *  unrenderable character corrupts the whole line into mojibake. */
function toLatin1(text: string): string {
  return text
    .replace(/[‘’‚′]/g, "'")
    .replace(/[“”„″]/g, '"')
    .replace(/[–—]/g, "-")
    .replace(/…/g, "...")
    .replace(/[   ]/g, " ")
    .replace(/[^\x00-\xFF]/g, "");
}

function toPdf(rows: Turn[]): Blob {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const margin = 48;
  const width = doc.internal.pageSize.getWidth() - margin * 2;
  const bottom = doc.internal.pageSize.getHeight() - margin;
  let y = margin;

  const line = (text: string, size: number, bold: boolean) => {
    doc.setFont("helvetica", bold ? "bold" : "normal");
    doc.setFontSize(size);
    for (const wrapped of doc.splitTextToSize(toLatin1(text), width)) {
      if (y > bottom) {
        doc.addPage();
        y = margin;
      }
      doc.text(wrapped, margin, y);
      y += size + 4;
    }
  };

  doc.setFontSize(16);
  doc.setFont("helvetica", "bold");
  doc.text("SkyDesk Support — conversation", margin, y);
  y += 28;

  for (const row of rows) {
    line(row.role === "user" ? "You" : "Assistant", 11, true);
    y += 2;
    line(row.text, 11, false);
    y += 14;
  }

  return doc.output("blob");
}

const BUILDERS: Record<ExportFormat, (rows: Turn[]) => Blob> = {
  json: toJson,
  csv: toCsv,
  pdf: toPdf,
};

/** Build the chosen file from the conversation and download it. */
export function exportConversation(
  messages: ChatMessage[],
  format: ExportFormat,
): void {
  const blob = BUILDERS[format](turns(messages));
  download(blob, `skydesk-conversation.${format}`);
}
