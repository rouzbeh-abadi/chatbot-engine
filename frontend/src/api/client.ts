/**
 * The only module that talks to the backend.
 *
 * The chat endpoint streams server-sent events, and `EventSource` cannot issue a
 * POST -- so this reads the response body directly. That means owning two details
 * the browser would otherwise handle: reassembling frames that a network chunk
 * split in half, and checking the status *before* consuming the body.
 */

import type { ChatEvent, ChatRequest, DocumentRecord } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

/** A backend response that was not a success, with the detail it explained itself with. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** The engine is reachable but that capability has no implementation yet. */
  get isNotImplemented(): boolean {
    return this.status === 501;
  }

  /** The engine service is not running. */
  get isEngineDown(): boolean {
    return this.status === 503;
  }
}

async function detailOf(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") return body.detail;
    return JSON.stringify(body);
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}

/**
 * Send a message and yield events as they arrive.
 *
 * The status check happens before the first yield, so a 501 or 503 surfaces as a
 * thrown `ApiError` rather than as a stream that produces nothing.
 */
export async function* streamChat(
  request: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new ApiError(response.status, await detailOf(response));
  }
  if (!response.body) {
    throw new ApiError(response.status, "the backend sent no response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Frames are separated by a blank line. A chunk can end mid-frame, so
      // anything after the last separator stays buffered for the next read.
      let split: number;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        const event = parseFrame(frame);
        if (event) yield event;
      }
    }
  } finally {
    // Releasing the lock lets an aborted request tear the connection down
    // instead of leaving it open until the tab closes.
    reader.releaseLock();
  }
}

/**
 * Turn one SSE frame into an event.
 *
 * Only the `data:` line matters -- the type is inside the JSON as well, so the
 * `event:` line is redundant here. Returns null for anything unparseable, which
 * keeps a malformed or unfamiliar frame from ending the conversation.
 */
function parseFrame(frame: string): ChatEvent | null {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("");

  if (!data) return null;

  try {
    const parsed = JSON.parse(data);
    return typeof parsed?.type === "string" ? (parsed as ChatEvent) : null;
  } catch {
    return null;
  }
}

/** What the assistant currently knows. */
export async function listDocuments(): Promise<DocumentRecord[]> {
  const response = await fetch(`${BASE}/documents`);
  if (!response.ok) {
    throw new ApiError(response.status, await detailOf(response));
  }
  return (await response.json()) as DocumentRecord[];
}
