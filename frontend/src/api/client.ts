/**
 * The only module that talks to the backend.
 *
 * The chat endpoint streams server-sent events, and `EventSource` cannot issue a
 * POST -- so this reads the response body directly. That means owning two details
 * the browser would otherwise handle: reassembling frames that a network chunk
 * split in half, and checking the status *before* consuming the body.
 */

import type {
  BookingRow,
  ChatEvent,
  ChatRequest,
  DocumentRecord,
  EvalCaseInfo,
  EvalRunResult,
  RagReport,
  TicketRow,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

/**
 * Where the operator key for `/admin` is kept.
 *
 * `sessionStorage`, not a build-time variable: baking the key into the bundle
 * would ship it to every visitor, which is the same as having no key at all.
 * The operator types it once per tab and it dies with the tab.
 */
const ADMIN_KEY_ITEM = "support-agent.admin-key";

/** The admin key this tab is holding, if any. */
export function getAdminKey(): string | null {
  try {
    return sessionStorage.getItem(ADMIN_KEY_ITEM);
  } catch {
    // Storage can be blocked entirely; the dashboard just asks again.
    return null;
  }
}

/** Remember the admin key for the rest of this tab's life. */
export function setAdminKey(key: string): void {
  try {
    sessionStorage.setItem(ADMIN_KEY_ITEM, key);
  } catch {
    // Nothing to do -- the caller's request carries the key regardless.
  }
}

/**
 * The header the admin routes want, or nothing when no key is held.
 *
 * Sending no header is right rather than sloppy: a backend with no
 * `BACKEND_ADMIN_KEY` set leaves `/admin` open, which is how localhost runs.
 */
function adminHeaders(): Record<string, string> {
  const key = getAdminKey();
  return key ? { "X-Admin-Key": key } : {};
}

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

/** The models the backend allows the UI to pick, the default first. */
export async function listModels(): Promise<string[]> {
  return getJson<string[]>("/models");
}

/** GET a JSON endpoint, turning a non-2xx into an `ApiError`. */
async function getJson<T>(
  path: string,
  headers?: Record<string, string>,
): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { headers });
  if (!response.ok) {
    throw new ApiError(response.status, await detailOf(response));
  }
  return (await response.json()) as T;
}

/** Every booking in the database (admin). */
export async function listBookings(): Promise<BookingRow[]> {
  return getJson<BookingRow[]>("/admin/bookings", adminHeaders());
}

/** Every support ticket (admin). */
export async function listTickets(): Promise<TicketRow[]> {
  return getJson<TicketRow[]>("/admin/tickets", adminHeaders());
}

/** The dataset's cases, for the run selector. */
export async function listEvalCases(): Promise<EvalCaseInfo[]> {
  return getJson<EvalCaseInfo[]>(
    "/admin/eval/system-prompt/cases",
    adminHeaders(),
  );
}

/**
 * Run the system-prompt evaluation and return the graded run.
 *
 * This makes one model call per case, so it can take a while; the caller should
 * show a progress state. `only` runs a single category or one case.
 */
export async function runSystemPromptEval(
  only?: string,
): Promise<EvalRunResult> {
  const query = only ? `?only=${encodeURIComponent(only)}` : "";
  const response = await fetch(`${BASE}/admin/eval/system-prompt${query}`, {
    method: "POST",
    headers: adminHeaders(),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await detailOf(response));
  }
  return (await response.json()) as EvalRunResult;
}

/** The retrieval cases, for the RAG run selector. */
export async function listRagCases(): Promise<EvalCaseInfo[]> {
  return getJson<EvalCaseInfo[]>("/admin/eval/rag/cases", adminHeaders());
}

/**
 * Run the RAG evaluation and return the scored run.
 *
 * Each case is answered and then graded by several RAGAS metric calls, so this
 * is slower than the system-prompt eval; the caller should show a progress
 * state. `only` runs a single category or one case.
 */
export async function runRagEval(only?: string): Promise<RagReport> {
  const query = only ? `?only=${encodeURIComponent(only)}` : "";
  const response = await fetch(`${BASE}/admin/eval/rag${query}`, {
    method: "POST",
    headers: adminHeaders(),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await detailOf(response));
  }
  return (await response.json()) as RagReport;
}
