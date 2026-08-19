/**
 * The backend's wire contract, as this frontend sees it.
 *
 * A deliberate copy, not a generated client: the frontend is a separate
 * deployable and the contract crosses a network boundary. Keep it in step with
 * `backend/src/support_agent/engine_client/models.py`.
 *
 * Every event carries a `type`, so unknown ones can be ignored rather than
 * crashing the stream -- new event types will be added.
 */

export interface SourceRef {
  doc_id: string;
  source: string;
  score: number;
  heading?: string | null;
  excerpt?: string | null;
}

export interface RetrievalEvent {
  type: "retrieval";
  query?: string | null;
  sources: SourceRef[];
}

export interface TokenEvent {
  type: "token";
  text: string;
}

export interface ToolCallStartedEvent {
  type: "tool_call_started";
  call_id: string;
  tool: string;
  server?: string | null;
  arguments: Record<string, unknown>;
}

export interface ToolCallFinishedEvent {
  type: "tool_call_finished";
  call_id: string;
  tool: string;
  ok: boolean;
  duration_ms?: number | null;
  result_preview?: string | null;
  error?: string | null;
}

export interface UsageEvent {
  type: "usage";
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd?: number | null;
  model?: string | null;
}

export interface ErrorEvent {
  type: "error";
  code: string;
  message: string;
}

export interface DoneEvent {
  type: "done";
  finish_reason: "stop" | "length" | "tool_limit" | "error" | "cancelled";
}

export type ChatEvent =
  | RetrievalEvent
  | TokenEvent
  | ToolCallStartedEvent
  | ToolCallFinishedEvent
  | UsageEvent
  | ErrorEvent
  | DoneEvent;

export interface ChatRequest {
  message: string;
  session_id?: string;
  project?: string;
  history?: { role: "user" | "assistant" | "system"; content: string }[];
}

export interface DocumentRecord {
  doc_id: string;
  external_id: string;
  filename: string;
  size_bytes: number;
  status: string;
  chunk_count: number;
  error?: string | null;
  updated_at?: string | null;
}

/** A tool call as the UI tracks it: started, then possibly finished. */
export interface ToolCall {
  call_id: string;
  tool: string;
  server?: string | null;
  arguments: Record<string, unknown>;
  ok?: boolean;
  duration_ms?: number | null;
  error?: string | null;
}
