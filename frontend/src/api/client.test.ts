/**
 * The SSE reader.
 *
 * This is where the frontend's real bug class lives: the browser hands us bytes,
 * not frames, so a chunk can split a frame anywhere. Those bugs are invisible on
 * a fast local connection and show up only under load, which is exactly why they
 * are worth pinning here.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  listBookings,
  runSystemPromptEval,
  setAdminKey,
  streamChat,
} from "./client";
import type { ChatEvent } from "./types";

/** A Response whose body streams the given pieces, one per read. */
function streaming(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status });
}

function mockFetch(response: Response | (() => Promise<Response>)) {
  const fn = typeof response === "function" ? response : async () => response;
  vi.stubGlobal("fetch", vi.fn(fn));
}

async function collect(): Promise<ChatEvent[]> {
  const events: ChatEvent[] = [];
  for await (const event of streamChat({ message: "hi" })) events.push(event);
  return events;
}

const frame = (event: ChatEvent) =>
  `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;

afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
});

describe("frame reassembly", () => {
  it("reads several frames arriving in one chunk", async () => {
    mockFetch(
      streaming([
        frame({ type: "token", text: "One " }) +
          frame({ type: "token", text: "two." }) +
          frame({ type: "done", finish_reason: "stop" }),
      ]),
    );

    expect((await collect()).map((e) => e.type)).toEqual([
      "token",
      "token",
      "done",
    ]);
  });

  it("reads one frame split across two chunks", async () => {
    // The split falls inside the JSON payload, which is the case that breaks a
    // naive implementation.
    const whole = frame({ type: "token", text: "hello" });
    const cut = Math.floor(whole.length / 2);

    mockFetch(streaming([whole.slice(0, cut), whole.slice(cut)]));

    const [event] = await collect();
    expect(event).toEqual({ type: "token", text: "hello" });
  });

  it("reads a frame whose separator itself is split", async () => {
    // The blank line between frames is "\n\n"; a chunk can end between them.
    const a = frame({ type: "token", text: "a" });
    mockFetch(streaming([a.slice(0, a.length - 1), "\n" + frame({ type: "done", finish_reason: "stop" })]));

    expect((await collect()).map((e) => e.type)).toEqual(["token", "done"]);
  });

  it("delivers a byte-at-a-time stream intact", async () => {
    const whole =
      frame({ type: "token", text: "drip" }) +
      frame({ type: "usage", input_tokens: 1, output_tokens: 2, total_tokens: 3 });

    mockFetch(streaming(whole.split("")));

    const events = await collect();
    expect(events).toHaveLength(2);
    expect(events[0]).toEqual({ type: "token", text: "drip" });
  });

  it("handles a multi-byte character split down the middle", async () => {
    // "→" is three bytes in UTF-8. A decoder that does not stream would emit a
    // replacement character here.
    const whole = frame({ type: "token", text: "Berlin → Amsterdam" });
    const bytes = new TextEncoder().encode(whole);
    const split = whole.indexOf("→") + 1; // lands inside the arrow's bytes

    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, split));
        controller.enqueue(bytes.slice(split));
        controller.close();
      },
    });
    mockFetch(new Response(body, { status: 200 }));

    const [event] = await collect();
    expect(event).toEqual({ type: "token", text: "Berlin → Amsterdam" });
  });
});

describe("robustness", () => {
  it("ignores keep-alive blank lines and comments", async () => {
    mockFetch(
      streaming([
        "\n\n: keep-alive\n\n" + frame({ type: "done", finish_reason: "stop" }),
      ]),
    );

    expect((await collect()).map((e) => e.type)).toEqual(["done"]);
  });

  it("passes through an event type it does not know", async () => {
    // The App ignores these. Dropping them here would hide a newer backend's
    // events instead of letting the caller decide.
    mockFetch(streaming(['event: future\ndata: {"type":"future","x":1}\n\n']));

    const [event] = await collect();
    expect(event).toMatchObject({ type: "future" });
  });

  it("skips a frame whose data is not JSON rather than ending the stream", async () => {
    mockFetch(
      streaming([
        "event: token\ndata: {not json\n\n" +
          frame({ type: "done", finish_reason: "stop" }),
      ]),
    );

    expect((await collect()).map((e) => e.type)).toEqual(["done"]);
  });

  it("drops a frame with no data line", async () => {
    mockFetch(streaming(["event: token\n\n" + frame({ type: "done", finish_reason: "stop" })]));

    expect((await collect()).map((e) => e.type)).toEqual(["done"]);
  });
});

describe("errors", () => {
  it("reports 501 as not-implemented, with the backend's detail", async () => {
    mockFetch(
      new Response(JSON.stringify({ detail: "no Agent -- get_agent()" }), {
        status: 501,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const error = await collect().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).isNotImplemented).toBe(true);
    expect((error as ApiError).message).toContain("get_agent");
  });

  it("reports 503 as the engine being down", async () => {
    mockFetch(
      new Response(JSON.stringify({ detail: "engine unreachable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const error = (await collect().catch((e: unknown) => e)) as ApiError;
    expect(error.isEngineDown).toBe(true);
    expect(error.isNotImplemented).toBe(false);
  });

  it("throws before yielding anything, so a route can still set its status", async () => {
    mockFetch(new Response("nope", { status: 500 }));

    const events: ChatEvent[] = [];
    await expect(async () => {
      for await (const event of streamChat({ message: "hi" })) events.push(event);
    }).rejects.toBeInstanceOf(ApiError);

    expect(events).toEqual([]);
  });

  it("falls back to the status text when the error body is not JSON", async () => {
    mockFetch(new Response("gateway exploded", { status: 502 }));

    const error = (await collect().catch((e: unknown) => e)) as ApiError;
    expect(error.status).toBe(502);
  });
});

describe("the request", () => {
  it("posts JSON to the chat endpoint", async () => {
    const spy = vi.fn(
      async (_url: RequestInfo | URL, _init?: RequestInit): Promise<Response> =>
        streaming([frame({ type: "done", finish_reason: "stop" })]),
    );
    vi.stubGlobal("fetch", spy);

    for await (const _ of streamChat({ message: "hello", history: [] })) {
      // drain
    }

    const call = spy.mock.calls[0];
    expect(call).toBeDefined();
    const [url, init] = call!;
    expect(String(url)).toContain("/chat");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ message: "hello", history: [] });
  });
});

describe("the admin key", () => {
  /** Capture the init of the one request the call under test makes. */
  function spyOnFetch(body: unknown = [], status = 200) {
    const spy = vi.fn(
      async (
        _url: RequestInfo | URL,
        _init?: RequestInit,
      ): Promise<Response> =>
        new Response(JSON.stringify(body), {
          status,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", spy);
    return spy;
  }

  it("sends no header when no key is held, so an open backend still works", async () => {
    const spy = spyOnFetch();

    await listBookings();

    const headers = spy.mock.calls[0]![1]?.headers as Record<string, string>;
    expect(headers).toEqual({});
  });

  it("sends the stored key on admin GETs", async () => {
    setAdminKey("operator-key");
    const spy = spyOnFetch();

    await listBookings();

    const headers = spy.mock.calls[0]![1]?.headers as Record<string, string>;
    expect(headers["X-Admin-Key"]).toBe("operator-key");
  });

  it("sends the stored key on the eval POST as well", async () => {
    setAdminKey("operator-key");
    const spy = spyOnFetch({ rows: [] });

    await runSystemPromptEval("greeting");

    const init = spy.mock.calls[0]![1]!;
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["X-Admin-Key"]).toBe(
      "operator-key",
    );
  });

  it("surfaces a refused key as a 401 ApiError for the dashboard to catch", async () => {
    spyOnFetch({ detail: "missing or invalid X-Admin-Key" }, 401);

    const error = (await listBookings().catch((e: unknown) => e)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(401);
  });
});
