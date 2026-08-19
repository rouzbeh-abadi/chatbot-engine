/**
 * The event loop, through the real components.
 *
 * These drive the app the way a person does -- type, send, wait -- so what is
 * asserted is what actually reaches the screen: tool chips while a tool runs,
 * sources, cost, and a legible message when the engine is not ready.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { ChatEvent } from "./api/types";

const frame = (event: ChatEvent) =>
  `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;

function sseResponse(events: ChatEvent[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) controller.enqueue(encoder.encode(frame(event)));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

/** Route /documents and /chat separately, the way the backend does. */
function routes(chat: () => Promise<Response>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL) => {
      if (String(url).includes("/documents")) {
        return new Response("[]", {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return chat();
    }),
  );
}

/**
 * Render, then wait for the knowledge panel's mount-time fetch to land.
 *
 * Without the wait its state update resolves after the test body has finished,
 * which React reports as an unwrapped act() warning -- noise that trains you to
 * ignore warnings.
 */
async function renderApp() {
  render(<App />);
  await screen.findByText(/Nothing indexed yet/);
}

async function ask(question = "Is my flight delayed?") {
  const box = screen.getByLabelText("Your message") as HTMLTextAreaElement;
  const { fireEvent } = await import("@testing-library/react");
  fireEvent.change(box, { target: { value: question } });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
}

beforeEach(() => {
  vi.stubGlobal("scrollIntoView", vi.fn());
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => vi.unstubAllGlobals());

describe("a successful turn", () => {
  const events: ChatEvent[] = [
    {
      type: "retrieval",
      query: "Is my flight delayed?",
      sources: [
        {
          doc_id: "d1",
          source: "baggage.md",
          score: 0.91,
          heading: "Baggage Policy > Cabin Baggage",
          excerpt: "One cabin bag up to 8 kg.",
        },
      ],
    },
    {
      type: "tool_call_started",
      call_id: "c1",
      tool: "get_booking_status",
      server: "support-tools",
      arguments: { booking_reference: "AB12CD" },
    },
    {
      type: "tool_call_finished",
      call_id: "c1",
      tool: "get_booking_status",
      ok: true,
      duration_ms: 142,
    },
    { type: "token", text: "Your flight " },
    { type: "token", text: "is on time." },
    {
      type: "usage",
      input_tokens: 1284,
      output_tokens: 48,
      total_tokens: 1332,
      cost_usd: 0.00061,
      model: "openai/gpt-5-mini",
    },
    { type: "done", finish_reason: "stop" },
  ];

  it("shows the question, the streamed answer, and the tool that ran", async () => {
    routes(async () => sseResponse(events));
    await renderApp();

    await ask();

    await waitFor(() =>
      expect(screen.getByText("Your flight is on time.")).toBeInTheDocument(),
    );
    expect(screen.getByText("Is my flight delayed?")).toBeInTheDocument();
    expect(screen.getByText("get_booking_status")).toBeInTheDocument();
    expect(screen.getByText("142 ms")).toBeInTheDocument();
  });

  it("reports token usage and cost", async () => {
    routes(async () => sseResponse(events));
    await renderApp();

    await ask();

    await waitFor(() =>
      expect(screen.getByText(/1,332 tokens/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/\$0\.00061/)).toBeInTheDocument();
    expect(screen.getByText(/openai\/gpt-5-mini/)).toBeInTheDocument();
  });

  it("keeps sources collapsed until asked, then shows the citation", async () => {
    routes(async () => sseResponse(events));
    await renderApp();

    await ask();

    const toggle = await screen.findByRole("button", { name: /1 source/ });
    expect(screen.queryByText("baggage.md")).not.toBeInTheDocument();

    const { fireEvent } = await import("@testing-library/react");
    fireEvent.click(toggle);

    expect(screen.getByText("baggage.md")).toBeInTheDocument();
    expect(screen.getByText("Baggage Policy > Cabin Baggage")).toBeInTheDocument();
    expect(screen.getByText("One cabin bag up to 8 kg.")).toBeInTheDocument();
  });

  it("marks a failed tool without losing the answer", async () => {
    routes(async () =>
      sseResponse([
        {
          type: "tool_call_started",
          call_id: "c1",
          tool: "get_flight_status",
          arguments: {},
        },
        {
          type: "tool_call_finished",
          call_id: "c1",
          tool: "get_flight_status",
          ok: false,
          error: "upstream timeout",
        },
        { type: "token", text: "I could not check that." },
        { type: "done", finish_reason: "stop" },
      ]),
    );
    await renderApp();

    await ask();

    await waitFor(() =>
      expect(screen.getByText("I could not check that.")).toBeInTheDocument(),
    );
    expect(screen.getByText("upstream timeout")).toBeInTheDocument();
  });
});

describe("when the engine is not ready", () => {
  it("explains a 501 and names the function to implement", async () => {
    routes(async () =>
      new Response(
        JSON.stringify({
          detail:
            "no Agent is registered -- return it from chatbot_engine.api.deps.get_agent()",
        }),
        { status: 501, headers: { "Content-Type": "application/json" } },
      ),
    );
    await renderApp();

    await ask();

    await waitFor(() =>
      expect(screen.getByText("The engine has no agent yet")).toBeInTheDocument(),
    );
    expect(screen.getByText(/get_agent\(\)/)).toBeInTheDocument();
  });

  it("distinguishes the engine being down, and says how to start it", async () => {
    routes(async () =>
      new Response(JSON.stringify({ detail: "engine unreachable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await renderApp();

    await ask();

    await waitFor(() =>
      expect(
        screen.getByText("The engine service is not running"),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText(/make engine/)).toBeInTheDocument();
  });

  it("shows a mid-stream error event as a problem, not a silent stop", async () => {
    routes(async () =>
      sseResponse([
        { type: "token", text: "partial" },
        { type: "error", code: "engine_error", message: "retriever exploded" },
        { type: "done", finish_reason: "error" },
      ]),
    );
    await renderApp();

    await ask();

    await waitFor(() =>
      expect(screen.getByText("The answer failed")).toBeInTheDocument(),
    );
    expect(screen.getByText("retriever exploded")).toBeInTheDocument();
  });

  it("says the backend is unreachable when fetch itself fails", async () => {
    routes(async () => {
      throw new TypeError("Failed to fetch");
    });
    await renderApp();

    await ask();

    await waitFor(() =>
      expect(
        screen.getByText("Could not reach the backend"),
      ).toBeInTheDocument(),
    );
  });
});

describe("the composer", () => {
  it("will not send an empty message", async () => {
    routes(async () => sseResponse([{ type: "done", finish_reason: "stop" }]));
    await renderApp();

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("offers example questions before anything is asked", async () => {
    routes(async () => sseResponse([]));
    await renderApp();

    expect(screen.getByText(/Nothing asked yet/)).toBeInTheDocument();
  });
});

describe("markdown in the answer", () => {
  it("renders the assistant's markdown but leaves the user's text literal", async () => {
    routes(async () =>
      sseResponse([
        { type: "token", text: "Booking **XY34ZT**:\n\n- one bag\n- 8 kg\n" },
        { type: "done", finish_reason: "stop" },
      ]),
    );
    await renderApp();

    await ask("Is **this** literal?");

    await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(2));
    // The assistant's markers became elements...
    expect(screen.getByText("XY34ZT").tagName).toBe("STRONG");
    // ...while the user's stayed exactly as typed. (The page header also shows a
    // booking reference, hence the different one above.)
    expect(screen.getByText("Is **this** literal?")).toBeInTheDocument();
  });

  it("keeps the answer out of the preformatted container", async () => {
    // `.msg__text` carries `white-space: pre-wrap` so a user's line breaks
    // survive. Rendered markdown must not inherit it, or every newline between
    // block elements becomes a visible blank line -- a bug that is invisible in
    // jsdom and obvious in a browser, so the structure is pinned here instead.
    routes(async () =>
      sseResponse([
        { type: "token", text: "- a\n- b\n" },
        { type: "done", finish_reason: "stop" },
      ]),
    );
    const { container } = render(<App />);
    await screen.findByText(/Nothing indexed yet/);

    await ask("list please");
    await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(2));

    const answer = container.querySelector(".answer");
    expect(answer).not.toBeNull();
    expect(answer!.closest(".msg__text")).toBeNull();
    expect(answer!.closest(".msg__answer")).not.toBeNull();
  });
});
