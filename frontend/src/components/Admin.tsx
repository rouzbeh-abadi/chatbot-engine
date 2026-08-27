import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  listBookings,
  listEvalCases,
  listRagCases,
  listTickets,
  runRagEval,
  runSystemPromptEval,
  setAdminKey,
} from "../api/client";
import type {
  BookingRow,
  EvalCaseInfo,
  EvalRunResult,
  RagCaseResult,
  RagMetricAverages,
  RagReport,
  TicketRow,
} from "../api/types";

type Tab = "data" | "prompt-eval" | "rag-eval";

/**
 * The admin dashboard.
 *
 * A full-screen overlay with tabs: the application data (bookings, tickets),
 * and the evaluations. Kept separate from the chat product; opened from the
 * header and closed with Escape or the close button.
 *
 * Whether the backend wants an operator key is not something the UI can know in
 * advance -- `BACKEND_ADMIN_KEY` may or may not be set -- so it asks only once a
 * route has answered 401, and then replays the tab with the key in hand.
 */
export function Admin({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<Tab>("data");
  const [locked, setLocked] = useState(false);
  // Bumped after a key is entered; it is the body's `key`, so the tabs remount
  // and re-run their fetches rather than sitting on the failed one.
  const [attempt, setAttempt] = useState(0);

  const onDenied = useCallback(() => setLocked(true), []);

  const unlock = (key: string) => {
    setAdminKey(key);
    setLocked(false);
    setAttempt((n) => n + 1);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="admin" role="dialog" aria-label="Admin dashboard">
      <div className="admin__head">
        <h2 className="admin__title">Admin dashboard</h2>
        <button className="btn btn--ghost" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="admin__tabs" role="tablist">
        <TabButton active={tab === "data"} onClick={() => setTab("data")}>
          Database
        </TabButton>
        <TabButton
          active={tab === "prompt-eval"}
          onClick={() => setTab("prompt-eval")}
        >
          System-prompt eval
        </TabButton>
        <TabButton active={tab === "rag-eval"} onClick={() => setTab("rag-eval")}>
          RAG eval
        </TabButton>
      </div>

      <div className="admin__body" key={attempt}>
        {locked ? (
          <AdminKeyForm onSubmit={unlock} />
        ) : (
          <>
            {tab === "data" && <DataTab onDenied={onDenied} />}
            {tab === "prompt-eval" && <PromptEvalTab onDenied={onDenied} />}
            {tab === "rag-eval" && <RagEvalTab onDenied={onDenied} />}
          </>
        )}
      </div>
    </div>
  );
}

/** Asks for the operator key after the backend has turned a request down. */
function AdminKeyForm({ onSubmit }: { onSubmit: (key: string) => void }) {
  const [value, setValue] = useState("");

  return (
    <form
      className="admin__auth"
      onSubmit={(e) => {
        e.preventDefault();
        if (value) onSubmit(value);
      }}
    >
      <h3 className="admin__h3">Admin key required</h3>
      <p className="admin__note">
        This backend runs with <code>BACKEND_ADMIN_KEY</code> set. Enter it to
        open the dashboard — it is kept for this browser tab only.
      </p>
      <div className="admin__auth-row">
        <input
          className="admin__auth-input"
          type="password"
          autoFocus
          aria-label="Admin key"
          placeholder="Admin key"
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        <button className="btn" type="submit" disabled={!value}>
          Unlock
        </button>
      </div>
    </form>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      role="tab"
      aria-selected={active}
      className={`admin__tab${active ? " admin__tab--on" : ""}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

/** Turn any thrown error into a readable line. */
function describe(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 503) return "The backend or database is not running.";
    return `Error ${error.status}: ${error.message}`;
  }
  return "Could not reach the backend.";
}

/** A rejected admin key, which the dialog answers with a prompt rather than text. */
function isDenied(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

/** What every tab needs: a way to say "the key was refused". */
type TabProps = { onDenied: () => void };

// --- Database tab ------------------------------------------------------------

function DataTab({ onDenied }: TabProps) {
  const [bookings, setBookings] = useState<BookingRow[] | null>(null);
  const [tickets, setTickets] = useState<TicketRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    Promise.all([listBookings(), listTickets()])
      .then(([b, t]) => {
        if (!live) return;
        setBookings(b);
        setTickets(t);
      })
      .catch((e: unknown) => {
        if (!live) return;
        if (isDenied(e)) onDenied();
        else setError(describe(e));
      });
    return () => {
      live = false;
    };
  }, [onDenied]);

  if (error) return <p className="admin__note">{error}</p>;
  if (!bookings || !tickets) return <p className="admin__note">Loading…</p>;

  return (
    <div className="admin__stack">
      <section>
        <h3 className="admin__h3">Bookings ({bookings.length})</h3>
        <div className="admin__scroll">
          <table className="admin__table">
            <thead>
              <tr>
                <th>Ref</th>
                <th>Passenger</th>
                <th>Route</th>
                <th>Date</th>
                <th>Flight</th>
                <th>Fare</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {bookings.map((b) => (
                <tr key={b.booking_reference}>
                  <td className="mono">{b.booking_reference}</td>
                  <td>{b.passenger_name}</td>
                  <td>
                    {b.origin} → {b.destination}
                  </td>
                  <td className="mono">{b.travel_date}</td>
                  <td className="mono">{b.flight_number}</td>
                  <td>{b.fare_type}</td>
                  <td>
                    <span className="pill">{b.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h3 className="admin__h3">Support tickets ({tickets.length})</h3>
        {tickets.length === 0 ? (
          <p className="admin__note">No tickets yet.</p>
        ) : (
          <div className="admin__scroll">
            <table className="admin__table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Booking</th>
                  <th>Category</th>
                  <th>Status</th>
                  <th>Summary</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {tickets.map((t) => (
                  <tr key={t.id}>
                    <td className="mono">{t.id}</td>
                    <td className="mono">{t.booking_reference}</td>
                    <td>{t.category}</td>
                    <td>
                      <span className="pill">{t.status}</span>
                    </td>
                    <td>{t.summary}</td>
                    <td className="mono">{t.created_at.slice(0, 16).replace("T", " ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

// --- System-prompt eval tab --------------------------------------------------

function PromptEvalTab({ onDenied }: TabProps) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<EvalRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cases, setCases] = useState<EvalCaseInfo[]>([]);
  const [selected, setSelected] = useState(""); // "" = all cases

  useEffect(() => {
    let live = true;
    listEvalCases()
      .then((found) => live && setCases(found))
      .catch((e: unknown) => live && isDenied(e) && onDenied());
    return () => {
      live = false;
    };
  }, [onDenied]);

  // Distinct categories with how many cases each has.
  const categories = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of cases) counts.set(c.category, (counts.get(c.category) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [cases]);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(await runSystemPromptEval(selected || undefined));
    } catch (e) {
      if (isDenied(e)) onDenied();
      else setError(describe(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="admin__stack">
      <div className="eval-intro">
        <h3 className="admin__h3">System-prompt evaluation</h3>
        <p className="admin__note">
          Grades the assistant's answers against a rubric — does it refuse what
          it should, stay grounded, and never invent a policy? One model call per
          case, so running all of them takes a minute or two.
        </p>
        <div className="eval-run">
          <label className="eval-pick">
            <span>Run</span>
            <select
              className="model__select"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              disabled={running}
            >
              <option value="">All cases ({cases.length})</option>
              {categories.length > 0 && (
                <optgroup label="A category">
                  {categories.map(([name, n]) => (
                    <option key={name} value={name}>
                      {name} ({n})
                    </option>
                  ))}
                </optgroup>
              )}
              {cases.length > 0 && (
                <optgroup label="A single case">
                  {cases.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.id} · {c.category}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </label>
          <button className="btn" onClick={run} disabled={running}>
            {running ? "Running…" : "Run eval"}
          </button>
        </div>
      </div>

      {error && <p className="admin__note admin__note--bad">{error}</p>}

      {result && (
        <section>
          <div className="eval-summary">
            <span className="eval-score">{result.overall ?? 0}/10</span>
            <span className="admin__note">
              {result.passed} of {result.total} at or above {result.pass_mark}/10
              {result.model && ` · judged by ${result.model}`}
            </span>
          </div>
          <div className="admin__scroll">
            <table className="admin__table">
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Category</th>
                  <th>Score</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row) => (
                  <tr
                    key={row.id}
                    className={
                      row.score != null && row.score < result.pass_mark
                        ? "row--fail"
                        : undefined
                    }
                  >
                    <td className="mono">{row.id}</td>
                    <td>{row.category}</td>
                    <td className="mono">
                      {row.score == null ? "—" : `${row.score}/10`}
                    </td>
                    <td>{row.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

// --- RAG eval tab ------------------------------------------------------------

const RAG_METRICS: { key: keyof RagMetricAverages; label: string }[] = [
  { key: "faithfulness", label: "Faithfulness" },
  { key: "answer_relevancy", label: "Answer relevancy" },
  { key: "context_precision", label: "Context precision" },
  { key: "context_recall", label: "Context recall" },
];

/** A metric as two decimals, or an em dash when RAGAS could not score it. */
function metric(value: number | null | undefined): string {
  return value == null ? "—" : value.toFixed(2);
}

function RagEvalTab({ onDenied }: TabProps) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RagReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cases, setCases] = useState<EvalCaseInfo[]>([]);
  const [selected, setSelected] = useState(""); // "" = all cases

  useEffect(() => {
    let live = true;
    listRagCases()
      .then((found) => live && setCases(found))
      .catch((e: unknown) => live && isDenied(e) && onDenied());
    return () => {
      live = false;
    };
  }, [onDenied]);

  const categories = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of cases) counts.set(c.category, (counts.get(c.category) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [cases]);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(await runRagEval(selected || undefined));
    } catch (e) {
      if (isDenied(e)) onDenied();
      else setError(describe(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="admin__stack">
      <div className="eval-intro">
        <h3 className="admin__h3">RAG evaluation</h3>
        <p className="admin__note">
          Scores retrieval with RAGAS — is the answer grounded in the retrieved
          context (faithfulness), does it address the question (answer
          relevancy), and did the search find relevant, sufficient chunks
          (context precision and recall)? Each metric runs 0 to 1. Several model
          calls per case, so a full run takes a few minutes.
        </p>
        <div className="eval-run">
          <label className="eval-pick">
            <span>Run</span>
            <select
              className="model__select"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              disabled={running}
            >
              <option value="">All cases ({cases.length})</option>
              {categories.length > 0 && (
                <optgroup label="A category">
                  {categories.map(([name, n]) => (
                    <option key={name} value={name}>
                      {name} ({n})
                    </option>
                  ))}
                </optgroup>
              )}
              {cases.length > 0 && (
                <optgroup label="A single case">
                  {cases.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.id} · {c.category}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </label>
          <button className="btn" onClick={run} disabled={running}>
            {running ? "Running…" : "Run eval"}
          </button>
        </div>
      </div>

      {error && <p className="admin__note admin__note--bad">{error}</p>}

      {result && (
        <section className="admin__stack">
          <div className="eval-metrics">
            {RAG_METRICS.map((m) => (
              <div className="eval-metric" key={m.key}>
                <span className="eval-metric__value">
                  {metric(result.overall[m.key])}
                </span>
                <span className="eval-metric__label">{m.label}</span>
              </div>
            ))}
          </div>
          <p className="admin__note">
            {result.results.length} cases
            {result.model && ` · scored by ${result.model}`}
          </p>

          <div className="admin__scroll">
            <table className="admin__table">
              <thead>
                <tr>
                  <th>By category</th>
                  <th>Cases</th>
                  {RAG_METRICS.map((m) => (
                    <th key={m.key}>{m.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.by_category.map((c) => (
                  <tr key={c.category}>
                    <td>{c.category}</td>
                    <td className="mono">{c.count}</td>
                    {RAG_METRICS.map((m) => (
                      <td className="mono" key={m.key}>
                        {metric(c.averages[m.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="admin__scroll">
            <table className="admin__table">
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Category</th>
                  {RAG_METRICS.map((m) => (
                    <th key={m.key}>{m.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.results.map((row: RagCaseResult) => (
                  <tr key={row.id}>
                    <td className="mono">{row.id}</td>
                    <td>{row.category}</td>
                    {RAG_METRICS.map((m) => (
                      <td className="mono" key={m.key}>
                        {metric(row[m.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
