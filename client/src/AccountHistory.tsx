import { useEffect, useState } from 'react'

type RecordData = Record<string, unknown>
type Session = RecordData & {
  id: string
  revision: number
  created_at: string
  updated_at: string
  run_count?: number
  explanation_count?: number
  decoded: { clues: { index: number; value: number }[] }
}
type Run = RecordData & {
  id: number
  session_id: string
  session_revision: number
  seed: number
  model_name: string
  model_sha256: string
  backend: string
  created_at: string
  explanation_count?: number
  decoded?: { request: RecordData; result: RecordData }
}
type Explanation = RecordData & {
  id: number
  question: string
  answer: string
  model: string
  elapsed_ms: number
  created_at: string
}
type AccountPage = {
  user: RecordData
  counts: { sessions: number; sampling_runs: number; explanations: number }
  sessions: Session[]
  has_more: boolean
}
type SessionPage = { session: Session; runs: Run[]; total: number; has_more: boolean }
type RunPage = {
  run: Run
  explanations: Explanation[]
  total_explanations: number
  has_more: boolean
}

function useHistory<T>(url: string) {
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<{
    key: string; data?: T; error?: string
  }>({ key: '' })
  const key = `${url}|${attempt}`
  useEffect(() => {
    const controller = new AbortController()
    async function load() {
      try {
        const response = await fetch(url, { signal: controller.signal })
        if (!response.ok) throw new Error(`History request failed (${response.status}).`)
        const data: T = await response.json()
        if (!controller.signal.aborted) setState({ key, data })
      } catch (error) {
        if (!controller.signal.aborted) {
          setState({
            key,
            error: error instanceof Error ? error.message : 'History unavailable.',
          })
        }
      }
    }
    void load()
    return () => controller.abort()
  }, [url, key])
  return {
    data: state.key === key ? state.data : undefined,
    error: state.key === key ? state.error : undefined,
    retry: () => setAttempt(value => value + 1),
  }
}

function Waiting({ error, retry }: { error?: string; retry: () => void }) {
  return error ? (
    <div className="actions">
      <p role="alert">{error}</p>
      <button type="button" onClick={retry}>Retry</button>
    </div>
  ) : <p className="fine" role="status">Loading saved records…</p>
}

function Raw({ value, label = 'Complete stored record' }: {
  value: unknown; label?: string
}) {
  return (
    <details className="history-raw">
      <summary>{label}</summary>
      <pre tabIndex={0}>{JSON.stringify(value, null, 2)}</pre>
    </details>
  )
}

function Pages({ offset, more, change }: {
  offset: number; more: boolean; change: (offset: number) => void
}) {
  if (offset === 0 && !more) return null
  return (
    <div className="actions history-pages">
      <button type="button" disabled={offset === 0}
        onClick={() => change(Math.max(0, offset - 20))}>Previous</button>
      <span>Page {Math.floor(offset / 20) + 1}</span>
      <button type="button" disabled={!more}
        onClick={() => change(offset + 20)}>Next</button>
    </div>
  )
}

function Grid({ values, label }: { values: number[]; label: string }) {
  return (
    <figure className="history-grid">
      <div className="pixel-grid" role="img" aria-label={label}>
        {values.map((value, index) => (
          <span key={index}
            className={`pixel ${value === 1 ? 'green' : value === 0 ? 'purple' : 'unknown'}`} />
        ))}
      </div>
      <figcaption>{label}</figcaption>
    </figure>
  )
}

function clueGrid(value: unknown) {
  const values = Array<number>(256).fill(-1)
  if (Array.isArray(value)) {
    for (const clue of value) {
      if (clue && Number.isInteger(clue.index) && clue.index >= 0 &&
        clue.index < 256 && (clue.value === 0 || clue.value === 1)) {
        values[clue.index] = clue.value
      }
    }
  }
  return values
}

function isBits(value: unknown): value is number[] {
  return Array.isArray(value) && value.length === 256 &&
    value.every(bit => bit === 0 || bit === 1)
}

function RunHistory({ base, id }: { base: string; id: number }) {
  const [offset, setOffset] = useState(0)
  const { data, error, retry } = useHistory<RunPage>(
    `${base}/runs/${id}/history?limit=20&offset=${offset}`
  )
  if (!data) return <Waiting error={error} retry={retry} />
  const run = data.run
  const result = run.decoded?.result ?? {}
  const request = run.decoded?.request ?? {}
  const samples = Array.isArray(result.samples) ? result.samples.filter(isBits) : []
  const probabilities = result.green_probabilities
  const hasProbabilities = Array.isArray(probabilities) &&
    probabilities.length === 256 &&
    probabilities.every(value => typeof value === 'number' &&
      Number.isFinite(value) && value >= 0 && value <= 1)

  return (
    <div className="history-run-body">
      <dl className="history-metadata">
        <div><dt>Seed</dt><dd>{run.seed}</dd></div>
        <div><dt>Session revision</dt><dd>{run.session_revision}</dd></div>
        <div><dt>Backend</dt><dd>{run.backend}</dd></div>
        <div><dt>Model</dt><dd>{run.model_name}</dd></div>
        <div><dt>Model fingerprint</dt><dd>{run.model_sha256}</dd></div>
        <div><dt>Recorded computation</dt><dd>
          {typeof result.compute_ms === 'number'
            ? `${result.compute_ms.toFixed(2)} ms` : 'Not recorded'}
        </dd></div>
      </dl>
      {typeof result.timing_scope === 'string' &&
        <p className="fine">{result.timing_scope}</p>}
      <div className="history-samples">
        <Grid values={clueGrid(result.clues ?? request.clues)} label="Clues used for this run" />
        {samples.map((sample, index) =>
          <Grid key={index} values={sample} label={`Saved sample ${index + 1}`} />
        )}
      </div>
      <details>
        <summary>Saved green probabilities · all 256 cells</summary>
        {hasProbabilities ? (
          <div className="history-probabilities" tabIndex={0}>
            <table>
              <caption>Recorded probability of green, in row-major order</caption>
              <thead><tr><th>Row</th><th>Column</th><th>Probability</th></tr></thead>
              <tbody>{(probabilities as number[]).map((value, index) => (
                <tr key={index}>
                  <td>{Math.floor(index / 16) + 1}</td>
                  <td>{index % 16 + 1}</td>
                  <td>{(value * 100).toFixed(3)}%</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : <p className="fine">This run has no saved probability map.</p>}
      </details>
      <Raw value={run} label="Complete run record · request, result, and metadata" />
      <h3 className="history-explanations-title">
        Saved AI explanations ({data.total_explanations})
      </h3>
      {data.total_explanations === 0 &&
        <p className="fine">No explanation was saved for this run.</p>}
      {data.explanations.map(explanation => (
        <article key={explanation.id} className="history-explanation">
          <p className="fine">
            #{explanation.id} · {explanation.created_at} UTC
            {' · '}{explanation.model}
            {' · '}{(explanation.elapsed_ms / 1000).toFixed(2)} s
          </p>
          <h4>{explanation.question}</h4>
          <p className="history-answer">{explanation.answer}</p>
          <p className="fine">Saved AI explanation; may contain errors.</p>
          <Raw value={explanation} label="Complete explanation and input snapshot" />
        </article>
      ))}
      <Pages offset={offset} more={data.has_more} change={setOffset} />
    </div>
  )
}

function RunItem({ run, base }: { run: Run; base: string }) {
  const [open, setOpen] = useState(false)
  return (
    <details className="history-item"
      onToggle={event => setOpen(event.currentTarget.open)}>
      <summary>
        Run #{run.id} · {run.created_at} UTC
        {' · '}{run.explanation_count ?? 0} AI explanations
      </summary>
      {open && <RunHistory base={base} id={run.id} />}
    </details>
  )
}

function SessionHistory({ base, id }: { base: string; id: string }) {
  const [offset, setOffset] = useState(0)
  const { data, error, retry } = useHistory<SessionPage>(
    `${base}/sessions/${encodeURIComponent(id)}/history?limit=20&offset=${offset}`
  )
  if (!data) return <Waiting error={error} retry={retry} />
  return (
    <div className="history-session-body">
      <p className="fine">
        Session {data.session.id} · revision {data.session.revision}
      </p>
      <div className="history-session-clues">
        <Grid values={clueGrid(data.session.decoded.clues)}
          label="Current saved session clues" />
      </div>
      <Raw value={data.session} label="Complete session record" />
      <h3>Sampling runs ({data.total})</h3>
      {data.total === 0 && <p className="fine">No sampling runs saved yet.</p>}
      {data.runs.map(run => <RunItem key={run.id} run={run} base={base} />)}
      <Pages offset={offset} more={data.has_more} change={setOffset} />
    </div>
  )
}

function SessionItem({ session, base }: { session: Session; base: string }) {
  const [open, setOpen] = useState(false)
  return (
    <details className="history-item"
      onToggle={event => setOpen(event.currentTarget.open)}>
      <summary>
        Session created {session.created_at} UTC
        {' · '}{session.run_count} runs
        {' · '}{session.explanation_count} explanations
      </summary>
      {open && <SessionHistory base={base} id={session.id} />}
    </details>
  )
}

function AccountRecords({ username }: { username: string }) {
  const [offset, setOffset] = useState(0)
  const base = `/api/accounts/${encodeURIComponent(username)}`
  const { data, error, retry } = useHistory<AccountPage>(
    `${base}/history?limit=20&offset=${offset}`
  )
  if (!data) return <Waiting error={error} retry={retry} />
  return (
    <>
      <p className="history-counts">
        <strong>{data.counts.sessions}</strong> sessions ·
        {' '}<strong>{data.counts.sampling_runs}</strong> saved runs ·
        {' '}<strong>{data.counts.explanations}</strong> AI explanations
      </p>
      <Raw value={data.user} label="Username record" />
      {data.sessions.map(session =>
        <SessionItem key={session.id} session={session} base={base} />
      )}
      {data.sessions.length === 0 && <p className="fine">No sessions saved.</p>}
      <Pages offset={offset} more={data.has_more} change={setOffset} />
    </>
  )
}

export default function AccountHistory({ username }: { username: string }) {
  const [open, setOpen] = useState(false)
  const [refresh, setRefresh] = useState(0)
  return (
    <section className="panel account-history">
      <p className="eyebrow">YOUR LOCAL RECORDS</p>
      <h2>Account history</h2>
      <p className="description">
        Browse saved experiments for <strong>{username}</strong>.
        Anyone using this shared username can view the same history.
        Browsing leaves your current canvas unchanged.
      </p>
      <div className="actions">
        <button type="button" aria-expanded={open}
          onClick={() => setOpen(value => !value)}>
          {open ? 'Close history' : 'Browse account history'}
        </button>
        {open && <button type="button"
          onClick={() => setRefresh(value => value + 1)}>Refresh history</button>}
        <a className="history-download"
          href={`/api/accounts/${encodeURIComponent(username)}/export`}
          download>
          Download account history
        </a>
      </div>
      {open && <AccountRecords key={`${username}:${refresh}`} username={username} />}
      <p className="fine">
        Expand a session, then a run. Refresh to include new saves.
        Individual clue edits between runs are not retained as a complete timeline.
        Raw records include all stored fields; decoded JSON is provided for readability.
      </p>
    </section>
  )
}
