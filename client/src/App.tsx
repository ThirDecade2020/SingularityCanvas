import AccountHistory from './AccountHistory'
import BrandCard from './BrandCard'
import TrainingGallery from './TrainingGallery'
import ProbabilityChanges from './ProbabilityChanges'
import ProbabilityMap from './ProbabilityMap'
import LocalGuide from './LocalGuide'
import { useEffect, useState } from 'react'
import { blank, useCanvasSession } from './useCanvasSession'
import './App.css'

const color = (value: number) =>
  value === 1 ? 'green' : value === 0 ? 'purple' : 'unknown'

function Pattern({ bits, label }: { bits: number[]; label: string }) {
  return (
    <div className="pixel-grid pattern" role="img" aria-label={label}>
      {bits.map((value, index) => (
        <span key={index} className={`pixel ${color(value)}`} />
      ))}
    </div>
  )
}

export default function App() {
  const {
    cells, session, batch, engine, busy, error, blocked, saveStatus,
    openSession, changeCells, newBatch,
  } = useCanvasSession()
  const [username, setUsername] = useState('')
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [focused, setFocused] = useState(0)

  useEffect(() => {
    document.title = 'Singularity Canvas'
  }, [])

  useEffect(() => {
    setFrame(0)
  }, [batch])

  useEffect(() => {
    if (!playing || !batch) return
    const timer = window.setInterval(
      () => setFrame(current => (current + 1) % batch.samples.length),
      700,
    )
    return () => window.clearInterval(timer)
  }, [batch, playing])

  const locked = cells.filter(value => value !== -1).length
  const green = cells.filter(value => value === 1).length

  return (
    <main>
      <header className="topbar">
        <div className="brand"><span className="brand-mark">S∴</span> SINGULARITY MACHINES</div>
        <span className={`status ${error ? 'failure' : ''}`}>
          <span className="status-dot" />
          {error ? 'Session needs attention' : busy ? 'Saving / computing locally' : session ? 'Local session loaded' : 'Choose a username to begin'}
        </span>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">PROBABILISTIC COMPUTING / EXPERIMENT 001</p>
          <h1>Singularity <span>Canvas</span></h1>
          <p className="lead">A few fixed bits. A changing field of possibilities.</p>
        </div>
        <BrandCard />
      </section>

      <div className="notice">
        Actual Torx samples from a learned model. This preview uses 16 training symbols;
        full-catalogue quality and physical hardware integration remain future work.
      </div>


      <section className="panel session-panel" aria-label="Shared session">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">YOUR LOCAL WORKSPACE</p>
            <h2>{session ? `@${session.username}` : 'Choose a shared username'}</h2>
          </div>
          {session && <span className="badge">Revision {session.revision}</span>}
        </div>
        <p className="description" id="username-help">
          No login or password. Anyone using the same username on this backend
          can resume and contribute to its sessions.
          Use 1–40 letters, numbers, underscores or hyphens.
        </p>
        <form className="session-form" onSubmit={event => {
          event.preventDefault()
          void openSession(username.trim())
        }}>
          <label htmlFor="shared-username">Shared username</label>
          <input
            id="shared-username"
            value={username}
            onChange={event => setUsername(event.target.value)}
            disabled={busy}
            required
            maxLength={40}
            pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,39}"
            autoCapitalize="none"
            autoComplete="off"
            spellCheck={false}
            aria-describedby="username-help"
            placeholder="singularitymachines"
          />
          <div className="actions">
            <button type="submit" className="primary"
              disabled={busy || !username.trim()}>
              Open / resume
            </button>
            {session && <>
              <button type="button" disabled={busy}
                onClick={() => void openSession(session.username)}>
                Reload saved session
              </button>
              <button type="button" disabled={busy || blocked}
                onClick={() => void openSession(session.username, true)}>
                New experiment
              </button>
            </>}
          </div>
        </form>
        <p className="fine" role="status">{saveStatus}</p>
        {session && <p className="fine">
          Session <code>{session.id}</code>
          {batch && <> · Saved run #{batch.run_id}</>}
        </p>}
        {blocked && <p className="fine">
          Reloading replaces local edits with the saved session.
        </p>}
      </section>

      <details className="explore-guide">
        <summary>How to explore · a quick start</summary>
        <p className="explore-start">
          Open or resume a shared username above, then explore freely.
        </p>
        <ol>
          <li>
            <strong>Set your clues</strong>
            <p>Click any cell: green → purple → unlocked.
            Try five clues, or choose any number.</p>
          </li>
          <li>
            <strong>Watch probabilities change</strong>
            <p>The probability map updates after sampling completes.
            Inspect a cell to see its chance of being green.</p>
          </li>
          <li>
            <strong>Explore Torx samples</strong>
            <p>Each displayed completion is an actual model sample.
            Draw another batch to explore the same clues again.</p>
          </li>
          <li>
            <strong>Compare references</strong>
            <p>Scroll to the training gallery to see which symbols match
            every clue. Click a reference grid to inspect it.</p>
          </li>
        </ol>
        <p className="explore-note">
          Several references may fit the same clues. Compatibility does not
          identify your intended symbol, and samples may differ from every reference.
        </p>
      </details>

      <section className="workspace">
        <article className="panel">
          <div className="panel-heading">
            <div><p className="eyebrow">01 / CONSTRAIN</p><h2>Your clues</h2></div>
            <span className="badge">{locked} locked</span>
          </div>
          <p className="description">Click freely: unknown → green → purple → unknown.</p>
          <div className="pixel-grid editor" role="group" aria-label="Editable 16 by 16 clue grid">
            {cells.map((value, index) => (
              <button
                key={index}
                id={`cell-${index}`}
                disabled={!session || blocked}
                className={`pixel ${color(value)}`}
                tabIndex={focused === index ? 0 : -1}
                aria-label={`Row ${Math.floor(index / 16) + 1}, column ${index % 16 + 1}: ${
                  value === -1 ? 'unknown' : value === 1 ? 'green, locked to one' : 'purple, locked to zero'
                }`}
                onFocus={() => setFocused(index)}
                onClick={() => {
                  const next = [...cells]
                  next[index] = value === -1 ? 1 : value === 1 ? 0 : -1
                  changeCells(next)
                }}
                onKeyDown={event => {
                  const offsets: Record<string, number> = {
                    ArrowLeft: -1, ArrowRight: 1, ArrowUp: -16, ArrowDown: 16,
                  }
                  const offset = offsets[event.key]
                  if (offset === undefined) return
                  event.preventDefault()
                  const next = Math.max(0, Math.min(255, index + offset))
                  setFocused(next)
                  document.getElementById(`cell-${next}`)?.focus()
                }}
              />
            ))}
          </div>
          <div className="legend">
            <span><i className="green" />1 · {green} green</span>
            <span><i className="purple" />0 · {locked - green} purple</span>
            <span><i className="unknown" />{256 - locked} free</span>
          </div>
          <div className="actions">
            <button onClick={() => changeCells(blank())} disabled={!session || blocked || locked === 0}>Clear clues</button>
            <span className="hint">Updates automatically</span>
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div><p className="eyebrow">02 / SAMPLE</p><h2>Possible completion</h2></div>
            <span className="badge">{batch ? `${frame + 1} / ${batch.samples.length}` : '—'}</span>
          </div>
          <p className="description">Every displayed grid is an unaltered sampled state.</p>
          {batch ? (
            <Pattern bits={batch.samples[frame]} label={`Actual sampled completion ${frame + 1}`} />
          ) : (
            <div className="empty-grid" role="status">
              <span>{error ? 'Session needs attention' : !session ? 'Open a shared session' : 'Preparing possibilities…'}</span>
              <small>{error ? 'See the message below.' : !session ? 'Enter a username above to begin or resume.' : 'Applying your current clues'}</small>
            </div>
          )}
          <div className="playback-note">
            {batch ? 'Replaying this batch of 8 draws; playback creates no new samples.' : 'Waiting for the current batch.'}
          </div>
          <div className="actions">
            <button disabled={!batch} onClick={() => setPlaying(value => !value)}>
              {playing ? 'Pause playback' : 'Play batch'}
            </button>
            <button className="primary" disabled={busy || !session || blocked} onClick={newBatch}>
              {'Draw new batch'}
            </button>
          </div>
        </article>

        <ProbabilityMap
          probabilities={batch?.green_probabilities}
          cells={cells}
          hasBatch={Boolean(batch)}
        />

        <aside className="panel telemetry">
          <p className="eyebrow">04 / OBSERVE</p><h2>Inside the engine</h2>
          <dl>
            <div><dt>Execution</dt><dd>{engine?.devices.join(', ') ?? 'Connecting…'}</dd></div>
            <div><dt>Torx version</dt><dd>{engine?.torx_version ?? '—'}</dd></div>
            <div><dt>Training symbols</dt><dd>{engine?.training_symbols ?? '—'}</dd></div>
            <div><dt>Active hidden bits</dt><dd>{engine?.active_hidden_units ?? '—'}</dd></div>
            <div><dt>Enumerated hidden states</dt><dd>{engine?.enumerated_hidden_states.toLocaleString() ?? '—'}</dd></div>
            <div><dt>Temperature</dt><dd>{engine?.temperature ?? '—'} <small>fixed</small></dd></div>
            <div><dt>Samples + probabilities</dt><dd>{batch ? `${batch.compute_ms.toFixed(2)} ms` : '—'}</dd></div>
            <div><dt>Random seed</dt><dd>{batch?.seed ?? '—'}</dd></div>
          </dl>
          <p className="fine">{batch?.timing_scope ?? 'Computation timing will appear after a successful request.'}</p>
          <details>
            <summary>How these samples are made</summary>
            <p>The backend enumerates hidden configurations, calculates their probabilities given your
              clues, then uses Torx to draw a configuration and sample its pixels.
              Locked cells stay fixed.</p>
            <p>These are independent draws, not a recorded physical p-bit trajectory.
              Sparse clues can admit several completions.</p>
            <code>{engine?.model ?? 'Model details pending'}</code>
          </details>
        </aside>
      </section>

      {error && <div className="error" role="alert">{error} Use the session controls above to reload after resolving the issue.</div>}

      {session && (
        <ProbabilityChanges
          key={"changes:" + session.id}
          batch={batch}
          cells={cells}
        />
      )}

      <section className="panel gallery">
        <div className="panel-heading">
          <div><p className="eyebrow">THE CURRENT BATCH</p><h2>Eight possibilities, no selection</h2></div>
          <span className="hint">Raw samples · no cleanup</span>
        </div>
        {batch ? (
          <div className="sample-list">
            {batch.samples.map((sample, index) => (
              <button
                key={index}
                className={`sample-card ${frame === index ? 'selected' : ''}`}
                aria-pressed={frame === index}
                onClick={() => { setFrame(index); setPlaying(false) }}
              >
                <Pattern bits={sample} label={`Completion ${index + 1}`} />
                <span>DRAW {String(index + 1).padStart(2, '0')}</span>
              </button>
            ))}
          </div>
        ) : <p className="description">Samples for your current clues will appear here.</p>}
      </section>

      {session && (
        <LocalGuide
          key={"guide:" + session.id}
          sessionId={session.id}
          revision={session.revision}
          runId={batch?.run_id ?? null}
          disabled={busy || blocked || !batch}
        />
      )}

      <TrainingGallery batchModelHash={batch?.model_sha256} cells={cells} />
      {session && (
        <AccountHistory key={'history:' + session.username} username={session.username} />
      )}

      <section className="roadmap">
        <div><p className="eyebrow">WORKING NOW</p><h3>Conditional sampling</h3>
          <p>Freely placed clues, local Torx draws, reproducible seeds, shared usernames, and saved sessions.</p></div>
        <div><p className="eyebrow">KNOWN LIMITS</p><h3>A diagnostic, openly labeled</h3>
          <p>No speed or energy advantage established. Alarm and stopwatch remain weak.
            These training-set results do not establish generalization.</p></div>
        <div><p className="eyebrow">NEXT ITERATIONS</p><h3>From experiment to platform</h3>
          <p>Expanded experiment analysis,
            broader symbol learning, and a compatible hardware adapter.</p></div>
      </section>
      <footer>MAXIMUM CONTROL. MINIMUM ABSTRACTION.<span>Singularity Machines · Local probability canvas</span></footer>
    </main>
  )
}
