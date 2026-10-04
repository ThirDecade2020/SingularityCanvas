import { useEffect, useRef, useState } from 'react'

type Explanation = {
  id: number
  answer: string
  model: string
  session_id: string
  session_revision: number
  run_id: number
  elapsed_ms: number
  label: string
}

type Props = {
  sessionId: string
  revision: number
  runId: number | null
  disabled: boolean
}

export default function LocalGuide({
  sessionId, revision, runId, disabled,
}: Props) {
  const [question, setQuestion] = useState(
    'Why can these same locked clues produce different completions?',
  )
  const [answer, setAnswer] = useState<Explanation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const inFlight = useRef(false)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  async function explain() {
    if (disabled || inFlight.current || runId === null || !question.trim()) return
    inFlight.current = true
    setBusy(true)
    setError('')
    const requestedRevision = revision
    const requestedRun = runId
    try {
      const response = await fetch(`/api/sessions/${sessionId}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          expected_revision: requestedRevision,
          question: question.trim(),
        }),
      })
      if (!response.ok) {
        const message = response.status === 409
          ? 'The saved session changed. Reload it before requesting another explanation.'
          : response.status === 429
            ? 'The local guide is busy. Try again shortly.'
            : 'The local guide could not complete the request. Check Ollama and try again.'
        throw new Error(message)
      }
      const result: Explanation = await response.json()
      if (
        result.session_id !== sessionId ||
        result.session_revision !== requestedRevision ||
        result.run_id !== requestedRun ||
        typeof result.answer !== 'string' || !result.answer.trim()
      ) throw new Error('The explanation did not match the requested experiment.')
      if (mounted.current) setAnswer(result)
    } catch (caught) {
      if (mounted.current) {
        setError(caught instanceof Error ? caught.message : 'Explanation failed.')
      }
    } finally {
      inFlight.current = false
      if (mounted.current) setBusy(false)
    }
  }

  const stale = answer && (
    answer.session_revision !== revision || answer.run_id !== runId
  )

  return (
    <section className="panel guide-panel" aria-label="Local AI guide">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">ON DEMAND / LOCAL AI</p>
          <h2>Understand this experiment</h2>
        </div>
        <span className="badge">Ollama · local</span>
      </div>
      <p className="description">
        Ask about the saved experiment. The guide receives measured facts and
        model details; it does not change clues or generate canvas samples.
      </p>
      <form onSubmit={event => {
        event.preventDefault()
        void explain()
      }}>
        <label htmlFor="guide-question">Your question</label>
        <textarea
          id="guide-question"
          value={question}
          onChange={event => setQuestion(event.target.value)}
          maxLength={1000}
          rows={3}
          required
          disabled={busy}
        />
        <div className="actions">
          <button className="primary" type="submit"
            disabled={disabled || busy || runId === null || !question.trim()}>
            {busy ? 'Explaining locally…' : 'Explain saved experiment'}
          </button>
          <span className="hint">
            {busy ? 'Model loading may take a moment.' : 'Runs only when requested'}
          </span>
        </div>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {answer && (
        <div className="guide-answer" aria-live="polite">
          <p className="eyebrow">
            {stale ? 'EARLIER EXPERIMENT — NOT THE CURRENT STATE' : 'SAVED EXPERIMENT EXPLANATION'}
          </p>
          <p>{answer.answer}</p>
          <p className="fine">
            Run #{answer.run_id} · Revision {answer.session_revision}
            {' · '}{answer.model}
            {' · '}{(answer.elapsed_ms / 1000).toFixed(2)} s AI request
          </p>
          <p className="fine">{answer.label}</p>
        </div>
      )}
    </section>
  )
}
