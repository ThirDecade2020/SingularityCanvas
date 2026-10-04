import { useEffect, useRef, useState } from 'react'

export type Cell = -1 | 0 | 1
export type Engine = {
  backend: string
  model: string
  model_scope: string
  training_symbols: number
  active_hidden_units: number
  enumerated_hidden_states: number
  temperature: number
  devices: string[]
  torx_version: string
  jax_version: string
  limitations: string[]
}
export type Batch = {
  green_probabilities?: number[]
  probability_method?: string
  probability_scope?: string
  samples: number[][]
  seed: number
  revision: number
  compute_ms: number
  timing_scope: string
  engine: Engine
  session_id: string
  session_revision: number
  model_sha256: string
  run_id: number
}
type Clue = { index: number; value: 0 | 1 }
type Session = {
  id: string
  username: string
  revision: number
  clues: Clue[]
  latest_batch: Batch | null
}
export const blank = () => Array<Cell>(256).fill(-1)

async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    if (response.status === 409) {
      throw new Error('This shared session changed elsewhere. Reload saved session to continue.')
    }
    throw new Error(`Request failed (${response.status}). Check the API, then reload the saved session.`)
  }
  return response.json() as Promise<T>
}

function verifyBatch(batch: Batch, clues: Clue[]) {
  const probabilities = batch.green_probabilities
  if (probabilities !== undefined && (
    !Array.isArray(probabilities) ||
    probabilities.length !== 256 ||
    probabilities.some(value => !Number.isFinite(value) || value < 0 || value > 1) ||
    clues.some(clue => probabilities[clue.index] !== clue.value)
  )) throw new Error('Invalid pixel probabilities. Reload the saved session.')
  if (
    !Array.isArray(batch.samples) || batch.samples.length !== 8 ||
    batch.samples.some(sample =>
      sample.length !== 256 ||
      sample.some(value => value !== 0 && value !== 1) ||
      clues.some(clue => sample[clue.index] !== clue.value)
    )
  ) throw new Error('Invalid sample batch. Reload the saved session.')
}

export function useCanvasSession() {
  const [cells, setCells] = useState<Cell[]>(blank)
  const [session, setSession] = useState<Session | null>(null)
  const [batch, setBatch] = useState<Batch | null>(null)
  const [engine, setEngine] = useState<Engine | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [blocked, setBlocked] = useState(false)
  const [requestNumber, setRequestNumber] = useState(0)
  const [saveStatus, setSaveStatus] = useState('Choose a shared username')

  const active = useRef<Session | null>(null)
  const serial = useRef<Promise<void>>(Promise.resolve())
  const version = useRef(0)
  const halted = useRef(false)
  const pending = useRef(false)
  const opening = useRef(false)

  function fail(caught: unknown) {
    halted.current = true
    setBlocked(true)
    setError(caught instanceof Error ? caught.message : 'Request failed.')
    setSaveStatus('Reload required; local edits may not be saved')
  }

  async function openSession(username: string, fresh = false) {
    if (busy || opening.current) return
    opening.current = true
    const current = ++version.current
    pending.current = false
    setBusy(true)
    setError('')
    setSaveStatus('Loading saved session…')
    try {
      // Finish any queued work before changing the active account.
      await serial.current
      const loaded = await api<Session>('/api/sessions/open', 'POST', {
        username, new: fresh,
      })
      if (loaded.latest_batch) verifyBatch(loaded.latest_batch, loaded.clues)
      const restored = blank()
      for (const clue of loaded.clues) restored[clue.index] = clue.value
      active.current = loaded
      halted.current = false
      setBlocked(false)
      setSession(loaded)
      setCells(restored)
      setBatch(loaded.latest_batch)
      setEngine(loaded.latest_batch?.engine ?? null)
      setSaveStatus(loaded.latest_batch ? 'Saved session and batch restored' : 'Session loaded')
      // A restored batch is displayed exactly as saved, without a new draw.
      if (!loaded.latest_batch) {
        pending.current = true
        setSaveStatus('Preparing first saved batch…')
        setRequestNumber(value => value + 1)
      }
    } catch (caught) {
      fail(caught)
    } finally {
      opening.current = false
      if (current === version.current && !pending.current) setBusy(false)
    }
  }

  function changeCells(next: Cell[]) {
    if (!active.current || halted.current || opening.current) return
    version.current += 1
    pending.current = true
    setCells(next)
    setBatch(null)
    setBusy(true)
    setSaveStatus('Saving clues…')
    setRequestNumber(value => value + 1)
  }

  function newBatch() {
    if (!active.current || halted.current || busy || opening.current) return
    version.current += 1
    pending.current = true
    setBatch(null)
    setBusy(true)
    setSaveStatus('Preparing another saved batch…')
    setRequestNumber(value => value + 1)
  }

  useEffect(() => {
    if (!pending.current || !active.current || halted.current) return
    const current = version.current
    const sessionId = active.current.id
    const clues: Clue[] = cells.flatMap((value, index) =>
      value === -1 ? [] : [{ index, value }]
    )
    const timer = window.setTimeout(() => {
      serial.current = serial.current.then(async () => {
        if (
          current !== version.current || halted.current ||
          active.current?.id !== sessionId
        ) return
        pending.current = false
        try {
          const saved = await api<Session>(
            `/api/sessions/${sessionId}/clues`, 'PUT',
            { expected_revision: active.current.revision, clues },
          )
          // Keep acknowledged server revisions even if newer local edits exist.
          active.current = saved
          setSession(saved)
          if (current !== version.current) return
          setSaveStatus('Clues saved; sampling…')
          const result = await api<Batch>(
            `/api/sessions/${sessionId}/sample`, 'POST',
            {
              expected_revision: saved.revision,
              count: 8,
              seed: crypto.getRandomValues(new Uint32Array(1))[0],
              revision: current,
            },
          )
          verifyBatch(result, clues)
          if (result.session_id !== sessionId || result.revision !== current) {
            throw new Error('Unexpected session response. Reload saved session.')
          }
          const updated = {
            ...saved,
            revision: result.session_revision,
            latest_batch: result,
          }
          active.current = updated
          setSession(updated)
          if (current !== version.current) return
          setBatch(result)
          setEngine(result.engine)
          setSaveStatus('Clues and batch saved locally')
        } catch (caught) {
          fail(caught)
          pending.current = false
        } finally {
          if (current === version.current || halted.current) setBusy(false)
        }
      })
    }, 250)
    return () => window.clearTimeout(timer)
  }, [cells, requestNumber])

  return {
    cells, session, batch, engine, busy, error, blocked, saveStatus,
    openSession, changeCells, newBatch,
  }
}
