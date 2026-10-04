import ReferencePreview from './ReferencePreview'
import { useEffect, useState } from 'react'

type Gallery = {
  model: string
  model_sha256: string
  source: string
  source_version: string
  width: number
  height: number
  symbol_count: number
  symbols: { id: string; names: string[]; bits: number[] }[]
}

export default function TrainingGallery({
  batchModelHash,
  cells,
}: { batchModelHash?: string; cells: readonly number[] }) {
  const [gallery, setGallery] = useState<Gallery | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [compatibleOnly, setCompatibleOnly] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    async function load() {
      setError('')
      try {
        const response = await fetch('/api/model/symbols', {
          signal: controller.signal,
        })
        if (!response.ok) throw new Error(`Reference request failed (${response.status}).`)
        const result: Gallery = await response.json()
        if (
          result.width !== 16 || result.height !== 16 ||
          !Array.isArray(result.symbols) ||
          result.symbol_count !== result.symbols.length ||
          result.symbol_count === 0 ||
          typeof result.model_sha256 !== 'string' ||
          !/^[a-f0-9]{64}$/.test(result.model_sha256) ||
          new Set(result.symbols.map(symbol => symbol.id)).size !== result.symbol_count ||
          result.symbols.some(symbol =>
            !Array.isArray(symbol.names) || symbol.names.length === 0 ||
            symbol.names.some(name => typeof name !== 'string' || !name) ||
            !Array.isArray(symbol.bits) || symbol.bits.length !== 256 ||
            symbol.bits.some(bit => bit !== 0 && bit !== 1)
          )
        ) throw new Error('Invalid training-reference data.')
        if (!controller.signal.aborted) setGallery(result)
      } catch (caught) {
        if (!controller.signal.aborted) {
          setGallery(null)
          setError(caught instanceof Error ? caught.message : 'Could not load references.')
        }
      }
    }
    void load()
    return () => controller.abort()
  }, [attempt])

  const mismatch = gallery && batchModelHash &&
    gallery.model_sha256 !== batchModelHash

  const lockedCount = cells.filter(value => value !== -1).length
  const references = gallery?.symbols.map(symbol => ({
    ...symbol,
    conflicts: symbol.bits.reduce(
      (count, bit, index) =>
        count + (cells[index] !== -1 && cells[index] !== bit ? 1 : 0),
      0,
    ),
  })) ?? []
  const compatibleCount = references.filter(symbol => symbol.conflicts === 0).length

  return (
    <section className="panel training-gallery" aria-label="Training symbol references">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">MODEL TRANSPARENCY / TRAINING REFERENCES</p>
          <h2>Symbols this model learned</h2>
        </div>
        <span className="badge">
          {gallery ? `${gallery.symbol_count} references` : 'References'}
        </span>
      </div>
      <p className="description">
        These are the original binary training examples for the currently loaded
        model. Compare them with the raw sampled completions above.
        They are reference images, not generated results or guaranteed outputs.
      </p>
      {mismatch && (
        <p className="error" role="status">
          Your displayed batch came from a different model.
          This gallery describes the currently loaded backend model.
        </p>
      )}
      {error ? (
        <div>
          <p className="error" role="alert">{error}</p>
          <div className="actions">
            <button type="button" onClick={() => setAttempt(value => value + 1)}>
              Retry references
            </button>
          </div>
        </div>
      ) : !gallery ? (
        <p className="description" role="status">Loading local training references…</p>
      ) : <>
        <div className="compatibility-summary" role="status" aria-live="polite">
          <strong>{compatibleCount} of {gallery.symbol_count}</strong>
          {' '}references compatible with your clues
          <span>
            {lockedCount === 0
              ? 'No locked cells yet—all references are compatible.'
              : `${lockedCount} locked cells checked, including green and purple.`}
          </span>
        </div>
        {compatibleCount === 0 && (
          <p className="description">
            None of these training references matches every current clue.
            The model can still generate other bit patterns that preserve your locks.
          </p>
        )}
        <p className="fine">
          Compatibility updates immediately from your current editor clues.
          It is an exact reference check, not a model probability or prediction
          of your intended symbol. It does not select or replace Torx samples.
        </p>
        <p className="fine">
          Locked positions: white outline = matching clue; white × = conflicting clue.
          Cell colors show the original reference bits.
        </p>
        <div className="reference-filter">
          <label>
            <input
              type="checkbox"
              checked={compatibleOnly}
              onChange={event => setCompatibleOnly(event.target.checked)}
            />
            Show compatible only
          </label>
          <span>
            Showing {compatibleOnly ? compatibleCount : references.length}
            {' '}of {references.length} references
          </span>
        </div>
        {compatibleOnly && compatibleCount === 0 && (
          <p className="description">
            No compatible references to display. Turn off the filter to inspect conflicts.
          </p>
        )}
        <div className="training-symbols">
          {references.filter(symbol => !compatibleOnly || symbol.conflicts === 0).map(symbol => (
            <figure
              className={`training-symbol ${symbol.conflicts === 0 ? 'reference-compatible' : 'reference-conflicting'}`}
              key={symbol.id}>
              <ReferencePreview
                names={symbol.names}
                bits={symbol.bits}
                cells={cells}
              />
              <figcaption>
                {symbol.names.join(' / ')}
                <span className="reference-match-label">
                  {symbol.conflicts === 0
                    ? 'Compatible'
                    : `${symbol.conflicts} clue ${symbol.conflicts === 1 ? 'conflict' : 'conflicts'}`}
                </span>
              </figcaption>
            </figure>
          ))}
        </div>
        <p className="fine">
          Source: {gallery.source} {gallery.source_version}
          {' · '}16 × 16 binary rasterizations
          {' · '}<a href="/api/model/symbols/license" target="_blank" rel="noreferrer">
            Source license
          </a>
        </p>
        <details>
          <summary>Reference model identity</summary>
          <p><code>{gallery.model}</code></p>
          <p>SHA-256: <code>{gallery.model_sha256}</code></p>
        </details>
      </>}
    </section>
  )
}
