import { useState } from 'react'
import type { Cell } from './useCanvasSession'

function shade(probability: number) {
  const purple = [135, 59, 208]
  const silver = [188, 194, 205]
  const green = [109, 237, 121]
  const start = probability <= 0.5 ? purple : silver
  const end = probability <= 0.5 ? silver : green
  const amount = probability <= 0.5 ? probability * 2 : (probability - 0.5) * 2
  return `rgb(${start.map((value, index) =>
    Math.round(value + (end[index] - value) * amount)
  ).join(',')})`
}

export default function ProbabilityMap({
  probabilities, cells, hasBatch,
}: {
  probabilities?: number[]
  cells: Cell[]
  hasBatch: boolean
}) {
  const [selected, setSelected] = useState(0)
  return (
    <article className="panel probability-panel">
      <div className="panel-heading">
        <div><p className="eyebrow">03 / PROBABILITY</p><h2>Chance of green</h2></div>
        <span className="badge">Model marginals</span>
      </div>
      <p className="description">
        Each cell shows its probability of being green given all current clues.
      </p>
      {probabilities ? <>
        <div className="pixel-grid" role="group" aria-label="Pixel probabilities">
          {probabilities.map((probability, index) => {
            const label = `Row ${Math.floor(index / 16) + 1}, column ${index % 16 + 1}: ${
              (probability * 100).toFixed(1)
            }% green${cells[index] !== -1 ? ', locked' : ''}`
            return (
              <button
                key={index}
                type="button"
                className={`pixel probability-cell ${cells[index] !== -1 ? 'locked-probability' : ''}`}
                style={{ backgroundColor: shade(probability) }}
                tabIndex={selected === index ? 0 : -1}
                aria-label={label}
                title={label}
                onMouseEnter={() => setSelected(index)}
                onFocus={() => setSelected(index)}
                onClick={() => setSelected(index)}
                onKeyDown={event => {
                  const offsets: Record<string, number> = {
                    ArrowLeft: -1, ArrowRight: 1, ArrowUp: -16, ArrowDown: 16,
                  }
                  const offset = offsets[event.key]
                  if (offset === undefined) return
                  event.preventDefault()
                  const next = Math.max(0, Math.min(255, index + offset))
                  const buttons = event.currentTarget.parentElement?.querySelectorAll('button')
                  buttons?.[next]?.focus()
                }}
              />
            )
          })}
        </div>
        <div className="probability-scale" aria-hidden="true" />
        <div className="legend"><span>0% green</span><span>50%</span><span>100% green</span></div>
        <p className="fine">
          Row {Math.floor(selected / 16) + 1}, column {selected % 16 + 1}
          {' · '}{(probabilities[selected] * 100).toFixed(1)}% green
          {cells[selected] !== -1 ? ' · Locked clue' : ' · Unlocked'}
        </p>
      </> : (
        <div className="empty-grid">
          <span>{hasBatch ? 'This saved batch predates the heatmap' : 'Probabilities appear with a saved batch'}</span>
          <small>{hasBatch ? 'Choose “Draw new batch” to calculate them.' : 'Open a session and place your clues.'}</small>
        </div>
      )}
      <p className="fine">
        Calculated by exact hidden-state enumeration, up to numerical precision.
        Not estimated from the eight displayed samples.
        Silver means 50% probability, not a third bit state.
        These values describe this model, not confidence in a symbol’s identity.
      </p>
    </article>
  )
}
