import { useEffect, useRef, useState } from 'react'
import type { Batch, Cell } from './useCanvasSession'

type Snapshot = {
  probabilities: number[]
  cells: Cell[]
  run: number
  model: string
}
type Shift = { index: number; before: number; after: number; points: number }
type Comparison = {
  from: number
  to: number
  edits: number
  eligible: number
  changed: number
  largest: Shift[]
}

export default function ProbabilityChanges({
  batch, cells,
}: { batch: Batch | null; cells: Cell[] }) {
  const previous = useRef<Snapshot | null>(null)
  const [comparison, setComparison] = useState<Comparison | null>(null)

  useEffect(() => {
    if (!batch) return
    if (!batch.green_probabilities) {
      previous.current = null
      setComparison(null)
      return
    }
    const next: Snapshot = {
      probabilities: [...batch.green_probabilities],
      cells: [...cells],
      run: batch.run_id,
      model: batch.model_sha256,
    }
    const before = previous.current
    previous.current = next
    if (!before || before.model !== next.model) {
      setComparison(null)
      return
    }

    const edits = cells.filter((value, index) => value !== before.cells[index]).length
    // Another draw with identical clues keeps the last clue-change comparison.
    if (edits === 0) return

    const shifts: Shift[] = []
    for (let index = 0; index < 256; index++) {
      if (cells[index] !== -1 || before.cells[index] !== -1) continue
      shifts.push({
        index,
        before: before.probabilities[index],
        after: next.probabilities[index],
        points: 100 * (next.probabilities[index] - before.probabilities[index]),
      })
    }
    const significant = shifts
      .filter(shift => Math.abs(shift.points) >= 1)
      .sort((a, b) => Math.abs(b.points) - Math.abs(a.points) || a.index - b.index)

    setComparison({
      from: before.run,
      to: next.run,
      edits,
      eligible: shifts.length,
      changed: significant.length,
      largest: significant.slice(0, 5),
    })
  }, [batch, cells])

  return (
    <section className="panel changes-panel" aria-label="Probability changes">
      <p className="eyebrow">CLUE INFLUENCE / WHAT CHANGED?</p>
      <h2>See how your clues affect other cells</h2>
      {!batch ? (
        <p className="description">Waiting for the current saved batch.</p>
      ) : !comparison ? (
        <p className="description">
          Once a batch with probabilities is loaded, edit a clue to compare
          the resulting probabilities with the previous completed state.
        </p>
      ) : <>
        <p className="changes-summary">
          <strong>{comparison.changed}</strong> of {comparison.eligible} cells
          unlocked in both states shifted by at least <strong>1 percentage point</strong>
          {' '}after {comparison.edits} clue {comparison.edits === 1 ? 'change' : 'changes'}.
        </p>
        <p className="fine">
          Last clue-change comparison: saved run #{comparison.from} → #{comparison.to}.
          Repeated draws with the same clues retain this comparison.
        </p>
        {comparison.largest.length > 0 ? (
          <div className="changes-table-wrap">
            <table className="changes-table">
              <caption>Largest shifts in probability of green</caption>
              <thead><tr>
                <th scope="col">Cell</th>
                <th scope="col">Before</th>
                <th scope="col">After</th>
                <th scope="col">Change</th>
              </tr></thead>
              <tbody>{comparison.largest.map(shift => (
                <tr key={shift.index}>
                  <th scope="row">
                    Row {Math.floor(shift.index / 16) + 1}, column {shift.index % 16 + 1}
                  </th>
                  <td>{(shift.before * 100).toFixed(1)}%</td>
                  <td>{(shift.after * 100).toFixed(1)}%</td>
                  <td className={shift.points > 0 ? 'shift-up' : 'shift-down'}>
                    {shift.points > 0 ? '+' : ''}{shift.points.toFixed(1)} pp
                  </td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : (
          <p className="description">
            {comparison.eligible === 0
              ? 'No cells were unlocked in both states.'
              : 'No eligible cell shifted by at least one percentage point.'}
          </p>
        )}
      </>}
      <p className="fine">
        “pp” means percentage points: 40% → 55% is +15 pp.
        This compares model probabilities, not sample colors or recognition accuracy.
        Rapid edits may be grouped between completed batches.
        Comparisons begin when this session is opened in this page.
      </p>
    </section>
  )
}
