import { useId, useRef } from 'react'

type Props = {
  names: string[]
  bits: number[]
  cells: readonly number[]
}

export default function ReferencePreview({ names, bits, cells }: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  const titleId = useId()
  const name = names.join(' / ')
  const clues = cells.flatMap((value, index) =>
    value === -1 ? [] : [{ index, value, matches: bits[index] === value }]
  )
  const conflicts = clues.filter(clue => !clue.matches).length

  function grid() {
    return (
      <span className="pixel-grid" aria-hidden="true">
        {bits.map((bit, index) => (
          <span
            key={index}
            className={`pixel ${bit === 1 ? 'green' : 'purple'} ${
              cells[index] === -1 ? '' :
              cells[index] === bit ? 'clue-match' : 'clue-conflict'
            }`}
            title={cells[index] === -1 ? undefined :
              `Row ${Math.floor(index / 16) + 1}, column ${index % 16 + 1}: ${
                cells[index] === bit ? 'matches' : 'conflicts with'
              } your ${cells[index] === 1 ? 'green' : 'purple'} clue`}
          />
        ))}
      </span>
    )
  }

  return (
    <>
      <button
        type="button"
        className="reference-preview-button"
        aria-label={`Inspect ${name}; ${conflicts} clue conflicts`}
        aria-haspopup="dialog"
        onClick={() => dialog.current?.showModal()}
      >
        {grid()}
        <span className="reference-inspect-hint">Inspect reference ↗</span>
      </button>
      <dialog
        ref={dialog}
        className="reference-dialog"
        aria-labelledby={titleId}
        onClick={event => {
          if (event.target === event.currentTarget) dialog.current?.close()
        }}
      >
        <div className="reference-dialog-content">
          <header className="reference-dialog-header">
            <div>
              <p className="eyebrow">TRAINING REFERENCE</p>
              <h2 id={titleId}>{name}</h2>
            </div>
            <form method="dialog">
              <button type="submit" className="card-close" aria-label="Close reference inspector">×</button>
            </form>
          </header>
          <p className="description">
            {conflicts === 0 ? 'Compatible with every current clue.' :
              `${conflicts} of ${clues.length} locked clues conflict with this reference.`}
          </p>
          <div className="reference-enlarged" role="img"
            aria-label={`${name}: 16 by 16 training reference, ${conflicts} clue conflicts`}>
            {grid()}
          </div>
          <p className="fine">
            White outline = matching clue. White × = conflicting clue.
            Green = 1; purple = 0. This is an original training reference.
          </p>
          <details className="reference-clue-details">
            <summary>Clue positions and values ({clues.length})</summary>
            {clues.length === 0 ? <p>No cells are locked.</p> : (
              <ul>{clues.map(clue => (
                <li key={clue.index}>
                  Row {Math.floor(clue.index / 16) + 1}, column {clue.index % 16 + 1}:
                  {' '}clue {clue.value === 1 ? 'green' : 'purple'},
                  {' '}reference {bits[clue.index] === 1 ? 'green' : 'purple'}
                  {' '}— {clue.matches ? 'matches' : 'conflicts'}.
                </li>
              ))}</ul>
            )}
          </details>
        </div>
      </dialog>
    </>
  )
}
