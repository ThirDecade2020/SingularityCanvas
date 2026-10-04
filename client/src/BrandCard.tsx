import { useRef } from 'react'

const image = '/brand/singularity-machines-card.png'

export default function BrandCard() {
  const dialog = useRef<HTMLDialogElement>(null)

  return (
    <aside className="brand-card-area" aria-label="About Singularity Machines">
      <button className="brand-card-preview" type="button"
        aria-label="Open the Singularity Machines company card"
        aria-haspopup="dialog"
        onClick={() => dialog.current?.showModal()}>
        <img src={image} alt="Singularity Machines company card" width={1080} height={1350} />
        <span>Explore Singularity Machines ↗</span>
      </button>
      <p className="brand-card-caption">LOCAL RESEARCH PREVIEW · 16 × 16</p>
      <dialog ref={dialog} className="brand-card-dialog"
        aria-labelledby="brand-card-title"
        onClick={event => {
          if (event.target === event.currentTarget) dialog.current?.close()
        }}>
        <div className="brand-card-dialog-content">
          <header>
            <h2 id="brand-card-title">Singularity Machines</h2>
            <form method="dialog">
              <button className="card-close" aria-label="Close company card">×</button>
            </form>
          </header>
          <img src={image}
            alt="Singularity Machines. Maximum Control, Minimum Abstraction. Fast app creation, rapid software services, AI agentic automation, sovereign compute systems, and local and cloud-based development. YouTube and LinkedIn QR codes."
            width={1080} height={1350} />
          <a href={image} target="_blank" rel="noreferrer">Open original image ↗</a>
        </div>
      </dialog>
    </aside>
  )
}
