import React from 'react'

export default function Modal({ open, onClose, title, children }) {
  if (!open) return null

  const onOverlay = (e) => { if (e.target === e.currentTarget) onClose?.() }

  return (
    <div className="modal-overlay" onMouseDown={onOverlay}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <header className="modal-header">
          <div className="modal-title">{title}</div>
          <button className="icon-btn" onClick={onClose} title="닫기 (Esc)" aria-label="닫기">
            <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
              <path fill="currentColor" d="M18.3 5.71L12 12l6.3 6.29-1.41 1.42L10.59 13.4 4.3 19.71 2.89 18.3 9.17 12 2.89 5.71 4.3 4.29 10.59 10.6 16.89 4.29z"/>
            </svg>
          </button>
        </header>
        <div className="modal-body">
          {children}
        </div>
      </div>
    </div>
  )
}
