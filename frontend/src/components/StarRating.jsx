import { useState } from 'react'

// Half-star widget: each star's clickable area is split left/right (left
// half -> s - 0.5, right half -> s), matching the training data's actual
// 0.5-5.0 rating granularity (the backend already accepts floats — this
// was previously the only place ratings got rounded to whole stars).
function halfFromEvent(e, starIndex) {
  const rect = e.currentTarget.getBoundingClientRect()
  const isLeftHalf = e.clientX - rect.left < rect.width / 2
  return isLeftHalf ? starIndex - 0.5 : starIndex
}

export default function StarRating({ value, onChange, disabled = false }) {
  const [hover, setHover] = useState(0)
  const display = hover || value

  return (
    <div className={`stars${disabled ? ' disabled' : ''}`} onMouseLeave={() => setHover(0)}>
      {[1, 2, 3, 4, 5].map(s => {
        const fillFraction = Math.max(0, Math.min(1, display - (s - 1)))
        return (
          <button
            key={s}
            type="button"
            className="star"
            onClick={e => !disabled && onChange(halfFromEvent(e, s))}
            onMouseMove={e => !disabled && setHover(halfFromEvent(e, s))}
            disabled={disabled}
            title={`${display >= s ? s : s - 0.5} star${s > 1 ? 's' : ''}`}
          >
            <span className="star-bg">★</span>
            <span className="star-fg" style={{ width: `${fillFraction * 100}%` }}>★</span>
          </button>
        )
      })}
      {value > 0 && <span className="star-label">{value.toFixed(1)}</span>}
    </div>
  )
}
