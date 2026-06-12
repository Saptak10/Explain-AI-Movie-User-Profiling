import { useState } from 'react'

export default function StarRating({ value, onChange }) {
  const [hover, setHover] = useState(0)
  const display = hover || value

  return (
    <div className="stars" onMouseLeave={() => setHover(0)}>
      {[1, 2, 3, 4, 5].map(s => (
        <button
          key={s}
          type="button"
          className={`star${display >= s ? ' filled' : ''}`}
          onClick={() => onChange(s)}
          onMouseEnter={() => setHover(s)}
          title={`${s} star${s > 1 ? 's' : ''}`}
        >
          ★
        </button>
      ))}
      {value > 0 && <span className="star-label">{value}.0</span>}
    </div>
  )
}
