export default function StarRating({ value, onChange }) {
  return (
    <div className="stars">
      {[1, 2, 3, 4, 5].map(s => (
        <button
          key={s}
          type="button"
          className={`star${value >= s ? ' filled' : ''}`}
          onClick={() => onChange(s)}
          title={`${s} star${s > 1 ? 's' : ''}`}
        >
          ★
        </button>
      ))}
      {value > 0 && <span className="star-label">{value}.0</span>}
    </div>
  )
}
