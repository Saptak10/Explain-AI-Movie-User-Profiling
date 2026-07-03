import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import StarRating from '../components/StarRating'
import { aiApi, moviesApi } from '../services/api'

export default function RatingsPage() {
  const [movies, setMovies] = useState([])
  const [ratings, setRatings] = useState({})
  const [savedRatings, setSavedRatings] = useState({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    Promise.all([moviesApi.popular(), aiApi.getRatings()])
      .then(([movRes, ratRes]) => {
        setMovies(movRes.data.movies)
        const r = ratRes.data.ratings
        setRatings(r)
        setSavedRatings(r)
      })
      .finally(() => setLoading(false))
  }, [])

  const MAX_RATINGS = 10
  const MIN_RATINGS = 5

  const ratedCount = Object.values(ratings).filter(r => r > 0).length

  const handleRate = (movieId, star) => {
    setRatings(prev => {
      const alreadyRated = (prev[movieId] || 0) > 0
      if (!alreadyRated && ratedCount >= MAX_RATINGS) return prev
      return { ...prev, [movieId]: star }
    })
  }

  const handleSave = async () => {
    setSaving(true)
    const changed = Object.entries(ratings).filter(
      ([id, r]) => r > 0 && savedRatings[id] !== r
    )
    try {
      await Promise.all(changed.map(([id, r]) => aiApi.submitRating(Number(id), r, 0)))
      setSavedRatings({ ...ratings })
      navigate('/profile')
    } finally {
      setSaving(false)
    }
  }

  if (loading) return <div className="loading">Loading movies…</div>

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Rate Some Movies</h1>
          <p className="text-muted">
            Rate movies you've seen — the AI will infer your genre preferences from your ratings
          </p>
        </div>
        <div className="page-header-actions">
          <span className="badge">{ratedCount}/{MAX_RATINGS} rated</span>
          <button
            className="btn-primary"
            onClick={handleSave}
            disabled={ratedCount < MIN_RATINGS || saving}
            title={ratedCount < MIN_RATINGS ? `Rate at least ${MIN_RATINGS} movies to continue` : undefined}
          >
            {saving ? 'Saving…' : 'Build My Profile →'}
          </button>
        </div>
      </div>

      <div className="movie-grid">
        {movies.map(m => {
          const isRated = ratings[m.id] > 0
          const limitReached = !isRated && ratedCount >= MAX_RATINGS
          return (
            <div
              key={m.id}
              className={`movie-card${isRated ? ' rated' : ''}${limitReached ? ' disabled' : ''}`}
              title={limitReached ? `Limit reached — you can rate up to ${MAX_RATINGS} movies` : undefined}
            >
              <div className="movie-title">{m.title}</div>
              <StarRating
                value={ratings[m.id] || 0}
                onChange={star => handleRate(m.id, star)}
                disabled={limitReached}
              />
            </div>
          )
        })}
      </div>

      {ratedCount > 0 && (
        <div className="sticky-cta">
          <span className="text-muted">
            {ratedCount}/{MAX_RATINGS} rated
            {ratedCount < MIN_RATINGS && ` — rate ${MIN_RATINGS - ratedCount} more to continue`}
          </span>
          <button
            className="btn-primary"
            onClick={handleSave}
            disabled={ratedCount < MIN_RATINGS || saving}
          >
            {saving ? 'Saving…' : 'Build My Profile →'}
          </button>
        </div>
      )}
    </div>
  )
}
