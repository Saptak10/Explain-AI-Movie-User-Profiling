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

  const handleRate = (movieId, star) => {
    setRatings(prev => ({ ...prev, [movieId]: star }))
  }

  const handleSave = async () => {
    setSaving(true)
    const changed = Object.entries(ratings).filter(
      ([id, r]) => r > 0 && savedRatings[id] !== r
    )
    try {
      await Promise.all(changed.map(([id, r]) => aiApi.submitRating(Number(id), r)))
      setSavedRatings({ ...ratings })
      navigate('/profile')
    } finally {
      setSaving(false)
    }
  }

  const ratedCount = Object.values(ratings).filter(r => r > 0).length

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
          <span className="badge">{ratedCount} rated</span>
          <button
            className="btn-primary"
            onClick={handleSave}
            disabled={ratedCount === 0 || saving}
          >
            {saving ? 'Saving…' : 'Build My Profile →'}
          </button>
        </div>
      </div>

      <div className="movie-grid">
        {movies.map(m => (
          <div key={m.id} className={`movie-card${ratings[m.id] > 0 ? ' rated' : ''}`}>
            <div className="movie-title">{m.title}</div>
            <StarRating
              value={ratings[m.id] || 0}
              onChange={star => handleRate(m.id, star)}
            />
          </div>
        ))}
      </div>

      {ratedCount > 0 && (
        <div className="sticky-cta">
          <span className="text-muted">{ratedCount} movie{ratedCount > 1 ? 's' : ''} rated</span>
          <button className="btn-primary" onClick={handleSave} disabled={saving}>
            {saving ? 'Saving…' : 'Build My Profile →'}
          </button>
        </div>
      )}
    </div>
  )
}
