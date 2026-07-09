import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import StarRating from '../components/StarRating'
import { aiApi, moviesApi } from '../services/api'

export default function RatingsPage() {
  const [movies, setMovies] = useState([])
  const [ratings, setRatings] = useState({})
  const [savedRatings, setSavedRatings] = useState({})
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching] = useState(false)
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

  // Debounced search-by-title, so a user can find and rate a specific
  // movie even if it never showed up in the random popular-sample batch.
  useEffect(() => {
    const query = searchQuery.trim()
    if (query.length < 2) {
      setSearchResults([])
      setSearching(false)
      return
    }
    setSearching(true)
    const timer = setTimeout(() => {
      moviesApi.search(query)
        .then(({ data }) => setSearchResults(data.movies))
        .finally(() => setSearching(false))
    }, 300)
    return () => clearTimeout(timer)
  }, [searchQuery])

  const handleRate = (movieId, star) => {
    setRatings(prev => {
      const alreadyRated = (prev[movieId] || 0) > 0
      if (!alreadyRated && ratedCount >= MAX_RATINGS) return prev
      return { ...prev, [movieId]: star }
    })
  }

  // Adds a searched-up movie to the rateable grid (a no-op if it's
  // already there, e.g. also present in the current popular sample).
  const handleAddMovie = (movie) => {
    setMovies(prev => (prev.some(m => m.id === movie.id) ? prev : [movie, ...prev]))
    setSearchQuery('')
    setSearchResults([])
  }

  // Swaps in a fresh batch of movies to rate, excluding the ones
  // currently shown, for users whose first batch didn't include enough
  // movies they've actually seen. Doesn't touch `ratings` — any stars
  // already given (including for movies no longer displayed) are kept
  // and still count toward the profile when saved.
  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      const { data } = await moviesApi.popular(movies.map(m => m.id))
      setMovies(data.movies)
    } finally {
      setRefreshing(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    const changed = Object.entries(ratings).filter(
      ([id, r]) => r > 0 && savedRatings[id] !== r
    )
    try {
      await Promise.all(changed.map(([id, r]) => aiApi.submitRating(Number(id), r, 0)))
      setSavedRatings({ ...ratings })
      // One-time personalization fine-tune on this user's own ratings,
      // right as their profile is built. The result is only used for this
      // navigation -- later visits to /profile fetch the shared model as usual.
      try {
        const { data } = await aiApi.personalizeProfile()
        navigate('/profile', { state: { profile: data.profile } })
      } catch {
        navigate('/profile')
      }
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
            className="btn-secondary"
            onClick={handleRefresh}
            disabled={refreshing}
            title="Don't recognize these? Get a different batch of movies to rate."
          >
            {refreshing ? 'Refreshing…' : '🔄 Refresh Suggestions'}
          </button>
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

      <div className="movie-search">
        <input
          type="text"
          className="movie-search-input"
          placeholder="Search for a movie by name to rate it…"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
        {searchQuery.trim().length >= 2 && (
          <div className="movie-search-results">
            {searching && <div className="text-muted">Searching…</div>}
            {!searching && searchResults.length === 0 && (
              <div className="text-muted">No movies found.</div>
            )}
            {!searching && searchResults.map(m => (
              <div key={m.id} className="movie-search-result">
                <span>{m.title}</span>
                <button
                  className="btn-secondary"
                  disabled={movies.some(mv => mv.id === m.id)}
                  onClick={() => handleAddMovie(m)}
                >
                  {movies.some(mv => mv.id === m.id) ? 'Added' : 'Add'}
                </button>
              </div>
            ))}
          </div>
        )}
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
