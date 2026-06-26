import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { aiApi } from '../services/api'

const GENRES = [
  "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

const LEVELS = [
  { label: '−−', value: 'stark dämpfen',     delta: -0.5 },
  { label: '−',  value: 'leicht dämpfen',    delta: -0.25 },
  { label: '○',  value: 'neutral',            delta:  0 },
  { label: '+',  value: 'leicht verstärken', delta: +0.25 },
  { label: '++', value: 'stark verstärken',  delta: +0.5 },
]

function GenreRow({ genre, value, editable, override, onOverrideChange }) {
  const level   = LEVELS.find(l => l.value === (override || 'neutral')) || LEVELS[2]
  const adjusted = editable
    ? Math.max(0, Math.min(1, value + level.delta))
    : value
  const pct     = Math.round(adjusted * 100)
  const barClass = editable && level.delta > 0
    ? 'bar-fill bar-fill-boost'
    : editable && level.delta < 0
      ? 'bar-fill bar-fill-suppress'
      : 'bar-fill'

  return (
    <div className={`genre-row${editable ? ' genre-row-edit' : ''}`}>
      <span className="genre-name">{genre}</span>
      <div className="bar-track">
        <div className={barClass} style={{ width: `${pct}%` }} />
      </div>
      <span className="genre-pct">{pct}%</span>
      {editable && (
        <div className="override-btns">
          {LEVELS.map(lvl => (
            <button
              key={lvl.value}
              className={`override-btn${(override || 'neutral') === lvl.value ? ' active' : ''}`}
              onClick={() => onOverrideChange(genre, lvl.value)}
            >
              {lvl.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ProfilePage() {
  const [profile, setProfile]     = useState(null)
  const [overrides, setOverrides] = useState({})
  const [editMode, setEditMode]   = useState(false)
  const [loading, setLoading]     = useState(true)
  const [recLoading, setRecLoading] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    aiApi.getProfile()
      .then(({ data }) => setProfile(data.profile))
      .finally(() => setLoading(false))
  }, [])

  // Sort order fixed to AI profile — never changes while editing
  const sortedGenres = useMemo(() => {
    if (!profile) return GENRES
    return GENRES.slice().sort((a, b) => (profile[b] || 0) - (profile[a] || 0))
  }, [profile])

  const handleOverrideChange = (genre, level) =>
    setOverrides(prev => ({ ...prev, [genre]: level }))

  const activeOverrides = Object.fromEntries(
    Object.entries(overrides).filter(([, v]) => v && v !== 'neutral')
  )
  const activeCount = Object.keys(activeOverrides).length

  const toggleEditMode = () => {
    setEditMode(m => !m)
    setOverrides({})
  }

  const goRecommend = async () => {
    setRecLoading(true)
    try {
      if (editMode && activeCount > 0) {
        // Build float genre_weights from AI profile + LEVELS delta
        const genre_weights = {}
        for (const genre of sortedGenres) {
          const level = LEVELS.find(l => l.value === (overrides[genre] || 'neutral')) || LEVELS[2]
          genre_weights[genre] = Math.max(0, Math.min(1, (profile[genre] || 0) + level.delta))
        }
        const { data } = await aiApi.recommendFromProfile(genre_weights)
        navigate('/recommend', { state: { recommendations: data.recommendations } })
      } else {
        navigate('/recommend')
      }
    } finally {
      setRecLoading(false)
    }
  }

  if (loading) return <div className="loading">Analysing your taste…</div>

  const topGenre = sortedGenres[0]

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Your Taste Profile</h1>
          <p className="text-muted">
            AI-inferred genre affinities based on your ratings
            {!editMode && topGenre && (
              <> — your top genre is <strong>{topGenre}</strong></>
            )}
          </p>
        </div>
        <div className="page-header-actions">
          <button
            className={`btn-secondary${editMode ? ' active' : ''}`}
            onClick={toggleEditMode}
          >
            {editMode ? '← AI Profile' : '✏ Edit Profile'}
          </button>
          <button className="btn-primary" onClick={goRecommend} disabled={recLoading}>
            {recLoading
              ? 'Loading…'
              : editMode && activeCount > 0
                ? `Get Recommendations (${activeCount} override${activeCount > 1 ? 's' : ''}) →`
                : 'Get Recommendations →'}
          </button>
        </div>
      </div>

      {editMode && (
        <div className="info-banner">
          <strong>Edit Mode:</strong> Use the buttons to boost (+ / ++) or suppress (− / −−) any genre.
          The bars show your AI-inferred affinities. Click <em>Get Recommendations</em> to apply your adjustments.
        </div>
      )}

      <div className="profile-card">
        <div className="profile-section-title">
          {editMode
            ? `Manual Genre Overrides${activeCount > 0 ? ` — ${activeCount} active` : ' — none active yet'}`
            : 'AI-Inferred Genre Profile'}
        </div>
        <div className="genre-list">
          {sortedGenres.map(genre => (
            <GenreRow
              key={genre}
              genre={genre}
              value={profile[genre] || 0}
              editable={editMode}
              override={overrides[genre]}
              onOverrideChange={handleOverrideChange}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
