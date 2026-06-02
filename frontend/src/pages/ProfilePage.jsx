import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { aiApi } from '../services/api'

const GENRES = [
  "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

function GenreRow({ genre, value, editable, onChange }) {
  const pct = Math.round(value * 100)
  return (
    <div className="genre-row">
      <span className="genre-name">{genre}</span>
      {editable ? (
        <input
          type="range"
          min={0}
          max={100}
          value={pct}
          onChange={e => onChange(Number(e.target.value) / 100)}
          className="genre-slider"
        />
      ) : (
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${pct}%` }} />
        </div>
      )}
      <span className="genre-pct">{pct}%</span>
    </div>
  )
}

export default function ProfilePage() {
  const [profile, setProfile] = useState(null)
  const [edited, setEdited] = useState(null)
  const [editMode, setEditMode] = useState(false)
  const [loading, setLoading] = useState(true)
  const [recLoading, setRecLoading] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    aiApi.getProfile()
      .then(({ data }) => {
        setProfile(data.profile)
        setEdited({ ...data.profile })
      })
      .finally(() => setLoading(false))
  }, [])

  const handleChange = (genre, val) => {
    setEdited(prev => ({ ...prev, [genre]: val }))
  }

  const goRecommend = async () => {
    setRecLoading(true)
    try {
      if (editMode) {
        const { data } = await aiApi.recommendFromProfile(edited)
        navigate('/recommend', {
          state: { recommendations: data.recommendations, fromEdited: true },
        })
      } else {
        navigate('/recommend')
      }
    } finally {
      setRecLoading(false)
    }
  }

  if (loading) return <div className="loading">Analysing your taste…</div>

  const display = editMode ? edited : profile
  const sorted = GENRES.slice().sort((a, b) => (display[b] || 0) - (display[a] || 0))
  const topGenre = sorted[0]

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
            onClick={() => setEditMode(m => !m)}
          >
            {editMode ? '← AI Profile' : '✏ Edit Profile'}
          </button>
          <button className="btn-primary" onClick={goRecommend} disabled={recLoading}>
            {recLoading ? 'Loading…' : 'Get Recommendations →'}
          </button>
        </div>
      </div>

      {editMode && (
        <div className="info-banner">
          <strong>Edit Mode:</strong> Drag sliders to set your genre affinities manually.
          Clicking <em>Get Recommendations</em> will use this edited profile instead of the AI profile.
        </div>
      )}

      <div className="profile-card">
        <div className="profile-section-title">
          {editMode ? 'Manual Genre Profile' : 'AI-Inferred Genre Profile'}
        </div>
        <div className="genre-list">
          {sorted.map(genre => (
            <GenreRow
              key={genre}
              genre={genre}
              value={display[genre] || 0}
              editable={editMode}
              onChange={val => handleChange(genre, val)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
