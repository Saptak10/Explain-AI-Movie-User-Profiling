import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { aiApi } from '../services/api'

// Deltas are on the same [0,1] scale the backend/model use for genre
// weights (see forward_interactive). Only the display math (GenreRow)
// converts to percentage points; the payload sent to the backend stays
// on the [0,1] scale these deltas are defined in. Persisted server-side
// (profile_overrides table) — picking a level here isn't a one-off
// nudge, it replaces the saved override for that genre going forward.
const LEVELS = [
  { label: '−−', value: 'stark dämpfen',     delta: -0.5 },
  { label: '−',  value: 'leicht dämpfen',    delta: -0.25 },
  { label: '○',  value: 'neutral',            delta:  0 },
  { label: '+',  value: 'leicht verstärken', delta: +0.25 },
  { label: '++', value: 'stark verstärken',  delta: +0.5 },
]

const levelForDelta = (delta) =>
  (LEVELS.find(l => l.delta === delta) || LEVELS[2]).value

function GenreCitations({ explaining, citations }) {
  if (explaining) return <div className="loading-sm">Analysing your ratings…</div>
  if (!citations || citations.length === 0) {
    return <p className="text-muted explain-text-inline">No single rating stands out — this score comes from your overall pattern.</p>
  }
  const maxImp = Math.max(...citations.map(c => Math.abs(c.importance)), 0.001)
  return (
    <div className="edit-genre-list">
      {citations.map(c => (
        <div key={c.movie_id} className="edit-genre-row">
          <div className="edit-genre-info">
            <span className="edit-genre-name" title={c.title}>{c.title}</span>
            <div className="mini-bar-track">
              <div className="mini-bar-fill" style={{ width: `${(Math.abs(c.importance) / maxImp) * 100}%` }} />
            </div>
          </div>
          <span className="score-label">{c.importance.toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}

function GenreRow({ genre, value, editable, override, onOverrideChange, expanded, explaining, citations, onExplainToggle }) {
  // `value` is already a 0-100 percentage from the backend.
  const level   = LEVELS.find(l => l.value === (override || 'neutral')) || LEVELS[2]
  const adjusted = editable
    ? Math.max(0, Math.min(100, value + level.delta * 100))
    : value
  const pct     = Math.round(adjusted)
  const barClass = editable && level.delta > 0
    ? 'bar-fill bar-fill-boost'
    : editable && level.delta < 0
      ? 'bar-fill bar-fill-suppress'
      : 'bar-fill'

  return (
    <div className={`genre-row-wrap${expanded ? ' expanded' : ''}`}>
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
        {!editable && (
          <button className="btn-edit-prefs" onClick={() => onExplainToggle(genre)}>
            {expanded ? 'Why? ▲' : 'Why? ▼'}
          </button>
        )}
      </div>
      {expanded && (
        <GenreCitations explaining={explaining} citations={citations} />
      )}
    </div>
  )
}

export default function ProfilePage() {
  const location = useLocation()
  // If we arrived here right after "Build My Profile →", RatingsPage already
  // fetched a personalized profile for us — use it instead of re-fetching
  // the shared (non-personalized) profile.
  const [profile, setProfile]     = useState(location.state?.profile || null)
  // Overrides currently being edited (level strings, keyed by genre) —
  // distinct from savedOverrides (raw persisted deltas) so the UI can
  // show unsaved changes before "Get Recommendations" is clicked.
  const [overrides, setOverrides] = useState({})
  const [savedOverrides, setSavedOverrides] = useState({})
  const [editMode, setEditMode]   = useState(false)
  const [loading, setLoading]     = useState(!location.state?.profile)
  const [recLoading, setRecLoading] = useState(false)
  const [recError, setRecError]     = useState('')
  const [resetting, setResetting]   = useState(false)
  const [explanations, setExplanations] = useState(null)
  const [explaining, setExplaining]     = useState(false)
  const [expandedGenre, setExpandedGenre] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    // profile already reflects any saved overrides (backend merges them
    // in, see GET /api/profile) — fetch the raw deltas separately just to
    // pre-select which level button is "active" per genre in edit mode.
    aiApi.getOverrides().then(({ data }) => setSavedOverrides(data.overrides))
    if (location.state?.profile) return
    aiApi.getProfile()
      .then(({ data }) => setProfile(data.profile))
      .finally(() => setLoading(false))
  }, [])

  // Sort order fixed to AI profile — never changes while editing.
  // Genre names come from the profile itself (backend's dynamic
  // genre vocabulary), never hardcoded, so this can't drift out of sync.
  const sortedGenres = useMemo(() => {
    if (!profile) return []
    return Object.keys(profile).sort((a, b) => (profile[b] || 0) - (profile[a] || 0))
  }, [profile])

  // `profile[genre]` already has any saved override baked in (the backend
  // merges it into GET /api/profile). To preview a *newly selected* level
  // without double-counting the already-applied saved delta, reconstruct
  // the pre-override AI value and let GenreRow's existing
  // `value + level.delta*100` math re-apply whichever level is currently
  // selected (the saved one, by default, or a new pick).
  const baseProfile = useMemo(() => {
    if (!profile) return {}
    const base = {}
    for (const genre of Object.keys(profile)) {
      base[genre] = (profile[genre] || 0) - (savedOverrides[genre] || 0) * 100
    }
    return base
  }, [profile, savedOverrides])

  const handleOverrideChange = (genre, level) =>
    setOverrides(prev => ({ ...prev, [genre]: level }))

  const activeOverrides = Object.fromEntries(
    Object.entries(overrides).filter(([, v]) => v && v !== 'neutral')
  )
  const activeCount = Object.keys(activeOverrides).length

  const toggleEditMode = () => {
    setEditMode(m => {
      const entering = !m
      if (entering) {
        // Pre-select whichever level is currently saved per genre, so
        // re-entering edit mode shows past edits instead of resetting.
        const initial = {}
        for (const [genre, delta] of Object.entries(savedOverrides)) {
          initial[genre] = levelForDelta(delta)
        }
        setOverrides(initial)
      } else {
        setOverrides({})
      }
      return entering
    })
  }

  const resetToAiProfile = async () => {
    setResetting(true)
    try {
      await aiApi.clearOverrides()
      setSavedOverrides({})
      setOverrides({})
      const { data } = await aiApi.getProfile()
      setProfile(data.profile)
    } finally {
      setResetting(false)
    }
  }

  // Explanations are fetched lazily on first "Why?" click, not on page
  // load — the endpoint costs extra forward passes per genre, so that
  // cost is only paid if the user actually asks for it. Cached in state
  // after the first fetch, shared across all genre rows.
  const handleExplainToggle = (genre) => {
    if (expandedGenre === genre) {
      setExpandedGenre(null)
      return
    }
    setExpandedGenre(genre)
    if (!explanations) {
      setExplaining(true)
      aiApi.explainProfile()
        .then(({ data }) => setExplanations(data.genre_explanations))
        .finally(() => setExplaining(false))
    }
  }

  const goRecommend = async () => {
    setRecLoading(true)
    setRecError('')
    try {
      if (editMode && Object.keys(overrides).length > 0) {
        // Every genre touched this session (pre-populated from saved
        // overrides on entering edit mode, or explicitly changed) — send
        // its delta as-is, including 0 (neutral), which tells the backend
        // to clear any previously saved override for that genre. This
        // persists server-side, so it keeps affecting recommendations on
        // every future visit, not just this one.
        const genre_deltas = {}
        for (const [genre, level] of Object.entries(overrides)) {
          const lvl = LEVELS.find(l => l.value === level) || LEVELS[2]
          genre_deltas[genre] = lvl.delta
        }
        const { data } = await aiApi.recommendFromProfile(genre_deltas)
        setSavedOverrides(prev => {
          const next = { ...prev }
          for (const [genre, delta] of Object.entries(genre_deltas)) {
            if (delta === 0) delete next[genre]
            else next[genre] = delta
          }
          return next
        })
        navigate('/recommend', { state: { topRated: data.top_rated, forYou: data.for_you } })
      } else {
        navigate('/recommend')
      }
    } catch (err) {
      setRecError(err.response?.data?.detail || 'Failed to get recommendations')
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
          {Object.keys(savedOverrides).length > 0 && (
            <button
              className="btn-secondary"
              onClick={resetToAiProfile}
              disabled={resetting}
              title="Discard all saved genre edits and revert to the pure AI-inferred profile"
            >
              {resetting ? 'Resetting…' : '↺ Reset to AI Profile'}
            </button>
          )}
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
          The bars show your AI-inferred affinities. Click <em>Get Recommendations</em> to save your
          adjustments — they'll keep applying on future visits until you change or reset them.
        </div>
      )}

      {recError && <p className="error-msg">{recError}</p>}

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
              value={(editMode ? baseProfile[genre] : profile[genre]) || 0}
              editable={editMode}
              override={overrides[genre]}
              onOverrideChange={handleOverrideChange}
              expanded={expandedGenre === genre}
              explaining={explaining && expandedGenre === genre}
              citations={explanations?.[genre]}
              onExplainToggle={handleExplainToggle}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
