import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { aiApi } from '../services/api'
import { useAuth } from '../context/AuthContext'
import StarRating from '../components/StarRating'

const LEVELS = [
  { label: '−−', value: 'stark dämpfen',     delta: -0.5 },
  { label: '−',  value: 'leicht dämpfen',    delta: -0.25 },
  { label: '○',  value: 'neutral',            delta:  0 },
  { label: '+',  value: 'leicht verstärken', delta: +0.25 },
  { label: '++', value: 'stark verstärken',  delta: +0.5 },
]

const VERSION_INFO = {
  O: {
    label: 'Version O — Transparent AI',
    badgeClass: 'version-o',
    description:
      'You are in the Transparent AI condition. You can see why each movie was recommended ' +
      'and adjust the genre weights that drive your recommendations.',
  },
  N: {
    label: 'Version N — Standard AI',
    badgeClass: 'version-n',
    description:
      'You are in the Standard AI condition. You receive AI-generated recommendations ' +
      'without explanations. Simply review the list and continue to the survey when ready.',
  },
}

function ScoreBar({ score }) {
  return (
    <div className="score-bar-wrap">
      <div className="score-track">
        <div className="score-fill" style={{ width: `${(score / 5) * 100}%` }} />
      </div>
      <span className="score-label">{score.toFixed(2)}</span>
    </div>
  )
}

function ImportanceBar({ value, max }) {
  const pct = max > 0 ? (Math.abs(value) / max) * 100 : 0
  return (
    <div className="mini-bar-track">
      <div className="mini-bar-fill" style={{ width: `${pct}%` }} />
    </div>
  )
}

function GenreOverrideRow({ genre, value, level, onOverrideChange }) {
  const lvl = LEVELS.find(l => l.value === (level || 'neutral')) || LEVELS[2]
  const adjusted = Math.max(0, Math.min(1, value + lvl.delta))
  const pct = Math.round(adjusted * 100)
  const barClass = lvl.delta > 0
    ? 'bar-fill bar-fill-boost'
    : lvl.delta < 0
      ? 'bar-fill bar-fill-suppress'
      : 'bar-fill'

  return (
    <div className="genre-row genre-row-edit">
      <span className="genre-name">{genre}</span>
      <div className="bar-track">
        <div className={barClass} style={{ width: `${pct}%` }} />
      </div>
      <span className="genre-pct">{pct}%</span>
      <div className="override-btns">
        {LEVELS.map(l => (
          <button
            key={l.value}
            className={`override-btn${(level || 'neutral') === l.value ? ' active' : ''}`}
            onClick={() => onOverrideChange(genre, l.value)}
          >
            {l.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function EditPanel({ explainData, genreProfile, overrides, applying, onOverrideChange, onApply }) {
  if (!explainData) return <div className="loading-sm">Analysing recommendation…</div>

  const topMovies = (explainData.feature_importance || []).slice(0, 5)
  const maxImp    = Math.max(...topMovies.map(m => Math.abs(m.importance)), 0.001)

  // Top genres from profile for override buttons (sorted by profile value desc)
  const topGenres = Object.entries(genreProfile || {})
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)

  return (
    <div className="edit-panel">
      {explainData.rationale && (
        <p className="explain-text-inline">{explainData.rationale}</p>
      )}

      {topMovies.length > 0 && (
        <>
          <p className="edit-panel-label">Influenced by movies you rated:</p>
          <div className="edit-genre-list">
            {topMovies.map(m => (
              <div key={m.movie_id} className="edit-genre-row">
                <div className="edit-genre-info">
                  <span className="edit-genre-name" title={m.title}>{m.title}</span>
                  <ImportanceBar value={m.importance} max={maxImp} />
                </div>
                <span className="score-label">{m.importance.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {topGenres.length > 0 && (
        <>
          <p className="edit-panel-label">Adjust genre preferences for this movie:</p>
          <div className="edit-genre-list">
            {topGenres.map(([genre, value]) => (
              <GenreOverrideRow
                key={genre}
                genre={genre}
                value={value}
                level={overrides[genre]}
                onOverrideChange={onOverrideChange}
              />
            ))}
          </div>
        </>
      )}

      <button className="btn-primary" onClick={onApply} disabled={applying}>
        {applying ? 'Updating recommendations…' : 'Apply & Refresh Recommendations'}
      </button>
    </div>
  )
}

function ProfileEditPanel({ genreProfile, overrides, applying, onOverrideChange, onApply }) {
  const topGenres = Object.entries(genreProfile || {})
    .sort(([, a], [, b]) => b - a)
    .slice(0, 6)

  return (
    <div className="profile-edit-panel">
      <p className="edit-panel-label">Adjust your overall taste profile:</p>
      <div className="edit-genre-list">
        {topGenres.map(([genre, value]) => (
          <GenreOverrideRow
            key={genre}
            genre={genre}
            value={value}
            level={overrides[genre]}
            onOverrideChange={onOverrideChange}
          />
        ))}
      </div>
      <button className="btn-primary" onClick={onApply} disabled={applying}>
        {applying ? 'Updating recommendations…' : 'Apply & Refresh Recommendations'}
      </button>
    </div>
  )
}

function RecCard({ rank, rec, expanded, explainData, genreProfile, overrides, applying,
                   onToggle, onOverrideChange, onApply, showEdit, rating, onRate }) {
  return (
    <div className={`rec-card-full${expanded ? ' expanded' : ''}`}>
      <div className="rec-card-header">
        <div className="rec-rank">#{rank}</div>
        <div className="rec-info">
          <div className="rec-title">{rec.title}</div>
          <ScoreBar score={rec.score} />
        </div>
        {showEdit && (
          <button className="btn-edit-prefs" onClick={onToggle}>
            {expanded ? 'Close ▲' : 'Edit Preferences ▼'}
          </button>
        )}
      </div>

      <div className="rec-card-rate">
        <span className="rec-card-rate-label">Rate this recommendation:</span>
        <StarRating value={rating || 0} onChange={onRate} />
      </div>

      {showEdit && expanded && (
        <EditPanel
          explainData={explainData}
          genreProfile={genreProfile}
          overrides={overrides}
          applying={applying}
          onOverrideChange={onOverrideChange}
          onApply={onApply}
        />
      )}
    </div>
  )
}

export default function RecommendPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const { user } = useAuth()
  const version     = user?.version || 'O'
  const isTransparent = version === 'O'
  const info        = VERSION_INFO[version] || VERSION_INFO['O']

  // Counterbalanced edit order assigned at registration (Version O only).
  // 'movie_first' (default) -> round 1 edits per-movie weights, round 2 edits the whole profile.
  // 'profile_first'         -> round 1 edits the whole profile, round 2 edits per-movie weights.
  const editOrder  = user?.edit_order || 'movie_first'
  const firstType  = editOrder === 'profile_first' ? 'profile' : 'movie'
  const secondType = firstType === 'movie' ? 'profile' : 'movie'

  const [round, setRound]                       = useState(location.state?.round || 1)
  const [recs, setRecs]                         = useState(location.state?.recommendations || [])
  const [loading, setLoading]                   = useState(!location.state?.recommendations)
  const [expandedId, setExpandedId]             = useState(null)
  const [explainCache, setExplainCache]         = useState({})
  const [movieOverrides, setMovieOverrides]     = useState({})
  const [profileOverrides, setProfileOverrides] = useState({})
  const [applyingFor, setApplyingFor]           = useState(null)
  const [applyingProfile, setApplyingProfile]   = useState(false)
  const [hasEdited, setHasEdited]               = useState(false) // ever edited (drives /user/mark-edited)
  const [roundEdited, setRoundEdited]           = useState(false) // edited during the current round
  const [recRatings, setRecRatings]             = useState({})
  const [genreProfile, setGenreProfile]         = useState({})

  const editType = round === 1 ? firstType : secondType

  const fireLogRecs = (newRecs, currentRound, recType) => {
    aiApi.logRecommendations(
      currentRound,
      recType,
      newRecs.map((r, i) => ({ movie_id: r.movie_id, position: i + 1, score: r.score }))
    ).catch(() => {})
  }

  const loadRecs = async () => {
    try {
      const { data } = await aiApi.recommend(10)
      setRecs(data.recommendations)
      fireLogRecs(data.recommendations, round, 'initial')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!location.state?.recommendations) loadRecs()
    aiApi.getProfile().then(({ data }) => setGenreProfile(data.profile))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleToggle = async (movieId) => {
    if (expandedId === movieId) { setExpandedId(null); return }
    setExpandedId(movieId)
    if (!explainCache[movieId]) {
      try {
        const { data } = await aiApi.explain(movieId)
        setExplainCache(prev => ({ ...prev, [movieId]: data }))
      } catch {
        setExplainCache(prev => ({ ...prev, [movieId]: { rationale: null, feature_importance: [] } }))
      }
    }
  }

  const handleOverrideChange = (movieId, genre, level) => {
    setMovieOverrides(prev => ({
      ...prev,
      [movieId]: { ...(prev[movieId] || {}), [genre]: level },
    }))
  }

  const handleProfileOverrideChange = (genre, level) =>
    setProfileOverrides(prev => ({ ...prev, [genre]: level }))

  const handleRateRec = async (movieId, star) => {
    setRecRatings(prev => ({ ...prev, [movieId]: star }))
    await aiApi.submitRating(movieId, star, round)
  }

  const logEdits = (edit_type, changedEntries, movieId = null) =>
    Promise.all(
      changedEntries.map(([genre, level]) =>
        aiApi.logProfileEdit(round, edit_type, genre, level, movieId)
      )
    )

  const markEditedOnce = async () => {
    if (!hasEdited) {
      setHasEdited(true)
      await aiApi.markEdited()
    }
  }

  const handleApply = async (movieId) => {
    const overrides = movieOverrides[movieId] || {}
    const baseProfile = genreProfile

    const genre_weights = {}
    const changed = []
    for (const [genre, level] of Object.entries(overrides)) {
      if (level === 'neutral') continue
      const lvl = LEVELS.find(l => l.value === level) || LEVELS[2]
      genre_weights[genre] = Math.max(0, Math.min(1, (baseProfile[genre] || 0) + lvl.delta))
      changed.push([genre, level])
    }

    setApplyingFor(movieId)
    try {
      const { data } = changed.length
        ? await aiApi.recommendFromProfile(genre_weights)
        : await aiApi.recommend(10)
      setRecs(data.recommendations)
      fireLogRecs(data.recommendations, round, 'edited')
      if (changed.length > 0) {
        await logEdits('movie', changed, movieId)
        setRoundEdited(true)
      }
      await markEditedOnce()
    } finally {
      setApplyingFor(null)
    }
  }

  const handleApplyProfile = async () => {
    const baseProfile = genreProfile
    const genre_weights = {}
    const changed = []
    for (const [genre, level] of Object.entries(profileOverrides)) {
      if (level === 'neutral') continue
      const lvl = LEVELS.find(l => l.value === level) || LEVELS[2]
      genre_weights[genre] = Math.max(0, Math.min(1, (baseProfile[genre] || 0) + lvl.delta))
      changed.push([genre, level])
    }

    setApplyingProfile(true)
    try {
      const { data } = changed.length
        ? await aiApi.recommendFromProfile(genre_weights)
        : await aiApi.recommend(10)
      setRecs(data.recommendations)
      fireLogRecs(data.recommendations, round, 'edited')
      if (changed.length > 0) {
        await logEdits('profile', changed)
        setRoundEdited(true)
      }
      await markEditedOnce()
    } finally {
      setApplyingProfile(false)
    }
  }

  // Gate: every currently displayed recommendation must be rated.
  // After Apply refreshes the list, any newly shown movies reset this gate —
  // intentional: the user should react to what they actually see each time.
  const hasRatedAll = recs.length > 0 && recs.every(r => recRatings[r.movie_id] > 0)
  const ratedCount  = recs.filter(r => recRatings[r.movie_id] > 0).length

  // O: need all rated + at least one edit applied  N: just need all rated
  const canAdvance = isTransparent ? (roundEdited && hasRatedAll) : hasRatedAll

  const goNextRound = () => {
    // Log round-2 starting recs (carried over from round-1 Apply)
    fireLogRecs(recs, 2, 'initial')
    setRound(2)
    setRoundEdited(false)
    setRecRatings({})
    setMovieOverrides({})
    setProfileOverrides({})
    setExpandedId(null)
    setExplainCache({})
  }

  const handlePrimaryAction = () => {
    if (isTransparent && round === 1) {
      goNextRound()
    } else {
      navigate('/sus')
    }
  }

  const unratedCount = recs.length - ratedCount
  const taskText = isTransparent
    ? editType === 'movie'
      ? `Rate all ${recs.length} recommendations below. Then click "Edit Preferences" on at least one, ` +
        'adjust the genre weights, and click Apply & Refresh.'
      : `Rate all ${recs.length} recommendations below. Then use the profile panel above to ` +
        'adjust your genre weights and click Apply & Refresh.'
    : `Please rate all ${recs.length} recommendations below before continuing to the survey.`

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Your Recommendations</h1>
          <div className="version-header-row">
            <span className={`version-badge ${info.badgeClass}`}>{info.label}</span>
            {isTransparent && <span className="round-badge">Round {round} of 2</span>}
          </div>
        </div>
        <div className="page-header-actions">
          <button
            className="btn-primary"
            onClick={handlePrimaryAction}
            disabled={!canAdvance}
            title={
              !canAdvance
                ? unratedCount > 0
                  ? `Rate ${unratedCount} more recommendation${unratedCount > 1 ? 's' : ''} first`
                  : 'Apply at least one genre weight change first'
                : undefined
            }
          >
            {isTransparent && round === 1 ? 'Next Round →' : 'Continue to Survey →'}
          </button>
        </div>
      </div>

      <div className="version-guidance-banner">
        <p>{info.description}</p>
      </div>

      {!canAdvance && (
        <div className="info-banner">
          <strong>Task:</strong> {taskText}
          {unratedCount > 0 && (
            <> — <strong>{ratedCount}/{recs.length} rated</strong></>
          )}
        </div>
      )}

      {loading ? (
        <div className="loading">Loading recommendations…</div>
      ) : recs.length === 0 ? (
        <div className="empty-state">No recommendations yet — rate some movies first.</div>
      ) : (
        <>
          {isTransparent && editType === 'profile' && (
            <ProfileEditPanel
              genreProfile={genreProfile}
              overrides={profileOverrides}
              applying={applyingProfile}
              onOverrideChange={handleProfileOverrideChange}
              onApply={handleApplyProfile}
            />
          )}

          <div className="rec-list-full">
            {recs.map((r, i) => (
              <RecCard
                key={r.movie_id}
                rank={i + 1}
                rec={r}
                expanded={expandedId === r.movie_id}
                explainData={explainCache[r.movie_id]}
                genreProfile={genreProfile}
                overrides={movieOverrides[r.movie_id] || {}}
                applying={applyingFor === r.movie_id}
                showEdit={isTransparent && editType === 'movie'}
                rating={recRatings[r.movie_id]}
                onRate={star => handleRateRec(r.movie_id, star)}
                onToggle={() => handleToggle(r.movie_id)}
                onOverrideChange={(genre, level) => handleOverrideChange(r.movie_id, genre, level)}
                onApply={() => handleApply(r.movie_id)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
