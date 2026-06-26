import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { aiApi } from '../services/api'
import { useAuth } from '../context/AuthContext'

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
      'and adjust the genre weights that drive your recommendations. ' +
      "Please edit at least one movie's preferences before continuing to the survey.",
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
          <p className="edit-panel-label">Adjust genre preferences:</p>
          <div className="edit-genre-list">
            {topGenres.map(([genre]) => (
              <div key={genre} className="edit-genre-row">
                <div className="edit-genre-info">
                  <span className="edit-genre-name">{genre}</span>
                </div>
                <div className="override-btns">
                  {LEVELS.map(lvl => (
                    <button
                      key={lvl.value}
                      className={`override-btn${(overrides[genre] || 'neutral') === lvl.value ? ' active' : ''}`}
                      onClick={() => onOverrideChange(genre, lvl.value)}
                      title={lvl.value}
                    >
                      {lvl.label}
                    </button>
                  ))}
                </div>
              </div>
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

function RecCard({ rank, rec, expanded, explainData, genreProfile, overrides, applying,
                   onToggle, onOverrideChange, onApply, showEdit }) {
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

  const [recs, setRecs]                     = useState(location.state?.recommendations || [])
  const [loading, setLoading]               = useState(!location.state?.recommendations)
  const [expandedId, setExpandedId]         = useState(null)
  const [explainCache, setExplainCache]     = useState({})
  const [movieOverrides, setMovieOverrides] = useState({})
  const [applyingFor, setApplyingFor]       = useState(null)
  const [hasEdited, setHasEdited]           = useState(false)

  // Genre profile kept in a ref — used by Apply to build genre_weights
  const genreProfileRef = useRef({})

  useEffect(() => {
    if (!location.state?.recommendations) loadRecs()
    // Load genre profile once for the override Apply logic
    aiApi.getProfile().then(({ data }) => { genreProfileRef.current = data.profile })
  }, [])

  const loadRecs = async () => {
    setLoading(true)
    try {
      const { data } = await aiApi.recommend(10)
      setRecs(data.recommendations)
    } finally {
      setLoading(false)
    }
  }

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

  const handleApply = async (movieId) => {
    const overrides = movieOverrides[movieId] || {}
    const baseProfile = genreProfileRef.current

    // Convert LEVELS button selections → float genre_weights
    const genre_weights = {}
    for (const [genre, level] of Object.entries(overrides)) {
      const lvl = LEVELS.find(l => l.value === level) || LEVELS[2]
      genre_weights[genre] = Math.max(0, Math.min(1, (baseProfile[genre] || 0) + lvl.delta))
    }

    setApplyingFor(movieId)
    try {
      const { data } = Object.keys(genre_weights).length
        ? await aiApi.recommendFromProfile(genre_weights)
        : await aiApi.recommend(10)
      setRecs(data.recommendations)
      if (!hasEdited) {
        setHasEdited(true)
        await aiApi.markEdited()
      }
    } finally {
      setApplyingFor(null)
    }
  }

  const canContinue = isTransparent ? hasEdited : true

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Your Recommendations</h1>
          <div className="version-header-row">
            <span className={`version-badge ${info.badgeClass}`}>{info.label}</span>
          </div>
        </div>
        <div className="page-header-actions">
          <button
            className="btn-primary"
            onClick={() => navigate('/sus')}
            disabled={!canContinue}
            title={!canContinue ? "Edit at least one movie's preferences first" : undefined}
          >
            Continue to Survey →
          </button>
        </div>
      </div>

      <div className="version-guidance-banner">
        <p>{info.description}</p>
      </div>

      {isTransparent && !hasEdited && (
        <div className="info-banner">
          <strong>Task:</strong> For at least one recommendation below, click{' '}
          <em>Edit Preferences</em> to see why it was recommended and adjust the genre weights.
          Then click <em>Apply &amp; Refresh</em> before continuing to the survey.
        </div>
      )}

      {loading ? (
        <div className="loading">Loading recommendations…</div>
      ) : recs.length === 0 ? (
        <div className="empty-state">No recommendations yet — rate some movies first.</div>
      ) : (
        <div className="rec-list-full">
          {recs.map((r, i) => (
            <RecCard
              key={r.movie_id}
              rank={i + 1}
              rec={r}
              expanded={expandedId === r.movie_id}
              explainData={explainCache[r.movie_id]}
              genreProfile={genreProfileRef.current}
              overrides={movieOverrides[r.movie_id] || {}}
              applying={applyingFor === r.movie_id}
              showEdit={isTransparent}
              onToggle={() => handleToggle(r.movie_id)}
              onOverrideChange={(genre, level) => handleOverrideChange(r.movie_id, genre, level)}
              onApply={() => handleApply(r.movie_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
