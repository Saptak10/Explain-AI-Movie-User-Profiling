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

function EditPanel({ explainData, genreProfile, overrides, applying, error, onOverrideChange, onApply }) {
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

      {error && <p className="error-msg">{error}</p>}

      <button className="btn-primary" onClick={onApply} disabled={applying}>
        {applying ? 'Updating recommendations…' : 'Apply & Refresh Recommendations'}
      </button>
    </div>
  )
}

function RecCard({ rank, rec, expanded, explainData, genreProfile, overrides, applying, applyError,
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
          error={applyError}
          onOverrideChange={onOverrideChange}
          onApply={onApply}
        />
      )}
    </div>
  )
}

// Two independent rankings over the same predictions (see
// AIService._rank_both): "top_rated" is the highest raw predicted score,
// period -- may include broadly popular titles. "for_you" is ranked by
// personalization lift (predicted score above what a typical viewer would
// get), so it favors movies specifically matched to this user's own
// ratings even when the raw score is a bit lower. Both use the same
// shared per-card state (expandedId/explainCache/movieOverrides are all
// keyed by movie_id, not list position), so editing a card works
// identically in either section.
function RecSection({ icon, title, subtitle, recs, expandedId, explainCache, genreProfile, movieOverrides,
                      applyingFor, applyError, showEdit, onToggle, onOverrideChange, onApply }) {
  if (recs.length === 0) return null
  return (
    <div className="rec-section">
      <div className="rec-section-header">
        <h2>{icon} {title}</h2>
        <p className="text-muted">{subtitle}</p>
      </div>
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
            applyError={expandedId === r.movie_id ? applyError : ''}
            showEdit={showEdit}
            onToggle={() => onToggle(r.movie_id)}
            onOverrideChange={(genre, level) => onOverrideChange(r.movie_id, genre, level)}
            onApply={() => onApply(r.movie_id)}
          />
        ))}
      </div>
    </div>
  )
}

// Dev-only override so both A/B study conditions can be previewed without
// re-registering accounts. Never touches the real user.version assigned
// at registration (backend/app/services/auth_service.py) — only affects
// which UI this browser session renders, and only in dev builds.
const DEV_PREVIEW_KEY = 'devPreviewVersion'

export default function RecommendPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const { user } = useAuth()
  const [devPreviewVersion, setDevPreviewVersion] = useState(() =>
    import.meta.env.DEV ? localStorage.getItem(DEV_PREVIEW_KEY) : null
  )
  const version     = devPreviewVersion || user?.version || 'O'
  const isTransparent = version === 'O'
  const info        = VERSION_INFO[version] || VERSION_INFO['O']

  const toggleDevPreview = () => {
    const next = version === 'O' ? 'N' : 'O'
    localStorage.setItem(DEV_PREVIEW_KEY, next)
    setDevPreviewVersion(next)
  }

  const [topRated, setTopRated]             = useState(location.state?.topRated || [])
  const [forYou, setForYou]                 = useState(location.state?.forYou || [])
  const [loading, setLoading]               = useState(!location.state?.topRated && !location.state?.forYou)
  const [expandedId, setExpandedId]         = useState(null)
  const [explainCache, setExplainCache]     = useState({})
  const [movieOverrides, setMovieOverrides] = useState({})
  const [applyingFor, setApplyingFor]       = useState(null)
  const [hasEdited, setHasEdited]           = useState(false)
  const [applyError, setApplyError]         = useState('')
  const [profileLoaded, setProfileLoaded]   = useState(false)

  // Genre profile kept in a ref — used by Apply to build genre_weights
  const genreProfileRef = useRef({})

  useEffect(() => {
    if (!location.state?.topRated && !location.state?.forYou) loadRecs()
    // Load genre profile once for the override Apply logic
    aiApi.getProfile().then(({ data }) => {
      genreProfileRef.current = data.profile
      setProfileLoaded(true)
    })
  }, [])

  const loadRecs = async () => {
    setLoading(true)
    try {
      const { data } = await aiApi.recommend(10)
      setTopRated(data.top_rated)
      setForYou(data.for_you)
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

    if (!profileLoaded) {
      setApplyError('Still loading your profile — try again in a moment.')
      return
    }

    // Only the genres touched in this card's panel need a delta — the
    // backend merges it onto the AI-inferred profile for every other
    // genre automatically (see ai_service.get_recommendations), and
    // persists it so it keeps applying on future visits too.
    const genre_deltas = {}
    for (const [genre, level] of Object.entries(overrides)) {
      const lvl = LEVELS.find(l => l.value === level) || LEVELS[2]
      genre_deltas[genre] = lvl.delta
    }

    setApplyingFor(movieId)
    setApplyError('')
    try {
      const { data } = Object.keys(genre_deltas).length
        ? await aiApi.recommendFromProfile(genre_deltas)
        : await aiApi.recommend(10)
      setTopRated(data.top_rated)
      setForYou(data.for_you)
      if (!hasEdited) {
        setHasEdited(true)
        await aiApi.markEdited()
      }
    } catch (err) {
      setApplyError(err.response?.data?.detail || 'Failed to update recommendations')
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
            {import.meta.env.DEV && (
              <button
                className="btn-secondary dev-preview-toggle"
                onClick={toggleDevPreview}
                title="Dev only: preview the other A/B condition without re-registering. Does not change your real assigned version."
              >
                🔧 Preview {isTransparent ? 'Standard (N)' : 'Transparent (O)'}
              </button>
            )}
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
      ) : topRated.length === 0 && forYou.length === 0 ? (
        <div className="empty-state">No recommendations yet — rate some movies first.</div>
      ) : (
        <>
          <RecSection
            icon="⭐"
            title="Your Highest-Rated Picks"
            subtitle="Movies we predict you'd rate the highest, period — including broadly popular titles most people enjoy."
            recs={topRated}
            expandedId={expandedId}
            explainCache={explainCache}
            genreProfile={genreProfileRef.current}
            movieOverrides={movieOverrides}
            applyingFor={applyingFor}
            applyError={applyError}
            showEdit={isTransparent}
            onToggle={handleToggle}
            onOverrideChange={handleOverrideChange}
            onApply={handleApply}
          />
          <RecSection
            icon="🎯"
            title="Picked Just For You"
            subtitle="Matched to your specific taste, not just what's generally popular — these may score a bit lower but fit you better."
            recs={forYou}
            expandedId={expandedId}
            explainCache={explainCache}
            genreProfile={genreProfileRef.current}
            movieOverrides={movieOverrides}
            applyingFor={applyingFor}
            applyError={applyError}
            showEdit={isTransparent}
            onToggle={handleToggle}
            onOverrideChange={handleOverrideChange}
            onApply={handleApply}
          />
        </>
      )}
    </div>
  )
}
