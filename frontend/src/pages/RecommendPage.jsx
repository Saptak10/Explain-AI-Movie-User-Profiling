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

      {error && <p className="error-msg">{error}</p>}

      <button className="btn-primary" onClick={onApply} disabled={applying}>
        {applying ? 'Updating recommendations…' : 'Apply & Refresh Recommendations'}
      </button>
    </div>
  )
}

function ProfileEditPanel({ genreProfile, overrides, applying, error, onOverrideChange, onApply }) {
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
      {error && <p className="error-msg">{error}</p>}
      <button className="btn-primary" onClick={onApply} disabled={applying}>
        {applying ? 'Updating recommendations…' : 'Apply & Refresh Recommendations'}
      </button>
    </div>
  )
}

function RecCard({ rank, rec, expanded, explainData, genreProfile, overrides, applying, applyError,
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
// shared per-card state (expandedId/explainCache/movieOverrides/recRatings
// are all keyed by movie_id, not list position), so editing or rating a
// card works identically in either section.
function RecSection({ icon, title, subtitle, recs, expandedId, explainCache, genreProfile, movieOverrides,
                      applyingFor, applyError, showEdit, recRatings, onToggle, onOverrideChange, onApply, onRate }) {
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
            rating={recRatings[r.movie_id]}
            onRate={star => onRate(r.movie_id, star)}
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

  // Counterbalanced edit order assigned at registration (Version O only).
  // 'movie_first' (default) -> round 1 edits per-movie weights, round 2 edits the whole profile.
  // 'profile_first'         -> round 1 edits the whole profile, round 2 edits per-movie weights.
  const editOrder  = user?.edit_order || 'movie_first'
  const firstType  = editOrder === 'profile_first' ? 'profile' : 'movie'
  const secondType = firstType === 'movie' ? 'profile' : 'movie'
  const editType   = round => (round === 1 ? firstType : secondType)

  const [round, setRound]                       = useState(location.state?.round || 1)
  const [topRated, setTopRated]                 = useState(location.state?.topRated || [])
  const [forYou, setForYou]                     = useState(location.state?.forYou || [])
  const [loading, setLoading]                   = useState(!location.state?.topRated && !location.state?.forYou)
  const [expandedId, setExpandedId]             = useState(null)
  const [explainCache, setExplainCache]         = useState({})
  const [movieOverrides, setMovieOverrides]     = useState({})
  const [profileOverrides, setProfileOverrides] = useState({})
  const [applyingFor, setApplyingFor]           = useState(null)
  const [applyingProfile, setApplyingProfile]   = useState(false)
  const [applyError, setApplyError]             = useState('')
  const [hasEdited, setHasEdited]               = useState(false) // ever edited (drives /user/mark-edited)
  const [roundEdited, setRoundEdited]           = useState(false) // edited during the current round
  const [recRatings, setRecRatings]             = useState({})
  const [genreProfile, setGenreProfile]         = useState({})
  const [profileLoaded, setProfileLoaded]       = useState(false)

  const currentEditType = editType(round)
  const allRecs = [...topRated, ...forYou]

  const fireLogRecs = (topRatedList, forYouList, currentRound, recType) => {
    const tagged = [
      ...topRatedList.map((r, i) => ({ movie_id: r.movie_id, position: i + 1, score: r.score, section: 'top_rated' })),
      ...forYouList.map((r, i) => ({ movie_id: r.movie_id, position: i + 1, score: r.score, section: 'for_you' })),
    ]
    aiApi.logRecommendations(
      currentRound,
      recType,
      tagged.map(m => ({ movie_id: m.movie_id, position: m.position, score: m.score }))
    ).catch(() => {})
  }

  const loadRecs = async () => {
    setLoading(true)
    try {
      const { data } = await aiApi.recommend(10)
      setTopRated(data.top_rated)
      setForYou(data.for_you)
      fireLogRecs(data.top_rated, data.for_you, round, 'initial')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!location.state?.topRated && !location.state?.forYou) loadRecs()
    aiApi.getProfile().then(({ data }) => {
      setGenreProfile(data.profile)
      setProfileLoaded(true)
    })
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

  // Shared by both per-movie and whole-profile Apply: builds the genre_deltas
  // payload the backend expects (see ai_service._merge_overrides_into_latent
  // -- deltas are added onto the AI-inferred profile server-side and
  // persisted, not absolute weights computed client-side), while separately
  // tracking which genres were actually changed away from neutral, since
  // that's what counts as a real "edit" for the round-gating/logging logic.
  const buildDeltas = (overrides) => {
    const genre_deltas = {}
    const changed = []
    for (const [genre, level] of Object.entries(overrides)) {
      const lvl = LEVELS.find(l => l.value === level) || LEVELS[2]
      genre_deltas[genre] = lvl.delta
      if (level !== 'neutral') changed.push([genre, level])
    }
    return { genre_deltas, changed }
  }

  const handleApply = async (movieId) => {
    const overrides = movieOverrides[movieId] || {}

    if (!profileLoaded) {
      setApplyError('Still loading your profile — try again in a moment.')
      return
    }

    const { genre_deltas, changed } = buildDeltas(overrides)

    setApplyingFor(movieId)
    setApplyError('')
    try {
      const { data } = Object.keys(genre_deltas).length
        ? await aiApi.recommendFromProfile(genre_deltas)
        : await aiApi.recommend(10)
      setTopRated(data.top_rated)
      setForYou(data.for_you)
      fireLogRecs(data.top_rated, data.for_you, round, 'edited')
      if (changed.length > 0) {
        await logEdits('movie', changed, movieId)
        setRoundEdited(true)
      }
      await markEditedOnce()
    } catch (err) {
      setApplyError(err.response?.data?.detail || 'Failed to update recommendations')
    } finally {
      setApplyingFor(null)
    }
  }

  const handleApplyProfile = async () => {
    if (!profileLoaded) {
      setApplyError('Still loading your profile — try again in a moment.')
      return
    }

    const { genre_deltas, changed } = buildDeltas(profileOverrides)

    setApplyingProfile(true)
    setApplyError('')
    try {
      const { data } = Object.keys(genre_deltas).length
        ? await aiApi.recommendFromProfile(genre_deltas)
        : await aiApi.recommend(10)
      setTopRated(data.top_rated)
      setForYou(data.for_you)
      fireLogRecs(data.top_rated, data.for_you, round, 'edited')
      if (changed.length > 0) {
        await logEdits('profile', changed)
        setRoundEdited(true)
      }
      await markEditedOnce()
    } catch (err) {
      setApplyError(err.response?.data?.detail || 'Failed to update recommendations')
    } finally {
      setApplyingProfile(false)
    }
  }

  // Gate: every currently displayed recommendation (across both sections)
  // must be rated. After Apply refreshes the lists, any newly shown movies
  // reset this gate — intentional: the user should react to what they
  // actually see each time.
  const hasRatedAll = allRecs.length > 0 && allRecs.every(r => recRatings[r.movie_id] > 0)
  const ratedCount  = allRecs.filter(r => recRatings[r.movie_id] > 0).length

  // O: need all rated + at least one edit applied  N: just need all rated
  const canAdvance = isTransparent ? (roundEdited && hasRatedAll) : hasRatedAll

  const goNextRound = () => {
    // Log round-2 starting recs (carried over from round-1 Apply)
    fireLogRecs(topRated, forYou, 2, 'initial')
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

  const unratedCount = allRecs.length - ratedCount
  const taskText = isTransparent
    ? currentEditType === 'movie'
      ? `Rate all ${allRecs.length} recommendations below. Then click "Edit Preferences" on at least one, ` +
        'adjust the genre weights, and click Apply & Refresh.'
      : `Rate all ${allRecs.length} recommendations below. Then use the profile panel above to ` +
        'adjust your genre weights and click Apply & Refresh.'
    : `Please rate all ${allRecs.length} recommendations below before continuing to the survey.`

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Your Recommendations</h1>
          <div className="version-header-row">
            <span className={`version-badge ${info.badgeClass}`}>{info.label}</span>
            {isTransparent && <span className="round-badge">Round {round} of 2</span>}
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
            <> — <strong>{ratedCount}/{allRecs.length} rated</strong></>
          )}
        </div>
      )}

      {loading ? (
        <div className="loading">Loading recommendations…</div>
      ) : topRated.length === 0 && forYou.length === 0 ? (
        <div className="empty-state">No recommendations yet — rate some movies first.</div>
      ) : (
        <>
          {isTransparent && currentEditType === 'profile' && (
            <ProfileEditPanel
              genreProfile={genreProfile}
              overrides={profileOverrides}
              applying={applyingProfile}
              error={applyError}
              onOverrideChange={handleProfileOverrideChange}
              onApply={handleApplyProfile}
            />
          )}

          <RecSection
            icon="⭐"
            title="Your Highest-Rated Picks"
            subtitle="Movies we predict you'd rate the highest, period — including broadly popular titles most people enjoy."
            recs={topRated}
            expandedId={expandedId}
            explainCache={explainCache}
            genreProfile={genreProfile}
            movieOverrides={movieOverrides}
            applyingFor={applyingFor}
            applyError={applyError}
            showEdit={isTransparent && currentEditType === 'movie'}
            recRatings={recRatings}
            onToggle={handleToggle}
            onOverrideChange={handleOverrideChange}
            onApply={handleApply}
            onRate={handleRateRec}
          />
          <RecSection
            icon="🎯"
            title="Picked Just For You"
            subtitle="Matched to your specific taste, not just what's generally popular — these may score a bit lower but fit you better."
            recs={forYou}
            expandedId={expandedId}
            explainCache={explainCache}
            genreProfile={genreProfile}
            movieOverrides={movieOverrides}
            applyingFor={applyingFor}
            applyError={applyError}
            showEdit={isTransparent && currentEditType === 'movie'}
            recRatings={recRatings}
            onToggle={handleToggle}
            onOverrideChange={handleOverrideChange}
            onApply={handleApply}
            onRate={handleRateRec}
          />
        </>
      )}
    </div>
  )
}
