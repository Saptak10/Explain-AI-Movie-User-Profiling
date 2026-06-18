import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { aiApi } from '../services/api'
import { useAuth } from '../context/AuthContext'

const LEVELS = [
  { label: '−−', value: 'stark dämpfen',     title: 'Strong suppress' },
  { label: '−',  value: 'leicht dämpfen',    title: 'Slight suppress' },
  { label: '○',  value: 'neutral',            title: 'Neutral' },
  { label: '+',  value: 'leicht verstärken', title: 'Slight boost' },
  { label: '++', value: 'stark verstärken',  title: 'Strong boost' },
]

const VERSION_INFO = {
  O: {
    label: 'Version O — Transparent AI',
    badgeClass: 'version-o',
    description:
      'You are in the Transparent AI condition. You can see why each movie was recommended ' +
      'and adjust the genre weights that drive your recommendations. ' +
      'Please edit at least one movie\'s preferences before continuing to the survey.',
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

function EditPanel({ explainData, overrides, applying, onOverrideChange, onApply }) {
  if (!explainData) return <div className="loading-sm">Analysing recommendation…</div>

  const top = (explainData.contributions || []).slice(0, 5)
  const maxVal = Math.max(...top.map(c => Math.abs(c.value)), 0.001)

  return (
    <div className="edit-panel">
      {explainData.text && (
        <p className="explain-text-inline">{explainData.text}</p>
      )}
      <p className="edit-panel-label">
        Adjust how important these genre factors are for your recommendations:
      </p>
      <div className="edit-genre-list">
        {top.map(c => (
          <div key={c.genre} className="edit-genre-row">
            <div className="edit-genre-info">
              <span className="edit-genre-name">{c.genre}</span>
              <div className="mini-bar-track">
                <div
                  className="mini-bar-fill"
                  style={{ width: `${(Math.abs(c.value) / maxVal) * 100}%` }}
                />
              </div>
            </div>
            <div className="override-btns">
              {LEVELS.map(lvl => (
                <button
                  key={lvl.value}
                  className={`override-btn${(overrides[c.genre] || 'neutral') === lvl.value ? ' active' : ''}`}
                  onClick={() => onOverrideChange(c.genre, lvl.value)}
                  title={lvl.title}
                >
                  {lvl.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <button className="btn-primary" onClick={onApply} disabled={applying}>
        {applying ? 'Updating recommendations…' : 'Apply & Refresh Recommendations'}
      </button>
    </div>
  )
}

function RecCard({ rank, rec, expanded, explainData, overrides, applying,
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
  const version = user?.version || 'O'
  const isTransparent = version === 'O'
  const info = VERSION_INFO[version] || VERSION_INFO['O']

  const [recs, setRecs]                     = useState(location.state?.recommendations || [])
  const [loading, setLoading]               = useState(!location.state?.recommendations)
  const [expandedId, setExpandedId]         = useState(null)
  const [explainCache, setExplainCache]     = useState({})
  const [movieOverrides, setMovieOverrides] = useState({})
  const [applyingFor, setApplyingFor]       = useState(null)
  const [hasEdited, setHasEdited]           = useState(false)

  useEffect(() => {
    if (!location.state?.recommendations) loadRecs()
  }, [])

  const loadRecs = async (overrides = null) => {
    setLoading(true)
    try {
      const { data } = await aiApi.recommend(10, overrides)
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
        const { data } = await aiApi.explain(movieId, 'soft')
        setExplainCache(prev => ({ ...prev, [movieId]: data }))
      } catch {
        setExplainCache(prev => ({ ...prev, [movieId]: { contributions: [], text: null } }))
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
    const raw = movieOverrides[movieId] || {}
    const active = Object.fromEntries(Object.entries(raw).filter(([, v]) => v && v !== 'neutral'))
    setApplyingFor(movieId)
    try {
      const { data } = await aiApi.recommend(10, Object.keys(active).length ? active : null)
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
            title={!canContinue ? 'Edit at least one movie\'s preferences first' : undefined}
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
