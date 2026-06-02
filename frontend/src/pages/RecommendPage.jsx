import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { aiApi } from '../services/api'

const GENRES = [
  "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

const LEVELS = [
  { label: '−−', value: 'stark dämpfen',      title: 'Strong suppress' },
  { label: '−',  value: 'leicht dämpfen',     title: 'Slight suppress' },
  { label: '○',  value: 'neutral',             title: 'Neutral' },
  { label: '+',  value: 'leicht verstärken',  title: 'Slight boost' },
  { label: '++', value: 'stark verstärken',   title: 'Strong boost' },
]

function ScoreBar({ score }) {
  return (
    <div className="score-track">
      <div className="score-fill" style={{ width: `${(score / 5) * 100}%` }} />
      <span className="score-label">{score.toFixed(2)}</span>
    </div>
  )
}

function ExplainModal({ data, onClose }) {
  const top = (data?.contributions || []).slice(0, 8)
  const maxVal = Math.max(...top.map(c => Math.abs(c.value)), 0.001)

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        <h2>Why was this recommended?</h2>
        {data?.text && <p className="explain-text">{data.text}</p>}
        {top.length > 0 && (
          <div className="explain-bars">
            {top.map(c => (
              <div key={c.genre} className="explain-row">
                <span className="explain-genre">{c.genre}</span>
                <div className="explain-track">
                  <div
                    className={`explain-fill${c.value >= 0 ? ' pos' : ' neg'}`}
                    style={{ width: `${(Math.abs(c.value) / maxVal) * 100}%` }}
                  />
                </div>
                <span className="explain-val">
                  {c.value >= 0 ? '+' : ''}{c.value.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}
        <p className="explain-note">
          Each bar shows how much that genre contributed to this recommendation score.
        </p>
      </div>
    </div>
  )
}

export default function RecommendPage() {
  const location = useLocation()
  const [recs, setRecs] = useState(location.state?.recommendations || [])
  const [overrides, setOverrides] = useState({})
  const [loading, setLoading] = useState(!location.state?.recommendations)
  const [refreshing, setRefreshing] = useState(false)
  const [explainData, setExplainData] = useState(null)
  const [explainLoading, setExplainLoading] = useState(null)

  useEffect(() => {
    if (!location.state?.recommendations) loadRecs()
  }, [])

  const activeOverrides = () =>
    Object.fromEntries(Object.entries(overrides).filter(([, v]) => v && v !== 'neutral'))

  const loadRecs = async (ovr) => {
    setLoading(true)
    try {
      const active = ovr !== undefined ? ovr : activeOverrides()
      const { data } = await aiApi.recommend(10, Object.keys(active).length ? active : null)
      setRecs(data.recommendations)
    } finally {
      setLoading(false)
    }
  }

  const applyOverrides = async () => {
    setRefreshing(true)
    try {
      const active = activeOverrides()
      const { data } = await aiApi.recommend(10, Object.keys(active).length ? active : null)
      setRecs(data.recommendations)
    } finally {
      setRefreshing(false)
    }
  }

  const resetOverrides = () => {
    setOverrides({})
    loadRecs({})
  }

  const handleExplain = async movieId => {
    setExplainLoading(movieId)
    try {
      const { data } = await aiApi.explain(movieId, 'soft')
      setExplainData(data)
    } finally {
      setExplainLoading(null)
    }
  }

  const activeCount = Object.values(overrides).filter(v => v && v !== 'neutral').length

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Recommendations</h1>
          <p className="text-muted">
            {location.state?.fromEdited
              ? 'Based on your manually edited profile'
              : 'Based on your AI-inferred profile'}
            {activeCount > 0 && ` · ${activeCount} genre override${activeCount > 1 ? 's' : ''} active`}
          </p>
        </div>
      </div>

      <div className="recommend-layout">
        {/* ── Movie list ── */}
        <div className="rec-list">
          {loading ? (
            <div className="loading">Loading recommendations…</div>
          ) : recs.length === 0 ? (
            <div className="empty-state">
              No recommendations yet — rate some movies first.
            </div>
          ) : (
            recs.map((r, i) => (
              <div key={r.movie_id} className="rec-card">
                <div className="rec-rank">#{i + 1}</div>
                <div className="rec-info">
                  <div className="rec-title">{r.title}</div>
                  <ScoreBar score={r.score} />
                </div>
                <button
                  className="btn-explain"
                  onClick={() => handleExplain(r.movie_id)}
                  disabled={explainLoading === r.movie_id}
                >
                  {explainLoading === r.movie_id ? '…' : 'Why?'}
                </button>
              </div>
            ))
          )}
        </div>

        {/* ── Override panel ── */}
        <div className="override-panel">
          <div className="panel-header">
            <h3>Genre Overrides</h3>
            {activeCount > 0 && <span className="badge">{activeCount} active</span>}
          </div>
          <p className="text-muted panel-desc">
            Boost or suppress genres to see how your recommendations change.
          </p>

          <div className="override-list">
            {GENRES.map(genre => {
              const cur = overrides[genre] || 'neutral'
              return (
                <div key={genre} className="override-row">
                  <span className="override-genre">{genre}</span>
                  <div className="override-btns">
                    {LEVELS.map(lvl => (
                      <button
                        key={lvl.value}
                        className={`override-btn${cur === lvl.value ? ' active' : ''}`}
                        onClick={() => setOverrides(prev => ({ ...prev, [genre]: lvl.value }))}
                        title={lvl.title}
                      >
                        {lvl.label}
                      </button>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>

          <div className="panel-actions">
            <button
              className="btn-primary btn-full"
              onClick={applyOverrides}
              disabled={refreshing}
            >
              {refreshing ? 'Updating…' : 'Apply Overrides'}
            </button>
            {activeCount > 0 && (
              <button className="btn-ghost btn-full" onClick={resetOverrides}>
                Reset All
              </button>
            )}
          </div>
        </div>
      </div>

      {explainData && (
        <ExplainModal data={explainData} onClose={() => setExplainData(null)} />
      )}
    </div>
  )
}
