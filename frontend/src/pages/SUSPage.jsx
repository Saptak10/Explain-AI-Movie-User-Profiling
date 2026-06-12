import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { susApi } from '../services/api'

const SUS_QUESTIONS = [
  "I think that I would like to use this system frequently.",
  "I found the system unnecessarily complex.",
  "I thought the system was easy to use.",
  "I think that I would need the support of a technical person to use this system.",
  "I found the various functions in this system were well integrated.",
  "I thought there was too much inconsistency in this system.",
  "I would imagine that most people would learn to use this system very quickly.",
  "I found the system very cumbersome to use.",
  "I felt very confident using the system.",
  "I needed to learn a lot of things before I could get going with this system.",
]

export default function SUSPage() {
  const navigate = useNavigate()

  const [responses, setResponses] = useState(Array(10).fill(0))
  const [submitting, setSubmitting] = useState(false)
  const [done, setDone] = useState(false)

  const answeredCount = responses.filter(r => r > 0).length
  const allAnswered   = answeredCount === 10

  const setResponse = (idx, val) =>
    setResponses(prev => { const next = [...prev]; next[idx] = val; return next })

  const handleSubmit = async () => {
    if (!allAnswered) return
    setSubmitting(true)
    try {
      await susApi.submit(responses)
      setDone(true)
    } finally {
      setSubmitting(false)
    }
  }

  // ── Post-submission screen ───────────────────────────────────────────────
  if (done) {
    return (
      <div className="page">
        <div className="sus-complete">
          <div className="sus-complete-icon">🎉</div>
          <h1>Thank You!</h1>
          <p className="text-muted">
            Your responses have been recorded. You can go back to the recommendations
            or rate more movies to refine your profile.
          </p>
          <button className="btn-primary" onClick={() => navigate('/recommend')}>
            Back to Recommendations
          </button>
        </div>
      </div>
    )
  }

  // ── Questionnaire ────────────────────────────────────────────────────────
  return (
    <div className="page sus-page">
      <div className="page-header">
        <div>
          <h1>System Usability Survey</h1>
          <p className="text-muted">
            Please rate your experience with the recommendation system.
          </p>
        </div>
      </div>

      <div className="sus-card">
        <p className="sus-instruction">
          For each statement below, select how strongly you agree or disagree
          (1 = Strongly Disagree, 5 = Strongly Agree).
        </p>

        <div className="sus-questions">
          {SUS_QUESTIONS.map((q, i) => (
            <div key={i} className={`sus-question${responses[i] > 0 ? ' answered' : ''}`}>
              <div className="sus-q-num">{i + 1}</div>
              <div className="sus-q-body">
                <p className="sus-q-text">{q}</p>
                <div className="sus-scale">
                  <span className="sus-scale-label">Strongly Disagree</span>
                  <div className="sus-options">
                    {[1, 2, 3, 4, 5].map(v => (
                      <label key={v} className="sus-option">
                        <input
                          type="radio"
                          name={`q${i}`}
                          value={v}
                          checked={responses[i] === v}
                          onChange={() => setResponse(i, v)}
                        />
                        <span className={`sus-radio${responses[i] === v ? ' checked' : ''}`}>
                          {v}
                        </span>
                      </label>
                    ))}
                  </div>
                  <span className="sus-scale-label">Strongly Agree</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="sus-footer">
          <span className="text-muted">{answeredCount} / 10 answered</span>
          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={!allAnswered || submitting}
          >
            {submitting ? 'Submitting…' : 'Submit Survey'}
          </button>
        </div>
      </div>
    </div>
  )
}
