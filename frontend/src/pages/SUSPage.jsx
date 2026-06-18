import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { susApi } from '../services/api'

const AGE_OPTIONS = ['18-23', '24-30', '30-45', '>45']

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

  const [ageGroup, setAgeGroup]               = useState('')
  const [degreeJob, setDegreeJob]             = useState('')
  const [netflixExp, setNetflixExp]           = useState(0)
  const [responses, setResponses]             = useState(Array(10).fill(0))
  const [submitting, setSubmitting]           = useState(false)
  const [done, setDone]                       = useState(false)

  const answeredCount  = responses.filter(r => r > 0).length
  const demoComplete   = ageGroup !== '' && degreeJob.trim() !== '' && netflixExp > 0
  const allAnswered    = answeredCount === 10 && demoComplete

  const setResponse = (idx, val) =>
    setResponses(prev => { const next = [...prev]; next[idx] = val; return next })

  const handleSubmit = async () => {
    if (!allAnswered) return
    setSubmitting(true)
    try {
      await susApi.submit(responses, ageGroup, degreeJob.trim(), netflixExp)
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
            Please answer the demographic questions and then rate your experience.
          </p>
        </div>
      </div>

      <div className="sus-card">

        {/* ── Demographic section ─────────────────────────────────────────── */}
        <div className="demo-section">
          <p className="demo-section-title">Background Information</p>

          {/* Q1 — Age */}
          <div className="demo-question">
            <p className="demo-q-text">
              How old are you? <span className="required-star">*</span>
            </p>
            <div className="demo-radio-group">
              {AGE_OPTIONS.map(opt => (
                <label key={opt} className={`demo-radio-label${ageGroup === opt ? ' selected' : ''}`}>
                  <input
                    type="radio"
                    name="age_group"
                    value={opt}
                    checked={ageGroup === opt}
                    onChange={() => setAgeGroup(opt)}
                  />
                  <span className={`demo-radio-btn${ageGroup === opt ? ' checked' : ''}`} />
                  {opt}
                </label>
              ))}
            </div>
          </div>

          {/* Q2 — Degree / Job */}
          <div className="demo-question">
            <p className="demo-q-text">
              Which degree are you currently taking or what is your job title?{' '}
              <span className="required-star">*</span>
            </p>
            <input
              className="demo-text-input"
              type="text"
              placeholder="e.g. M.Sc. Computer Science / Software Engineer"
              value={degreeJob}
              onChange={e => setDegreeJob(e.target.value)}
            />
          </div>

          {/* Q3 — Netflix experience */}
          <div className="demo-question">
            <p className="demo-q-text">
              I have experience with movie recommendation platforms such as Netflix.{' '}
              <span className="required-star">*</span>
            </p>
            <div className="sus-scale demo-likert">
              <span className="sus-scale-label">Strongly Disagree</span>
              <div className="sus-options">
                {[1, 2, 3, 4, 5].map(v => (
                  <label key={v} className="sus-option">
                    <input
                      type="radio"
                      name="netflix_exp"
                      value={v}
                      checked={netflixExp === v}
                      onChange={() => setNetflixExp(v)}
                    />
                    <span className={`sus-radio${netflixExp === v ? ' checked' : ''}`}>{v}</span>
                  </label>
                ))}
              </div>
              <span className="sus-scale-label">Strongly Agree</span>
            </div>
          </div>
        </div>

        <div className="demo-divider" />

        {/* ── SUS questions ───────────────────────────────────────────────── */}
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
          <span className="text-muted">
            {demoComplete ? `${answeredCount} / 10 questions answered` : 'Complete background info above first'}
          </span>
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
