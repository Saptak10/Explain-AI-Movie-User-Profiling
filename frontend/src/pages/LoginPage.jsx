import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { authApi } from '../services/api'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
  const [tab, setTab] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  const submit = async e => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const fn = tab === 'login' ? authApi.login : authApi.register
      const { data } = await fn(username, password)
      login(data.token, data.user_id, data.username, data.version)
      navigate('/rate')
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">🎬</div>
          <h1 className="auth-title">CineProfile</h1>
          <p className="auth-subtitle">
            AI-powered movie recommendations with transparent, editable explanations
          </p>
        </div>

        <div className="tab-group">
          <button
            className={`tab${tab === 'login' ? ' active' : ''}`}
            onClick={() => { setTab('login'); setError('') }}
          >
            Sign In
          </button>
          <button
            className={`tab${tab === 'register' ? ' active' : ''}`}
            onClick={() => { setTab('register'); setError('') }}
          >
            Register
          </button>
        </div>

        <form onSubmit={submit} className="auth-form">
          <div className="field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="your_username"
              autoComplete="username"
              required
            />
          </div>
          <div className="field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
              required
            />
          </div>
          {error && <p className="error-msg">{error}</p>}
          <button type="submit" className="btn-primary btn-full" disabled={loading}>
            {loading ? 'Please wait…' : tab === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  )
}
