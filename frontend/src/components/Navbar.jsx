import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const NAV_LINKS = [
  { to: '/rate',      label: 'Rate Movies' },
  { to: '/profile',   label: 'My Profile' },
  { to: '/recommend', label: 'Recommendations' },
]

export default function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const { pathname } = useLocation()

  const handleLogout = () => { logout(); navigate('/login') }

  return (
    <nav className="navbar">
      <Link to="/" className="nav-brand">🎬 CineProfile</Link>

      {user && (
        <>
          <div className="nav-links">
            {NAV_LINKS.map(l => (
              <Link
                key={l.to}
                to={l.to}
                className={`nav-link${pathname === l.to ? ' active' : ''}`}
              >
                {l.label}
              </Link>
            ))}
          </div>
          <div className="nav-user">
            <span className="nav-username">{user.username}</span>
            <button className="btn-ghost sm" onClick={handleLogout}>Logout</button>
          </div>
        </>
      )}
    </nav>
  )
}
