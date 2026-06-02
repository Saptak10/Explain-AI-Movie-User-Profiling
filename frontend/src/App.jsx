import { Navigate, Route, Routes } from 'react-router-dom'
import Navbar from './components/Navbar'
import { useAuth } from './context/AuthContext'
import LoginPage from './pages/LoginPage'
import ProfilePage from './pages/ProfilePage'
import RatingsPage from './pages/RatingsPage'
import RecommendPage from './pages/RecommendPage'

function Guard({ children }) {
  const { user } = useAuth()
  return user ? children : <Navigate to="/login" replace />
}

export default function App() {
  const { user } = useAuth()
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/login"     element={user ? <Navigate to="/rate" replace /> : <LoginPage />} />
        <Route path="/rate"      element={<Guard><RatingsPage /></Guard>} />
        <Route path="/profile"   element={<Guard><ProfilePage /></Guard>} />
        <Route path="/recommend" element={<Guard><RecommendPage /></Guard>} />
        <Route path="*"          element={<Navigate to={user ? '/rate' : '/login'} replace />} />
      </Routes>
    </>
  )
}
