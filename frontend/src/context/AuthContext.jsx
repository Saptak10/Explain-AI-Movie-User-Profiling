/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState } from 'react'

const Ctx = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('user')) } catch { return null }
  })

  const login = (token, user_id, username, version, edit_order) => {
    const u = { id: user_id, username, version, edit_order }
    localStorage.setItem('token', token)
    localStorage.setItem('user', JSON.stringify(u))
    setUser(u)
  }

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
  }

  return (
    <Ctx.Provider value={{ user, login, logout }}>
      {children}
    </Ctx.Provider>
  )
}

export const useAuth = () => useContext(Ctx)
