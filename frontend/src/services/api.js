import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const client = axios.create({ baseURL: BASE })

client.interceptors.request.use(cfg => {
  const token = localStorage.getItem('token')
  if (token) cfg.headers.Authorization = `Bearer ${token}`
  return cfg
})

export const authApi = {
  register: (username, password) => client.post('/api/auth/register', { username, password }),
  login:    (username, password) => client.post('/api/auth/login',    { username, password }),
}

export const moviesApi = {
  popular: () => client.get('/api/movies/popular'),
}

export const aiApi = {
  submitRating:         (movie_id, rating)                    => client.post('/api/ratings', { movie_id, rating }),
  getRatings:           ()                                     => client.get('/api/ratings'),
  getProfile:           ()                                     => client.get('/api/profile'),
  recommend:            (top_n = 10, overrides = null, alpha = 3.0) =>
                          client.post('/api/recommend', { top_n, overrides, alpha }),
  recommendFromProfile: (profile, top_n = 10)                 =>
                          client.post('/api/recommend/edited-profile', { profile, top_n }),
  explain:              (movie_id, method = 'soft')            => client.post('/api/explain', { movie_id, method }),
  getImportance:        ()                                     => client.get('/api/importance'),
  markEdited:           ()                                     => client.post('/api/user/mark-edited'),
}

export const susApi = {
  getQuestions: ()          => client.get('/api/sus/questions'),
  submit:       (responses) => client.post('/api/sus/submit', { responses }),
  getResults:   ()          => client.get('/api/sus/results'),
}
