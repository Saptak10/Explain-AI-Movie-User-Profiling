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
  popular: (excludeIds = []) =>
    client.get('/api/movies/popular', { params: { exclude: excludeIds.join(',') } }),
  search: (q) =>
    client.get('/api/movies/search', { params: { q } }),
}

export const aiApi = {
  submitRating:         (movie_id, rating)              => client.post('/api/ratings', { movie_id, rating }),
  getRatings:           ()                               => client.get('/api/ratings'),
  getProfile:           ()                               => client.get('/api/profile'),
  explainProfile:       ()                               => client.get('/api/profile/explain'),
  recommend:            (top_n = 10)                     => client.post('/api/recommend', { top_n }),
  recommendFromProfile: (genre_deltas, top_n = 10, source = 'profile_page') =>
                          client.post('/api/recommend/edited-profile', { genre_deltas, top_n, source }),
  getOverrides:         ()                               => client.get('/api/profile/overrides'),
  clearOverrides:       ()                               => client.delete('/api/profile/overrides'),
  personalizeProfile:   (top_n = 10)                     => client.post('/api/profile/personalize', { top_n }),
  explain:              (movie_id)                       => client.post('/api/explain', { movie_id }),
  getImportance:        ()                               => client.get('/api/importance'),
  markEdited:           ()                               => client.post('/api/user/mark-edited'),
  setCondition:         (version)                        => client.post('/api/user/set-condition', { version }),
}

export const susApi = {
  getQuestions: ()                                                    => client.get('/api/sus/questions'),
  submit:       (responses, age_group, degree_job, netflix_experience) =>
                  client.post('/api/sus/submit', { responses, age_group, degree_job, netflix_experience }),
  getResults:   ()                                                    => client.get('/api/sus/results'),
}
