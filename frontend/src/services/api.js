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
  submitRating:         (movie_id, rating, round = 0)    => client.post('/api/ratings', { movie_id, rating, round }),
  getRatings:           ()                               => client.get('/api/ratings'),
  getProfile:           ()                               => client.get('/api/profile'),
  recommend:            (top_n = 10)                     => client.post('/api/recommend', { top_n }),
  recommendFromProfile: (genre_weights, top_n = 10)      =>
                          client.post('/api/recommend/edited-profile', { genre_weights, top_n }),
  explain:              (movie_id)                       => client.post('/api/explain', { movie_id }),
  getImportance:        ()                               => client.get('/api/importance'),
  markEdited:           ()                               => client.post('/api/user/mark-edited'),
  logProfileEdit:       (round, edit_type, genre, level, movie_id = null) =>
                          client.post('/api/profile-edits', { round, edit_type, genre, level, movie_id }),
  logRecommendations:   (round, rec_type, movies) =>
                          client.post('/api/recommendation-log', { round, rec_type, movies }),
}

export const susApi = {
  getQuestions: ()                                                    => client.get('/api/sus/questions'),
  submit:       (responses, age_group, degree_job, netflix_experience) =>
                  client.post('/api/sus/submit', { responses, age_group, degree_job, netflix_experience }),
  getResults:   ()                                                    => client.get('/api/sus/results'),
}
