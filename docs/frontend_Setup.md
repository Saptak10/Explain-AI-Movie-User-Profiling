# Frontend Setup Guide

## 1. Create the React App (Vite)

```bash
# npm
npm create vite@latest frontend -- --template react
cd frontend
npm install

# yarn
yarn create vite frontend --template react
cd frontend
yarn

# pnpm
pnpm create vite frontend --template react
cd frontend
pnpm install
```

## 2. Start Development Server

```bash
npm run dev
```

Runs on: `http://localhost:5173`

---

## 3. Common Frontend Packages

```bash
# Routing
npm install react-router-dom

# API calls
npm install axios

# Tailwind CSS
npm install -D tailwindcss @tailwindcss/vite
```

**`vite.config.js`**

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```

**`src/index.css`**

```css
@import "tailwindcss";
```

---

## 4. UI Libraries (Optional)

```bash
# shadcn/ui
npx shadcn@latest init

# Material UI
npm install @mui/material @emotion/react @emotion/styled

# Bootstrap
npm install bootstrap
```

For Bootstrap, add to `main.jsx`:

```js
import 'bootstrap/dist/css/bootstrap.min.css'
```

---

## 5. State Management (Optional)

```bash
# Redux Toolkit
npm install @reduxjs/toolkit react-redux

# Zustand (lighter alternative)
npm install zustand
```

---

## 6. Forms & Validation

```bash
npm install react-hook-form zod
```

---

## 7. Authentication Helpers (Optional)

```bash
# JWT decode
npm install jwt-decode

# Firebase
npm install firebase
```

---

## 8. Charts & Visualization

```bash
npm install recharts
# or
npm install chart.js react-chartjs-2
```

---

## 9. AI / ML Frontend Packages

```bash
# Markdown rendering
npm install react-markdown

# Syntax highlighting
npm install react-syntax-highlighter

# OpenAI SDK
npm install openai

# LangChain
npm install langchain
```

---

## 10. Environment Variables

Create a `.env` file inside `frontend/`:

```bash
touch .env
```

Example:

```env
VITE_API_URL=http://localhost:8000
VITE_OPENAI_API_KEY=your_key
```

Access in React:

```js
import.meta.env.VITE_API_URL
```

> All Vite env vars must be prefixed with `VITE_` to be exposed to the client.

---

## 11. Recommended Folder Structure

```
src/
├── components/
├── pages/
├── layouts/
├── hooks/
├── services/
├── api/
├── context/
├── store/
├── utils/
├── assets/
├── styles/
└── App.jsx
```

---

## 12. ESLint + Prettier

```bash
npm install -D eslint prettier eslint-config-prettier eslint-plugin-react
```

---

## 13. Build & Preview

```bash
# Production build
npm run build

# Preview production build locally
npm run preview
```

---

## 14. Recommended Stack for This Project

Full install for a Django + React + GenAI stack:

```bash
npm install react-router-dom axios react-hook-form zod zustand \
  react-markdown react-syntax-highlighter recharts

# Optional AI packages
npm install openai langchain
```

### One-shot setup

```bash
npm create vite@latest frontend -- --template react && \
  cd frontend && \
  npm install && \
  npm install react-router-dom axios react-hook-form zod zustand \
    react-markdown react-syntax-highlighter recharts
```

---

## 15. Git

```bash
git init
git add .
git commit -m "Initial React frontend setup"
```
