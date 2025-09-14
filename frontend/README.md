# **Overview**

Frontend for [F1-Betting](https://github.com/JadoreThompson/f1-betting) — a sleek, **React + TypeScript + Vite** client providing real‑time Formula 1 race odds, driver and constructor stats, bet placement, and results tracking. Built for speed and responsiveness, this frontend seamlessly integrates with the backend API to deliver an engaging betting experience for F1 fans.

# **Pre-requisites**

Make sure you have the following installed before proceeding:

- **Node.js** (>= 20.18)
- **npm** (>= 10.8)
- Backend API running ([F1-Betting API](https://github.com/JadoreThompson/f1-betting))

# **Requirements**

- API base URL configured in `.env` file (see below)

# **Installation**

```bash
git clone https://github.com/JadoreThompson/f1-betting-frontend

cd f1-betting-frontend

npm install

npm run dev
```

# **Environment Variables**

Create a `.env` file in the root directory and configure the following variables:

```
VITE_BASE_URL=http://192.168.1.145:8000
```