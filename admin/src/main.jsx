import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import Dashboard from './pages/Dashboard'
import Playground from './pages/Playground'
import Benchmarks from './pages/Benchmarks'
import Earnings from './pages/Earnings'
import Settings from './pages/Settings'
import Models from './pages/Models'
import Users from './pages/Users'
import Account from './pages/Account'

const pages = { Dashboard, Playground, Models, Benchmarks, Earnings, Users, Account, Settings }

function App() {
  const [page, setPage] = useState('Dashboard')
  const Page = pages[page]
  return (
    <div className="app">
      <nav>
        <div className="brand"><span className="bolt">⚡</span> llmfa.st</div>
        {Object.keys(pages).map((p) => <button key={p} className={p === page ? 'active' : ''} onClick={() => setPage(p)}>{p}</button>)}
      </nav>
      <main><Page /></main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>)
