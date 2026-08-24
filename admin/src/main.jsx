import React, { useEffect, useState } from 'react'
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
import Launch from './pages/Launch'

const pages = { Dashboard, Playground, Models, Benchmarks, Earnings, Users, Account, Launch, Settings }

// The tab bar and the loaded bundle must agree; if a deploy changed the bundle while this tab
// was open, offer a reload instead of failing silently on a page that does not exist yet.
function useDeployWatch() {
  const [stale, setStale] = useState(false)
  useEffect(() => {
    const mine = document.querySelector('script[type=module]')?.src || ''
    const check = () => fetch('/admin/ui', { cache: 'no-store' }).then((r) => r.text()).then((html) => {
      const m = html.match(/src="([^"]*assets\/index[^"]*)"/)
      if (m && mine && !mine.endsWith(m[1])) setStale(true)
    }).catch(() => {})
    const t = setInterval(check, 30000)
    check()
    return () => clearInterval(t)
  }, [])
  return stale
}

function App() {
  const [page, setPage] = useState('Dashboard')
  const stale = useDeployWatch()
  const Page = pages[page]
  return (
    <div className="app">
      <nav>
        <div className="brand"><span className="bolt">⚡</span> llmfa.st</div>
        {Object.keys(pages).map((p) => <button key={p} className={p === page ? 'active' : ''} onClick={() => setPage(p)}>{p}</button>)}
      </nav>
      <main>
        {stale && <div className="card" style={{ marginBottom: 16, borderColor: 'var(--accent)' }}>
          A newer version of the dashboard is deployed. <a href="#" onClick={(e) => { e.preventDefault(); location.reload(true) }}>Reload</a> to pick it up.
        </div>}
        <Page />
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>)
