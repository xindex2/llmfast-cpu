// The customer-facing console at llmfa.st/. Same design tokens as the admin UI and the
// marketing site, but a different audience: it never shows anyone else's traffic, and every
// page works with nothing but a login cookie. Admin lives at /admin/ui and is not linked here.
import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import { api } from './api'
import { Bolt, Logout } from './icons'
import Auth from './app/Auth'
import Overview from './app/Overview'
import Keys from './app/Keys'
import Credits from './app/Credits'
import Docs from './app/Docs'
import Playground from './pages/Playground'

// Hash routing, so the gateway only has to serve one HTML file for the whole console —
// no server-side deep-link rules to keep in sync with the page list.
const pages = {
  overview: ['Overview', Overview],
  playground: ['Playground', () => <Playground session />],
  keys: ['API keys', Keys],
  credits: ['Credits', Credits],
  docs: ['Docs', Docs],
}

function useHash() {
  const [hash, setHash] = useState(() => location.hash.slice(2) || 'overview')
  useEffect(() => {
    const on = () => setHash(location.hash.slice(2) || 'overview')
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  return [hash, (h) => { location.hash = '#/' + h }]
}

function App() {
  const [me, setMe] = useState(undefined) // undefined = still checking, null = signed out
  const [page, go] = useHash()
  const load = () => api.me().then(setMe).catch(() => setMe(null))
  useEffect(load, [])

  if (me === undefined) return <div className="app-solo"><p className="muted">loading…</p></div>
  if (me === null) return <Auth onDone={load} />

  const [, Page] = pages[page] || pages.overview
  return (
    <div className="app">
      <nav>
        <div className="brand"><Bolt size={18} /> llmfa.st</div>
        {Object.entries(pages).map(([k, [label]]) => (
          <button key={k} className={k === page ? 'active' : ''} onClick={() => go(k)}>{label}</button>
        ))}
        <div className="nav-foot">
          <div className="credit-pill" title="prepaid balance">
            <span className="muted small">credit</span>
            <strong>${me.user.credit_usd.toFixed(2)}</strong>
          </div>
          <div className="muted small" style={{ margin: '8px 12px' }}>{me.user.email}</div>
          <button onClick={() => api.logout().then(() => setMe(null))}><Logout size={14} /> Sign out</button>
        </div>
      </nav>
      <main><Page me={me} reload={load} /></main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>)
