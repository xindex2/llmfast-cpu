import { useEffect, useState } from 'react'
import { api, cfg } from '../api'

export default function Settings() {
  const [base, setBase] = useState(cfg.base)
  const [admin, setAdmin] = useState(cfg.adminToken)
  const [key, setKey] = useState(cfg.apiKey)
  const [keys, setKeys] = useState([])
  const [err, setErr] = useState('')

  const load = () => api.keys().then((d) => setKeys(d.keys)).catch((e) => setErr(e.message))
  useEffect(load, [])

  const save = () => { cfg.base = base; cfg.adminToken = admin; cfg.apiKey = key; load() }
  const create = async () => { try { const d = await api.createKey(); setKeys([...keys, d.key]) } catch (e) { setErr(e.message) } }

  return (
    <>
      <h2>Settings & API keys</h2>
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="row">
          <label style={{ flex: 1 }}>gateway base URL (empty = same origin / dev proxy)<input value={base} onChange={(e) => setBase(e.target.value)} placeholder="https://api.yourdomain.com" /></label>
          <label>admin token<input value={admin} onChange={(e) => setAdmin(e.target.value)} /></label>
          <label>playground API key<input value={key} onChange={(e) => setKey(e.target.value)} /></label>
          <button className="primary" onClick={save}>save</button>
        </div>
      </div>
      {err && <p className="bad">{err}</p>}
      <div className="card">
        <div className="row"><div className="label" style={{ flex: 1 }}>API keys (give one to OpenRouter)</div><button className="primary" onClick={create}>create key</button></div>
        <table><tbody>{keys.map((k) => <tr key={k}><td><code>{k}</code></td></tr>)}</tbody></table>
      </div>
    </>
  )
}
