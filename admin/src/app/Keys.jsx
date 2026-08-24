// API key management. The raw key exists only in the response that creates it: we store a
// sha256 hash, so if the customer loses it there is nothing to recover — hence the one-time
// panel below rather than a "reveal" button that could never work.
import { useEffect, useState } from 'react'
import { api, cfg } from '../api'
import { Key, Trash, Plus } from '../icons'
import { Paged } from '../table'

// Go serialises a never-used timestamp as year 0001 rather than omitting it (omitempty does
// not apply to time.Time), so "never" has to be detected rather than assumed from absence.
const used = (t) => (t && !t.startsWith('0001-') ? new Date(t).toLocaleString() : 'never')

export default function Keys() {
  const [keys, setKeys] = useState([])
  const [name, setName] = useState('')
  const [fresh, setFresh] = useState(null)
  const [copied, setCopied] = useState(false)
  const [err, setErr] = useState('')

  const load = () => api.myKeys().then((d) => setKeys(d.keys || [])).catch((e) => setErr(e.message))
  useEffect(load, [])

  const create = async () => {
    setErr('')
    try { const k = await api.createMyKey(name.trim() || 'default'); setFresh(k); setName(''); setCopied(false); load() }
    catch (e) { setErr(e.message) }
  }
  const revoke = async (k) => {
    if (!confirm(`Revoke ${k.prefix}? Any client using it will start getting 401s immediately.`)) return
    try { await api.revokeMyKey(k.prefix); load() } catch (e) { setErr(e.message) }
  }
  const copy = () => navigator.clipboard?.writeText(fresh.key).then(() => setCopied(true)).catch(() => {})

  const base = cfg.base || location.origin

  return (
    <>
      <h2>API keys</h2>
      <p className="lede">Use a key as a bearer token against <code>{base}/v1</code>. Any OpenAI-compatible client works — set the base URL and the key, leave the rest alone.</p>

      {fresh && (
        <div className="card notice">
          <div className="label">New key — copy it now</div>
          <p className="small muted" style={{ margin: '4px 0 10px' }}>
            This is the only time it will be shown. We store a hash, not the key, so it cannot be recovered later.
          </p>
          <div className="row">
            <code className="keybox">{fresh.key}</code>
            <button className="primary" onClick={copy}>{copied ? 'Copied' : 'Copy'}</button>
            <button className="ghost" onClick={() => setFresh(null)}>Done</button>
          </div>
        </div>
      )}

      <div className="card">
        <div className="row">
          <label style={{ flex: 1 }}>name (optional)
            <input value={name} placeholder="production, laptop, ci…" onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && create()} />
          </label>
          <button className="primary" onClick={create}><Plus /> Create key</button>
        </div>
        {err && <p className="bad small">{err}</p>}
      </div>

      <div className="card">
        <div className="label">Your keys</div>
        <Paged items={keys} per={10} unit="keys">{(page) => (
        <table>
          <thead><tr><th>key</th><th>name</th><th>created</th><th>last used</th><th></th></tr></thead>
          <tbody>{page.map((k) => (
            <tr key={k.prefix}>
              <td className="mono"><Key size={13} /> {k.prefix}</td>
              <td>{k.name}</td>
              <td className="muted">{new Date(k.created_at).toLocaleDateString()}</td>
              <td className="muted">{used(k.last_used)}</td>
              <td><button className="danger small" onClick={() => revoke(k)}><Trash size={13} /> Revoke</button></td>
            </tr>))}</tbody>
        </table>)}
        </Paged>
        {!keys.length && <p className="muted">no keys yet</p>}
      </div>
    </>
  )
}
