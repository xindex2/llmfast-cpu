import { useEffect, useState } from 'react'
import { api, cfg } from '../api'

const usd = (n) => '$' + (n || 0).toFixed(4)

export default function Account() {
  const [me, setMe] = useState(null)
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [pass, setPass] = useState('')
  const [err, setErr] = useState('')
  const [fresh, setFresh] = useState(null) // newly created key, shown once

  const load = () => api.me().then(setMe).catch(() => setMe(null))
  useEffect(load, [])

  const submit = async () => {
    setErr('')
    try {
      await (mode === 'login' ? api.login(email, pass) : api.signup(email, pass))
      setPass('')
      load()
    } catch (e) { setErr(e.message) }
  }
  const newKey = async () => {
    try { const k = await api.createMyKey('default'); setFresh(k.key); load() } catch (e) { setErr(e.message) }
  }
  const revoke = async (prefix) => { try { await api.revokeMyKey(prefix); load() } catch (e) { setErr(e.message) } }

  if (!me) return (
    <div className="auth">
      <div className="card">
        <h2>{mode === 'login' ? 'Sign in' : 'Create account'}</h2>
        <p className="lede" style={{ textAlign: 'center' }}>API access to open models, billed per token.</p>
        <div className="row">
          <input placeholder="you@company.com" value={email} onChange={(e) => setEmail(e.target.value)} />
          <input type="password" placeholder="password (8+ characters)" value={pass} onChange={(e) => setPass(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()} />
          <button className="primary" onClick={submit}>{mode === 'login' ? 'Sign in' : 'Create account'}</button>
        </div>
        {err && <p className="bad">{err}</p>}
        <p className="muted" style={{ textAlign: 'center', margin: 0 }}>
          {mode === 'login' ? 'No account? ' : 'Already have one? '}
          <a href="#" onClick={(e) => { e.preventDefault(); setMode(mode === 'login' ? 'signup' : 'login'); setErr('') }}>
            {mode === 'login' ? 'Create one' : 'Sign in'}
          </a>
        </p>
      </div>
    </div>
  )

  const u = me.user
  return (
    <>
      <h2>Your account</h2>
      <p className="lede">Signed in as {u.email}. Keys authenticate against <code>{cfg.base || window.location.origin}/v1/chat/completions</code> — any OpenAI-compatible client works.</p>

      <div className="grid">
        <Card label="Credit" value={'$' + u.credit_usd.toFixed(2)} sub={u.credit_usd <= 0 ? <span className="bad">top up to keep serving</span> : 'prepaid balance'} />
        <Card label="Spent (lifetime)" value={'$' + u.spent_usd.toFixed(2)} sub={`${me.usage.requests} requests in 30d`} />
        <Card label="Tokens (30d)" value={me.usage.prompt_tokens + me.usage.completion_tokens} sub={`${me.usage.prompt_tokens} in · ${me.usage.completion_tokens} out`} />
      </div>

      {fresh && <div className="card" style={{ marginBottom: 16, borderColor: 'var(--accent)' }}>
        <div className="label">New key — copy it now, it is not shown again</div>
        <div style={{ marginTop: 8 }}><code style={{ fontSize: 14 }}>{fresh}</code></div>
      </div>}

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ marginBottom: 8 }}>
          <div className="label" style={{ flex: 1 }}>API keys</div>
          <button className="primary" onClick={newKey}>Create key</button>
          <button className="ghost" onClick={() => api.logout().then(load)}>Sign out</button>
        </div>
        <table><tbody>{me.keys.map((k) => (
          <tr key={k.prefix}>
            <td><code>{k.prefix}</code></td>
            <td className="muted">{k.name}</td>
            <td className="muted">{k.last_used ? 'used ' + new Date(k.last_used).toLocaleDateString() : 'never used'}</td>
            <td><button className="ghost" onClick={() => revoke(k.prefix)}>Revoke</button></td>
          </tr>))}</tbody></table>
        {!me.keys.length && <p className="muted">no keys yet</p>}
      </div>
      {err && <p className="bad">{err}</p>}

      <div className="card">
        <div className="label">Quickstart</div>
        <pre style={{ margin: '10px 0 0', fontSize: 13, overflowX: 'auto' }}>{`curl ${cfg.base || window.location.origin}/v1/chat/completions \\
  -H "Authorization: Bearer YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"qwen3.8-27b","stream":true,
       "messages":[{"role":"user","content":"Hello"}]}'`}</pre>
      </div>
    </>
  )
}

function Card({ label, value, sub }) {
  return <div className="card"><div className="label">{label}</div><div className="value">{value}</div><div className="sub">{sub}</div></div>
}
