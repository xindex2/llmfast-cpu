// Sign-in / sign-up gate. Deliberately the only thing a signed-out visitor to the console
// can see: no model list, no pricing table, no traffic.
import { useState } from 'react'
import { api } from '../api'
import { Bolt } from '../icons'

export default function Auth({ onDone }) {
  const [mode, setMode] = useState(() => location.hash.includes('signup') ? 'signup' : 'login')
  const [email, setEmail] = useState('')
  const [pass, setPass] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const submit = async (e) => {
    e?.preventDefault()
    if (busy) return
    if (mode === 'signup' && pass.length < 8) return setErr('password must be at least 8 characters')
    setBusy(true); setErr('')
    try {
      await (mode === 'login' ? api.login(email, pass) : api.signup(email, pass))
      setPass('')
      onDone()
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="app-solo">
      <form className="card auth-card" onSubmit={submit}>
        <div className="brand" style={{ justifyContent: 'center', marginBottom: 4 }}><Bolt size={18} /> llmfa.st</div>
        <h2 style={{ textAlign: 'center' }}>{mode === 'login' ? 'Sign in' : 'Create an account'}</h2>
        <p className="muted small" style={{ textAlign: 'center', marginTop: -6 }}>
          An OpenAI-compatible API for open models, billed per token.
        </p>
        <label>email
          <input type="email" autoComplete="email" required placeholder="you@company.com"
            value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label>password
          <input type="password" required minLength={8}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            placeholder={mode === 'login' ? '' : 'at least 8 characters'}
            value={pass} onChange={(e) => setPass(e.target.value)} />
        </label>
        <button className="primary" type="submit" disabled={busy}>
          {busy ? 'working…' : mode === 'login' ? 'Sign in' : 'Create account'}
        </button>
        {err && <p className="bad small" style={{ margin: 0 }}>{err}</p>}
        <p className="muted small" style={{ textAlign: 'center', margin: 0 }}>
          {mode === 'login' ? 'No account? ' : 'Already have one? '}
          <a href="#" onClick={(e) => { e.preventDefault(); setMode(mode === 'login' ? 'signup' : 'login'); setErr('') }}>
            {mode === 'login' ? 'Create one' : 'Sign in'}
          </a>
        </p>
      </form>
    </div>
  )
}
