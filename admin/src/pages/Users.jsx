import { useEffect, useState } from 'react'
import { api } from '../api'

const usd = (n) => '$' + (n || 0).toFixed(2)

export default function Users() {
  const [rows, setRows] = useState([])
  const [err, setErr] = useState('')
  const [amount, setAmount] = useState(10)

  const load = () => api.users().then((d) => setRows(d.users || [])).catch((e) => setErr(e.message))
  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t) }, [])

  const topup = async (id) => {
    try { await api.topup(id, +amount); load() } catch (e) { setErr(e.message) }
  }

  return (
    <>
      <h2>Customers</h2>
      <p className="lede">Accounts that bought API access directly from us. Credit is prepaid: requests are refused with HTTP 402 when a balance runs out, so we never serve unpaid tokens.</p>
      {err && <p className="bad">{err}</p>}
      <div className="row"><label>top-up amount ($)<input type="number" value={amount} onChange={(e) => setAmount(e.target.value)} style={{ width: 110 }} /></label></div>
      <div className="card">
        <table>
          <thead><tr><th>email</th><th>joined</th><th>credit</th><th>lifetime spend</th><th>requests (30d)</th><th>tokens (30d)</th><th></th></tr></thead>
          <tbody>{rows.map(({ user: u, usage }) => (
            <tr key={u.id}>
              <td>{u.email} {u.is_admin && <span className="pill">admin</span>}</td>
              <td className="muted">{new Date(u.created_at).toLocaleDateString()}</td>
              <td className={u.credit_usd <= 0 ? 'bad' : 'ok'}>{usd(u.credit_usd)}</td>
              <td>{usd(u.spent_usd)}</td>
              <td>{usage.requests}</td>
              <td>{usage.prompt_tokens + usage.completion_tokens}</td>
              <td><button className="ghost" onClick={() => topup(u.id)}>+ {usd(+amount)}</button></td>
            </tr>))}</tbody>
        </table>
        {!rows.length && <p className="muted">no customers yet — they sign up at /signup with the site's API</p>}
      </div>
    </>
  )
}
