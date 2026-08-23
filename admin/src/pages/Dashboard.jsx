import { useEffect, useState } from 'react'
import { api } from '../api'

const fmt = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(Math.round(n))
const usd = (n) => '$' + n.toFixed(n < 1 ? 4 : 2)

export default function Dashboard() {
  const [hours, setHours] = useState(24)
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    let alive = true
    const load = () => api.stats(hours).then((d) => alive && setData(d)).catch((e) => alive && setErr(e.message))
    load()
    const t = setInterval(load, 5000)
    return () => { alive = false; clearInterval(t) }
  }, [hours])

  if (err) return <p className="bad">{err}</p>
  if (!data) return <p className="muted">loading…</p>
  const s = data.summary
  const total = s.prompt_tokens + s.completion_tokens
  const maxBar = Math.max(1, ...(s.hourly || []).map((h) => h.tokens))
  const target = 100e6

  return (
    <>
      <div className="row">
        <h2 style={{ margin: 0, flex: 1 }}>Dashboard</h2>
        <select value={hours} onChange={(e) => setHours(+e.target.value)}>
          <option value={1}>last hour</option><option value={24}>last 24h</option>
          <option value={168}>last 7 days</option><option value={720}>last 30 days</option>
        </select>
      </div>

      <div className="grid">
        <Card label="Tokens served" value={fmt(total)} sub={`${fmt(s.prompt_tokens)} prompt · ${fmt(s.completion_tokens)} completion`} />
        <Card label="Earnings" value={usd(s.earnings_usd)} sub={`≈ ${usd(s.earnings_usd / (hours / 24))} / day`} />
        <Card label="Run-rate" value={fmt(s.tokens_per_day_rate) + ' tok/day'} sub={`${((s.tokens_per_day_rate / target) * 100).toFixed(2)}% of 100M/day target`} />
        <Card label="Requests" value={fmt(s.requests)} sub={<span className={s.errors ? 'bad' : 'ok'}>{s.errors} errors · {data.inflight} in flight</span>} />
        <Card label="Avg TTFT" value={s.avg_ttft_ms.toFixed(0) + ' ms'} sub="time to first token" />
        <Card label="Avg speed" value={s.avg_tok_per_sec.toFixed(1) + ' tok/s'} sub="per stream" />
      </div>

      <div className="card" style={{ marginBottom: 22 }}>
        <div className="label">Tokens per hour</div>
        <div className="bars" style={{ marginTop: 10 }}>
          {(s.hourly || []).map((h) => <div key={h.hour} className="bar" title={`${h.hour}: ${fmt(h.tokens)} tokens, ${usd(h.earnings_usd)}`} style={{ height: `${(h.tokens / maxBar) * 100}%` }} />)}
          {!s.hourly?.length && <span className="muted">no traffic in window</span>}
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="card">
          <div className="label">Engines</div>
          <table><tbody>{data.engines.map((e) => <tr key={e.name}><td>{e.name}</td><td className="muted">{e.model} · {e.device || '?'}</td><td className={e.healthy ? 'ok' : 'bad'}>{e.healthy ? 'healthy' : 'DOWN'}</td></tr>)}</tbody></table>
        </div>
        <div className="card">
          <div className="label">Tokens by model</div>
          <table><tbody>{Object.entries(s.tokens_by_model).map(([m, t]) => <tr key={m}><td>{m}</td><td>{fmt(t)}</td></tr>)}</tbody></table>
        </div>
      </div>

      <div className="card">
        <div className="label">Recent requests</div>
        <table>
          <thead><tr><th>time</th><th>key</th><th>model</th><th>engine</th><th>prompt</th><th>completion</th><th>TTFT</th><th>tok/s</th><th>earned</th><th>status</th></tr></thead>
          <tbody>{(s.recent || []).map((r) => (
            <tr key={r.id}>
              <td>{new Date(r.at).toLocaleTimeString()}</td><td><code>{r.key.slice(0, 10)}</code></td><td>{r.model}</td><td>{r.engine}</td>
              <td>{r.prompt_tokens}</td><td>{r.completion_tokens}</td><td>{r.ttft_ms.toFixed(0)}ms</td><td>{r.tok_per_sec.toFixed(1)}</td>
              <td>{usd(r.earnings_usd)}</td><td className={r.error ? 'bad' : 'ok'}>{r.error || 'ok'}</td>
            </tr>))}</tbody>
        </table>
      </div>
    </>
  )
}

function Card({ label, value, sub }) {
  return <div className="card"><div className="label">{label}</div><div className="value">{value}</div><div className="sub">{sub}</div></div>
}
