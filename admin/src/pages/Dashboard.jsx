import { useEffect, useState } from 'react'
import { api } from '../api'

const fmt = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(Math.round(n || 0))
const usd = (n) => '$' + (n || 0).toFixed(n < 1 ? 4 : 2)
const iso = (d) => new Date(d).toISOString()

const PERIODS = [['today', 'Today'], ['yesterday', 'Yesterday'], ['7d', 'Last 7 days'], ['30d', 'Last 30 days'], ['custom', 'Custom…']]

export default function Dashboard() {
  const [period, setPeriod] = useState('today')
  const [from, setFrom] = useState(new Date(Date.now() - 7 * 864e5).toISOString().slice(0, 10))
  const [to, setTo] = useState(new Date().toISOString().slice(0, 10))
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    let alive = true
    const q = period === 'custom'
      ? `period=custom&from=${iso(from + 'T00:00:00')}&to=${iso(to + 'T23:59:59')}`
      : `period=${period}`
    const load = () => api.stats(q).then((d) => alive && (setData(d), setErr(''))).catch((e) => alive && setErr(e.message))
    load()
    const t = setInterval(load, 5000)
    return () => { alive = false; clearInterval(t) }
  }, [period, from, to])

  if (err) return <p className="bad">{err}</p>
  if (!data) return <p className="muted">loading…</p>
  const s = data.summary
  const total = s.prompt_tokens + s.completion_tokens
  const maxBar = Math.max(1, ...(s.hourly || []).map((h) => h.tokens))
  const errs = Object.entries(s.by_error || {}).sort((a, b) => b[1] - a[1])

  return (
    <>
      <h2>Dashboard</h2>
      <p className="lede">Traffic, tokens and earnings for the selected period. Uptime is what OpenRouter measures for routing: successful requests over total (95%+ keeps normal priority).</p>

      <div className="row">
        {PERIODS.map(([v, label]) => (
          <button key={v} className={period === v ? 'primary' : 'ghost'} onClick={() => setPeriod(v)}>{label}</button>
        ))}
        {period === 'custom' && <>
          <label>from<input type="date" value={from} onChange={(e) => setFrom(e.target.value)} /></label>
          <label>to<input type="date" value={to} onChange={(e) => setTo(e.target.value)} /></label>
        </>}
      </div>

      <div className="grid">
        <Card label="Tokens" value={fmt(total)} sub={`${fmt(s.prompt_tokens)} in · ${fmt(s.completion_tokens)} out · ${fmt(s.cached_tokens)} cached`} />
        <Card label="Earnings" value={usd(s.earnings_usd)} sub={`${fmt(s.requests)} requests · ${s.users || 0} users`} />
        <Card label="Uptime" value={(s.uptime_pct || 0).toFixed(2) + '%'} sub={<span className={s.errors ? 'bad' : 'ok'}>{s.errors} failed · {s.canceled || 0} canceled · {data.inflight} in flight</span>} />
        <Card label="TTFT p50 / p95" value={`${(s.p50_ttft_ms || 0).toFixed(0)} / ${(s.p95_ttft_ms || 0).toFixed(0)} ms`} sub="time to first token" />
        <Card label="Throughput" value={(s.avg_tok_per_sec || 0).toFixed(1) + ' tok/s'} sub="per stream, average" />
        <Card label="Run-rate" value={fmt(s.tokens_per_day_rate) + '/day'} sub={`${((s.tokens_per_day_rate / 100e6) * 100).toFixed(2)}% of 100M/day`} />
      </div>

      <div className="card" style={{ marginBottom: 22 }}>
        <div className="label">Tokens per hour</div>
        <div className="bars" style={{ marginTop: 12 }}>
          {(s.hourly || []).map((h) => <div key={h.hour} className="bar" title={`${h.hour}: ${fmt(h.tokens)} tokens, ${usd(h.earnings_usd)}`} style={{ height: `${(h.tokens / maxBar) * 100}%` }} />)}
          {!s.hourly?.length && <span className="muted">no traffic in this period</span>}
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="card">
          <div className="label">Engines</div>
          <table><tbody>{data.engines.map((e) => (
            <tr key={e.name}>
              <td>{e.model || '—'}</td>
              <td className="muted">{e.device || (e.loading ? 'loading' : '?')}</td>
              <td><span className={`pill ${e.healthy ? 'ok' : e.loading ? '' : 'bad'}`}>
                {e.healthy ? 'healthy' : e.loading ? `loading ${((e.progress || 0) * 100).toFixed(0)}%` : 'down'}
              </span></td>
            </tr>
          ))}</tbody></table>
          {!data.engines.length && <p className="muted">no engines running</p>}
        </div>
        <div className="card">
          <div className="label">{errs.length ? 'Failures' : 'Tokens by model'}</div>
          <table><tbody>
            {errs.length
              ? errs.map(([e, n]) => <tr key={e}><td className="bad">{e}</td><td>{n}</td></tr>)
              : Object.entries(s.tokens_by_model || {}).map(([m, t]) => <tr key={m}><td>{m}</td><td>{fmt(t)}</td></tr>)}
          </tbody></table>
        </div>
      </div>

      <div className="card">
        <div className="label">Recent requests</div>
        <table>
          <thead><tr><th>time</th><th>user</th><th>model</th><th>in</th><th>out</th><th>cached</th><th>TTFT</th><th>tok/s</th><th>earned</th><th>status</th></tr></thead>
          <tbody>{(s.recent || []).map((r) => (
            <tr key={r.id}>
              <td>{new Date(r.at).toLocaleTimeString()}</td>
              <td className="muted">{r.user_id ? r.user_id.slice(4, 12) : 'provider'}</td>
              <td>{r.model}</td><td>{r.prompt_tokens}</td><td>{r.completion_tokens}</td><td>{r.cached_tokens || 0}</td>
              <td>{r.ttft_ms.toFixed(0)}ms</td><td>{r.tok_per_sec.toFixed(1)}</td><td>{usd(r.earnings_usd)}</td>
              <td><span className={`pill ${r.error ? 'bad' : 'ok'}`}>{r.error ? 'failed' : 'ok'}</span></td>
            </tr>))}</tbody>
        </table>
        {!s.recent?.length && <p className="muted">nothing yet</p>}
      </div>
    </>
  )
}

function Card({ label, value, sub }) {
  return <div className="card"><div className="label">{label}</div><div className="value">{value}</div><div className="sub">{sub}</div></div>
}
