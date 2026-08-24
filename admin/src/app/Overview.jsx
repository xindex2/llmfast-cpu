// The customer's own usage. Same numbers the admin dashboard shows, filtered server-side to
// this account — the response never contains another customer's requests.
import { useEffect, useState } from 'react'
import { api } from '../api'

const fmt = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(Math.round(n || 0))
const usd = (n) => '$' + (n || 0).toFixed(n < 1 ? 4 : 2)
const PERIODS = [['today', 'Today'], ['7d', '7 days'], ['30d', '30 days']]

export default function Overview() {
  const [period, setPeriod] = useState('7d')
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    let alive = true
    const load = () => api.usage(`period=${period}`)
      .then((d) => alive && (setData(d), setErr('')))
      .catch((e) => alive && setErr(e.message))
    load()
    const t = setInterval(load, 10000)
    return () => { alive = false; clearInterval(t) }
  }, [period])

  if (err) return <p className="bad">{err}</p>
  if (!data) return <p className="muted">loading…</p>
  const s = data.summary
  const u = data.user
  const total = s.prompt_tokens + s.completion_tokens
  const maxBar = Math.max(1, ...(s.hourly || []).map((h) => h.tokens))

  return (
    <>
      <h2>Overview</h2>
      <p className="lede">Your usage and spend. Requests are billed per token at the prices on each model; cached prompt tokens are billed at the cache rate.</p>

      {u.credit_usd <= 0 && (
        <div className="card notice">
          <strong>Your balance is empty.</strong> Requests will return <code>402 insufficient_quota</code> until you top up.
          {' '}<a href="#/credits">Add credit</a>.
        </div>
      )}

      <div className="row">
        {PERIODS.map(([v, label]) => (
          <button key={v} className={period === v ? 'primary' : 'ghost'} onClick={() => setPeriod(v)}>{label}</button>
        ))}
      </div>

      <div className="grid">
        <Card label="Credit remaining" value={usd(u.credit_usd)} sub={`${usd(u.spent_usd)} spent all time`} />
        <Card label="Spend" value={usd(s.earnings_usd)} sub={`${fmt(s.requests)} requests`} />
        <Card label="Tokens" value={fmt(total)} sub={`${fmt(s.prompt_tokens)} in · ${fmt(s.completion_tokens)} out · ${fmt(s.cached_tokens)} cached`} />
        <Card label="TTFT p50 / p95" value={`${(s.p50_ttft_ms || 0).toFixed(0)} / ${(s.p95_ttft_ms || 0).toFixed(0)} ms`} sub="time to first token" />
        <Card label="Throughput" value={(s.avg_tok_per_sec || 0).toFixed(1) + ' tok/s'} sub="per stream, average" />
        <Card label="Failed" value={String(s.errors || 0)} sub={<span className={s.errors ? 'bad' : 'ok'}>{s.errors ? 'see recent requests' : 'no errors'}</span>} />
      </div>

      <div className="card" style={{ marginBottom: 22 }}>
        <div className="label">Tokens per hour</div>
        <div className="bars" style={{ marginTop: 12 }}>
          {(s.hourly || []).map((h) => (
            <div key={h.hour} className="bar" title={`${h.hour}: ${fmt(h.tokens)} tokens, ${usd(h.earnings_usd)}`}
              style={{ height: `${(h.tokens / maxBar) * 100}%` }} />
          ))}
          {!s.hourly?.length && <span className="muted">no traffic in this period</span>}
        </div>
      </div>

      <div className="card">
        <div className="label">Recent requests</div>
        <table>
          <thead><tr><th>time</th><th>model</th><th>key</th><th>in</th><th>out</th><th>cached</th><th>TTFT</th><th>tok/s</th><th>cost</th><th>status</th></tr></thead>
          <tbody>{(s.recent || []).map((r) => (
            <tr key={r.id}>
              <td>{new Date(r.at).toLocaleTimeString()}</td>
              <td>{r.model}</td>
              <td className="muted mono">{r.key}</td>
              <td>{r.prompt_tokens}</td><td>{r.completion_tokens}</td><td>{r.cached_tokens || 0}</td>
              <td>{r.ttft_ms.toFixed(0)}ms</td><td>{r.tok_per_sec.toFixed(1)}</td><td>{usd(r.earnings_usd)}</td>
              <td><span className={`pill ${r.error ? 'bad' : 'ok'}`}>{r.error ? 'failed' : 'ok'}</span></td>
            </tr>))}</tbody>
        </table>
        {!s.recent?.length && <p className="muted">nothing yet — try the <a href="#/playground">playground</a></p>}
      </div>
    </>
  )
}

function Card({ label, value, sub }) {
  return <div className="card"><div className="label">{label}</div><div className="value">{value}</div><div className="sub">{sub}</div></div>
}
