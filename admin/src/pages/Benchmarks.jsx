import { useEffect, useState } from 'react'
import { api } from '../api'
import { Play } from '../icons'

export default function Benchmarks() {
  const [list, setList] = useState([])
  const [models, setModels] = useState([])
  const [serverCost, setServerCost] = useState(450)
  const [cfg, setCfg] = useState({ model: '', concurrency: 4, requests: 8, max_tokens: 128 })
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const load = () => api.benchmarks().then((d) => setList((d.benchmarks || []).slice().reverse())).catch((e) => setErr(e.message))
  useEffect(() => { load(); api.models().then((d) => { setModels(d.data); setCfg((c) => ({ ...c, model: d.data[0]?.id })) }).catch(() => {}) }, [])

  const run = async () => {
    setBusy(true); setErr('')
    try { await api.runBenchmark({ ...cfg, concurrency: +cfg.concurrency, requests: +cfg.requests, max_tokens: +cfg.max_tokens }); await load() }
    catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  const set = (k) => (e) => setCfg({ ...cfg, [k]: e.target.value })

  return (
    <>
      <h2>Benchmarks</h2>
      <p className="lede">Runs real requests against the live engine at the concurrency you choose. <span className="hand">Per-stream tok/s and p50 TTFT are what OpenRouter publishes for your endpoint</span> — aggregate tok/s is what sizes the fleet: servers needed = target tok/s ÷ aggregate tok/s per server. 100M tok/day ≈ 1,160 tok/s.</p>
      <div className="row">
        <label>model<select value={cfg.model} onChange={set('model')}>{models.map((m) => <option key={m.id}>{m.id}</option>)}</select></label>
        <label>concurrency<input type="number" value={cfg.concurrency} onChange={set('concurrency')} style={{ width: 80 }} /></label>
        <label>requests<input type="number" value={cfg.requests} onChange={set('requests')} style={{ width: 80 }} /></label>
        <label>max tokens<input type="number" value={cfg.max_tokens} onChange={set('max_tokens')} style={{ width: 90 }} /></label>
        <button className="primary" onClick={run} disabled={busy}>{busy ? 'Running…' : <><Play /> Run benchmark</>}</button>
      </div>
      {err && <p className="bad">{err}</p>}
      <div className="row"><label>server cost $/month (for the profit column)<input type="number" value={serverCost} onChange={(e) => setServerCost(+e.target.value)} style={{ width: 110 }} /></label></div>
      <div className="card">
        <table>
          <thead><tr><th>when</th><th>model</th><th>conc</th><th>reqs</th><th>TTFT p50 / p95</th><th>per-stream tok/s</th><th>aggregate tok/s</th><th>servers for 100M/day</th><th>revenue/day at this rate</th><th>revenue/month</th><th>profit/month</th><th>errors</th></tr></thead>
          <tbody>{list.map((b) => {
            const m = models.find((x) => x.id === b.model)
            // A server running flat out at this aggregate rate: completion tokens earn the output price;
            // assume ~2.3 prompt tokens per completion token (typical chat) at the input price.
            const out = m?.completion_price_per_m || 0, inp = m?.prompt_price_per_m || 0
            const perDayTok = b.agg_tok_per_sec * 86400
            const revDay = perDayTok / 1e6 * out + perDayTok * 2.3 / 1e6 * inp
            const revMonth = revDay * 30
            return (
            <tr key={b.id}>
              <td>{new Date(b.at).toLocaleString()}</td><td>{b.model}</td><td>{b.concurrency}</td><td>{b.requests}</td>
              <td>{(b.p50_ttft_ms || b.avg_ttft_ms).toFixed(0)} / {(b.p95_ttft_ms || b.avg_ttft_ms).toFixed(0)} ms</td><td>{b.avg_tok_per_sec.toFixed(1)}</td><td><b>{b.agg_tok_per_sec.toFixed(1)}</b></td>
              <td>{b.agg_tok_per_sec > 0 ? Math.ceil(1160 / b.agg_tok_per_sec) : '—'}</td>
              <td>{m ? '$' + revDay.toFixed(2) : <span className="muted">no price</span>}</td>
              <td>{m ? '$' + revMonth.toFixed(0) : '—'}</td>
              <td className={revMonth - serverCost >= 0 ? 'ok' : 'bad'}>{m ? (revMonth - serverCost >= 0 ? '$' : '-$') + Math.abs(revMonth - serverCost).toFixed(0) : '—'}</td>
              <td className={b.errors ? 'bad' : 'ok'} title={b.last_error || ''}>{b.errors}{b.last_error ? ' ⓘ' : ''}</td>
            </tr>)})}</tbody>
        </table>
        <p className="muted" style={{ marginBottom: 0 }}>Revenue assumes the server is saturated 24/7 at the measured aggregate rate, with ~2.3 prompt tokens per completion token, at the model's prices from the Models page. Profit subtracts the server cost above. Use the Earnings page for utilization, OpenRouter's cut, and fleet sizing.</p>
        {list.length === 0 && <p className="muted">no benchmarks yet</p>}
      </div>
    </>
  )
}
