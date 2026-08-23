import { useState } from 'react'

// Unit economics of a CPU inference provider. Every input is editable — change them to match
// real benchmarks and real server quotes; the math is the business plan.
const defaults = {
  targetTokPerDay: 100e6,
  promptShare: 0.7,            // typical chat traffic: ~70% prompt tokens, ~30% completion
  promptPrice: 0.10,           // USD per 1M prompt tokens
  completionPrice: 0.30,       // USD per 1M completion tokens
  openrouterCut: 0.20,         // OpenRouter takes a share; adjust to the real contract
  aggTokPerSec: 300,           // per server, from the Benchmarks page (completion tokens)
  prefillTokPerSec: 3000,      // per server, prompt ingestion speed
  serverCostPerMonth: 450,     // bare-metal EPYC w/ 256GB RAM, typical Hetzner/OVH-class price
  utilization: 0.6,            // traffic is bursty; servers sit idle part of the day
  bandwidthAndOps: 150,        // USD/month: domain, TLS, monitoring, egress
}

const fmt = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : n.toFixed(0)
const usd = (n) => (n < 0 ? '-$' : '$') + Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: 0 })

export default function Earnings() {
  const [p, setP] = useState(defaults)
  const set = (k) => (e) => setP({ ...p, [k]: +e.target.value })

  const promptTok = p.targetTokPerDay * p.promptShare
  const compTok = p.targetTokPerDay * (1 - p.promptShare)
  const grossPerDay = (promptTok / 1e6) * p.promptPrice + (compTok / 1e6) * p.completionPrice
  const netPerDay = grossPerDay * (1 - p.openrouterCut)

  // Server-seconds needed per day for completion + prefill, divided by utilization.
  const secNeeded = compTok / p.aggTokPerSec + promptTok / p.prefillTokPerSec
  const servers = Math.max(1, Math.ceil(secNeeded / 86400 / p.utilization))
  const costPerMonth = servers * p.serverCostPerMonth + p.bandwidthAndOps
  const revenuePerMonth = netPerDay * 30
  const profit = revenuePerMonth - costPerMonth
  const costPerMTok = costPerMonth / 30 / (p.targetTokPerDay / 1e6)
  const revPerMTok = netPerDay / (p.targetTokPerDay / 1e6)
  const breakEvenTokPerSec = (costPerMonth / 30) / netPerDay * (p.targetTokPerDay / 86400)

  const fields = [
    ['targetTokPerDay', 'target tokens / day'], ['promptShare', 'prompt share (0–1)'], ['promptPrice', 'prompt price $/M'],
    ['completionPrice', 'completion price $/M'], ['openrouterCut', 'OpenRouter cut (0–1)'], ['aggTokPerSec', 'completion tok/s per server'],
    ['prefillTokPerSec', 'prefill tok/s per server'], ['serverCostPerMonth', 'server $/month'], ['utilization', 'utilization (0–1)'], ['bandwidthAndOps', 'ops $/month'],
  ]

  return (
    <>
      <h2>Earnings calculator</h2>
      <div className="row">{fields.map(([k, l]) => <label key={k}>{l}<input type="number" step="any" value={p[k]} onChange={set(k)} style={{ width: 130 }} /></label>)}</div>

      <div className="grid">
        <Card label="Servers needed" value={servers} sub={`${fmt(p.targetTokPerDay)} tok/day at ${p.aggTokPerSec} tok/s/server, ${(p.utilization * 100).toFixed(0)}% util`} />
        <Card label="Revenue / month (net)" value={usd(revenuePerMonth)} sub={`gross ${usd(grossPerDay * 30)}, after ${(p.openrouterCut * 100).toFixed(0)}% cut`} />
        <Card label="Cost / month" value={usd(costPerMonth)} sub={`${servers} × ${usd(p.serverCostPerMonth)} + ${usd(p.bandwidthAndOps)} ops`} />
        <Card label="Profit / month" value={<span className={profit >= 0 ? 'ok' : 'bad'}>{usd(profit)}</span>} sub={`margin ${revenuePerMonth ? ((profit / revenuePerMonth) * 100).toFixed(0) : 0}%`} />
        <Card label="Cost per 1M tokens" value={'$' + costPerMTok.toFixed(3)} sub={`vs $${revPerMTok.toFixed(3)} earned per 1M`} />
        <Card label="Break-even speed" value={breakEvenTokPerSec.toFixed(0) + ' tok/s'} sub="blended tok/s the fleet must sustain to cover cost" />
      </div>

      <div className="card">
        <div className="label">Scale table (same assumptions)</div>
        <table>
          <thead><tr><th>tokens/day</th><th>servers</th><th>net revenue/mo</th><th>cost/mo</th><th>profit/mo</th></tr></thead>
          <tbody>{[10e6, 100e6, 500e6, 1e9, 10e9].map((t) => {
            const pt = t * p.promptShare, ct = t * (1 - p.promptShare)
            const rev = ((pt / 1e6) * p.promptPrice + (ct / 1e6) * p.completionPrice) * (1 - p.openrouterCut) * 30
            const sv = Math.max(1, Math.ceil((ct / p.aggTokPerSec + pt / p.prefillTokPerSec) / 86400 / p.utilization))
            const cost = sv * p.serverCostPerMonth + p.bandwidthAndOps
            return <tr key={t}><td>{fmt(t)}</td><td>{sv}</td><td>{usd(rev)}</td><td>{usd(cost)}</td><td className={rev - cost >= 0 ? 'ok' : 'bad'}>{usd(rev - cost)}</td></tr>
          })}</tbody>
        </table>
      </div>
    </>
  )
}

function Card({ label, value, sub }) {
  return <div className="card"><div className="label">{label}</div><div className="value">{value}</div><div className="sub">{sub}</div></div>
}
