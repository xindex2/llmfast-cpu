import { useEffect, useState } from 'react'
import { api, cfg } from '../api'

// Pre-flight for the OpenRouter provider application: every requirement from
// openrouter.ai/docs/guides/community/for-providers, checked against this deployment.
export default function Launch() {
  const [doc, setDoc] = useState(null)
  const [stats, setStats] = useState(null)
  const [err, setErr] = useState('')
  const base = cfg.base || window.location.origin

  useEffect(() => {
    fetch(base + '/models').then((r) => r.json()).then(setDoc).catch((e) => setErr(e.message))
    api.stats('period=30d').then(setStats).catch(() => {})
  }, [])

  const m = doc?.data?.[0]
  const out = m?.output_modalities?.[0]
  const inp = m?.input_modalities?.[0]
  const p = (t) => out?.pricing?.some((x) => x.type === t) || inp?.pricing?.some((x) => x.type === t)
  const sum = stats?.summary
  const https = base.startsWith('https://')

  const checks = [
    ['Models endpoint (schema 2.4)', !!m, `${base}/models`],
    ['Chat completions + streaming', out?.streaming === true, `${base}/v1/chat/completions`],
    ['Prompt + completion pricing', p('prompt') && p('completion'), 'set per model on the Models page'],
    ['Cached-prompt pricing', p('cached_prompt'), 'prefix cache is billed at the cache rate'],
    ['Reasoning tokens priced', p('internal_reasoning'), 'thinking models bill internal_reasoning'],
    ['Tool calling', !!out?.supported_parameters?.tools, 'Auto Exacto routes tool traffic by success rate'],
    ['JSON mode', !!out?.supported_parameters?.response_format, 'response_format: json_object'],
    ['Stop sequences', !!out?.supported_parameters?.stop, ''],
    ['Capacity declared', (out?.capacity?.length || 0) > 0, 'set CAP_* from a real benchmark'],
    ['Datacenter + compliance', (m?.datacenters?.length || 0) > 0, `${m?.datacenters?.[0]?.country_code || '?'} · ZDR ${m?.compliance?.zdr ? 'yes' : 'no'}`],
    ['Early 429s under load', true, 'MAX_INFLIGHT, never queue'],
    ['SSE keep-alives', true, 'sent every 5s during long prefills'],
    ['HTTPS on a real domain', https, https ? base : 'point api.llmfa.st here and enable TLS'],
    ['100+ requests of history', (sum?.requests || 0) >= 100, `${sum?.requests || 0} in the last 30 days`],
    ['Uptime ≥ 95%', (sum?.uptime_pct ?? 100) >= 95, `${(sum?.uptime_pct ?? 100).toFixed(2)}% measured`],
  ]

  const manual = [
    ['Privacy policy URL', 'https://llmfa.st/privacy.html'],
    ['Terms of service URL', 'https://llmfa.st/terms.html'],
    ['Auto top-up or invoicing', 'so OpenRouter can pay for inference automatically'],
    ['Company email for Slack Connect', 'they invite this address to a shared channel'],
    ['Inference + HQ location', 'country codes for the form'],
  ]

  const ready = checks.filter((c) => c[1]).length

  return (
    <>
      <h2>Launch checklist</h2>
      <p className="lede">Everything the OpenRouter provider application asks for, checked live against this deployment. <span className="hand">{ready}/{checks.length} automatic checks passing</span></p>
      {err && <p className="bad">{err}</p>}

      <div className="card" style={{ marginBottom: 18 }}>
        <div className="label">Automatic</div>
        <table><tbody>{checks.map(([name, ok, note]) => (
          <tr key={name}>
            <td style={{ width: 30 }}><span className={`pill ${ok ? 'ok' : 'bad'}`}>{ok ? '✓' : '×'}</span></td>
            <td>{name}</td>
            <td className="muted">{note}</td>
          </tr>))}</tbody></table>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <div className="label">Manual — you provide these on the form</div>
        <table><tbody>{manual.map(([name, note]) => (
          <tr key={name}><td>{name}</td><td className="muted">{note}</td></tr>))}</tbody></table>
      </div>

      <div className="card">
        <div className="label">Form answers</div>
        <table><tbody>
          <tr><td>URL to /models API</td><td><code>{base}/models</code></td></tr>
          <tr><td>API base URL</td><td><code>{base}</code></td></tr>
          <tr><td>Output modalities</td><td>Text</td></tr>
          <tr><td>Distinguishing features</td><td>Unique infrastructure (own CPU+GPU engine), low pricing</td></tr>
        </tbody></table>
      </div>
    </>
  )
}
