// Everything a customer needs to make the first call, with the live model list and the prices
// they will actually be billed at — so the docs cannot drift from what the gateway serves.
import { useEffect, useState } from 'react'
import { api, cfg } from '../api'

const usd = (n) => '$' + (n || 0).toFixed(3)

export default function Docs() {
  const [models, setModels] = useState([])
  useEffect(() => { api.models().then((d) => setModels(d.data || [])).catch(() => {}) }, [])
  const base = cfg.base || location.origin
  const id = models[0]?.id || 'qwen3-0.6b'

  return (
    <>
      <h2>Docs</h2>
      <p className="lede">The API is OpenAI-compatible. Point any client at <code>{base}/v1</code> with your key and it works — no SDK of ours to install.</p>

      <div className="card">
        <div className="label">Models and prices</div>
        <table>
          <thead><tr><th>model</th><th>context</th><th>input $/M</th><th>output $/M</th><th>cache read $/M</th></tr></thead>
          <tbody>{models.map((m) => (
            <tr key={m.id}>
              <td className="mono">{m.id}</td>
              <td>{(m.context_length || 0).toLocaleString()}</td>
              <td>{usd(m.prompt_price_per_m)}</td>
              <td>{usd(m.completion_price_per_m)}</td>
              <td>{usd(m.cached_price_per_m)}</td>
            </tr>))}</tbody>
        </table>
        {!models.length && <p className="muted">no models are being served right now</p>}
      </div>

      <Snippet title="curl" code={`curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer $LLMFAST_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${id}",
    "messages": [{"role": "user", "content": "Say hello."}],
    "stream": true
  }'`} />

      <Snippet title="Python (openai)" code={`from openai import OpenAI

client = OpenAI(base_url="${base}/v1", api_key="YOUR_KEY")

stream = client.chat.completions.create(
    model="${id}",
    messages=[{"role": "user", "content": "Say hello."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`} />

      <Snippet title="Node (openai)" code={`import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${base}/v1", apiKey: process.env.LLMFAST_KEY });

const stream = await client.chat.completions.create({
  model: "${id}",
  messages: [{ role: "user", content: "Say hello." }],
  stream: true,
});
for await (const c of stream) process.stdout.write(c.choices[0]?.delta?.content ?? "");`} />

      <div className="card">
        <div className="label">What is supported</div>
        <table><tbody>
          <Row k="Streaming" v="Server-sent events, with a final usage chunk." />
          <Row k="Reasoning" v={<>Thinking models emit <code>delta.reasoning</code> separately from <code>delta.content</code>.</>} />
          <Row k="Tools" v={<>Pass <code>tools</code> and read <code>tool_calls</code>, as with OpenAI.</>} />
          <Row k="JSON mode" v={<><code>response_format: {'{'} "type": "json_object" {'}'}</code>.</>} />
          <Row k="Prompt caching" v="Automatic. Repeated prefixes bill at the cache-read price." />
          <Row k="Errors" v={<><code>401</code> bad key · <code>402</code> out of credit · <code>429</code> at capacity, retry · <code>503</code> model still loading.</>} />
        </tbody></table>
      </div>
    </>
  )
}

const Row = ({ k, v }) => <tr><td style={{ whiteSpace: 'nowrap' }}><strong>{k}</strong></td><td>{v}</td></tr>

function Snippet({ title, code }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="card">
      <div className="row" style={{ alignItems: 'center' }}>
        <div className="label" style={{ flex: 1 }}>{title}</div>
        <button className="ghost small" onClick={() => navigator.clipboard?.writeText(code).then(() => setCopied(true))}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="code">{code}</pre>
    </div>
  )
}
