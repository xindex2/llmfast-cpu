import { useEffect, useRef, useState } from 'react'
import { api, streamChat } from '../api'
import Markdown from '../md'
import { Send, Stop, Trash } from '../icons'

export default function Playground({ session = false }) {
  const [models, setModels] = useState([])
  const [model, setModel] = useState('')
  const [system, setSystem] = useState('You are a helpful assistant.')
  const [temp, setTemp] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [input, setInput] = useState('')
  const [msgs, setMsgs] = useState([])
  const [busy, setBusy] = useState(false)
  const [stat, setStat] = useState(null)
  const abort = useRef(null)
  const bottom = useRef(null)

  useEffect(() => { api.models().then((d) => { setModels(d.data); setModel(d.data[0]?.id || '') }).catch(() => {}) }, [])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const send = async () => {
    if (!input.trim() || busy) return
    const history = [...msgs, { role: 'user', content: input }]
    setMsgs([...history, { role: 'assistant', content: '' }])
    setInput('')
    setBusy(true)
    abort.current = new AbortController()
    const t0 = performance.now()
    let first = 0, n = 0
    try {
      const usage = await streamChat(
        { model, temperature: +temp, max_tokens: +maxTokens, messages: [{ role: 'system', content: system }, ...history] },
        (tok) => {
          if (!first) first = performance.now()
          n++
          setMsgs((m) => { const c = [...m]; c[c.length - 1] = { ...c[c.length - 1], content: c[c.length - 1].content + tok }; return c })
        },
        (think) => {
          if (!first) first = performance.now()
          setMsgs((m) => { const c = [...m]; c[c.length - 1] = { ...c[c.length - 1], reasoning: (c[c.length - 1].reasoning || '') + think }; return c })
        },
        abort.current.signal,
        { session },
      )
      const gen = (performance.now() - first) / 1000
      // Two clocks, deliberately both shown. The browser one starts before fetch(), so it
      // includes DNS, TCP, TLS and the round trip to the datacenter; `usage.timing` is what
      // the server measured. They disagree by exactly your network latency, which is the
      // point — one is what a user in this location feels, the other is what we can improve.
      setStat({ ttft: first - t0, tps: (usage?.completion_tokens || n) / gen, usage, server: usage?.timing })
    } catch (e) {
      if (e.name !== 'AbortError') setMsgs((m) => [...m, { role: 'assistant', content: '⚠ ' + e.message }])
    } finally { setBusy(false) }
  }

  return (
    <div className="play">
      <h2>Playground</h2>
      <div className="row">
        <label>model<select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((m) => <option key={m.id}>{m.id}</option>)}</select></label>
        <label>temperature<input type="number" step="0.1" min="0" max="2" value={temp} onChange={(e) => setTemp(e.target.value)} style={{ width: 80 }} /></label>
        <label>max tokens<input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} style={{ width: 90 }} /></label>
        <label style={{ flex: 1 }}>system prompt<input value={system} onChange={(e) => setSystem(e.target.value)} /></label>
        <button className="ghost" onClick={() => { setMsgs([]); setStat(null) }}><Trash /> Clear</button>
      </div>

      <div className="chat">
        {msgs.map((m, i) => <div key={i} className={`msg ${m.role}`}>
          {m.reasoning && <details className="think" open><summary>reasoning</summary><Markdown text={m.reasoning} /></details>}
          {m.role === 'assistant'
            ? (m.content ? <Markdown text={m.content} /> : (busy && !m.reasoning ? <span className="caret">▍</span> : null))
            : m.content}
        </div>)}
        <div ref={bottom} />
      </div>

      {stat && <p className="muted">
        TTFT {stat.ttft.toFixed(0)} ms · {stat.tps.toFixed(1)} tok/s · {stat.usage?.prompt_tokens} prompt / {stat.usage?.completion_tokens} completion tokens
        {stat.usage?.prompt_tokens_details?.cached_tokens > 0 && <span title="Prompt tokens whose KV state was reused from an earlier request — these skip prefill entirely, which is what TTFT is made of.">
          {' '}· {stat.usage.prompt_tokens_details.cached_tokens} of {stat.usage.prompt_tokens} prompt cached
        </span>}
        {stat.server && <>
          {'\u2003'}
          <span title="Measured by the server. The difference from the number on the left is network latency between you and the box.">
            server-side: {stat.server.ttft_ms.toFixed(0)} ms · {stat.server.tok_per_sec.toFixed(1)} tok/s
            {stat.ttft - stat.server.ttft_ms > 5 && ` (+${(stat.ttft - stat.server.ttft_ms).toFixed(0)} ms network)`}
            {stat.server.accept_rate > 0 && ` · ${(stat.server.accept_rate * 100).toFixed(0)}% drafts accepted`}
          </span>
        </>}
      </p>}

      <div className="row composer">
        <textarea rows={2} value={input} placeholder="Message… (Enter to send, Shift+Enter for newline)" onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }} style={{ flex: 1 }} />
        {busy ? <button className="danger" onClick={() => abort.current?.abort()}><Stop /> Stop</button>
              : <button className="primary" onClick={send} disabled={!model}><Send /> Send</button>}
      </div>
    </div>
  )
}
