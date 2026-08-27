import { useEffect, useState } from 'react'
import { api } from '../api'
import { Play, Stop, Trash, Download, Save, Refresh } from '../icons'

const gb = (b) => (b / 1e9).toFixed(2) + ' GB'

export default function Models() {
  const [data, setData] = useState({ models: [] })
  const [form, setForm] = useState({ hf_id: '', id: '', context_length: 8192, prompt_price_per_m: 0.1, completion_price_per_m: 0.3, cached_price_per_m: 0.025, quant: 'q8' })
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState('')

  const load = () => api.models_admin().then(setData).catch((e) => setErr(e.message))
  useEffect(() => { load(); const t = setInterval(load, 2000); return () => clearInterval(t) }, [])

  const add = async () => {
    setErr(''); setBusy('add')
    try { await api.addModel({ ...form, context_length: +form.context_length, prompt_price_per_m: +form.prompt_price_per_m, completion_price_per_m: +form.completion_price_per_m, cached_price_per_m: +form.cached_price_per_m }); setForm({ ...form, hf_id: '', id: '' }); load() }
    catch (e) { setErr(e.message) } finally { setBusy('') }
  }
  const action = async (id, a, q = '') => { setErr(''); setBusy(id + a); try { await api.modelAction(id, a, q); load() } catch (e) { setErr(e.message) } finally { setBusy('') } }
  const save = async (m) => { setErr(''); try { await api.updateModel(m); load() } catch (e) { setErr(e.message) } }
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })

  return (
    <>
      <h2>Models</h2>
      <p className="muted">Paste a Hugging Face model id or link (safetensors checkpoint, Qwen3 dense or MoE). The gateway downloads it to <code>{data.models_dir}</code>, then you can start an engine for it and it appears in the playground and the OpenRouter <code>/models</code> document with these prices.</p>

      <div className="card" style={{ marginBottom: 18 }}>
        <div className="row">
          <label style={{ flex: 2 }}>Hugging Face id or link<input value={form.hf_id} onChange={set('hf_id')} placeholder="Qwen/Qwen3-30B-A3B  or  https://huggingface.co/Qwen/Qwen3-0.6B" /></label>
          <label>served id (optional)<input value={form.id} onChange={set('id')} placeholder="qwen3-30b-a3b" style={{ width: 160 }} /></label>
          <label>context<input type="number" value={form.context_length} onChange={set('context_length')} style={{ width: 90 }} /></label>
          <label>quant<select value={form.quant} onChange={set('quant')}><option>q8</option><option>q4</option><option>bf16</option></select></label>
        </div>
        <div className="row">
          <label>input $/M<input type="number" step="0.001" value={form.prompt_price_per_m} onChange={set('prompt_price_per_m')} style={{ width: 100 }} /></label>
          <label>output $/M<input type="number" step="0.001" value={form.completion_price_per_m} onChange={set('completion_price_per_m')} style={{ width: 100 }} /></label>
          <label>cache read $/M<input type="number" step="0.001" value={form.cached_price_per_m} onChange={set('cached_price_per_m')} style={{ width: 100 }} /></label>
          <button className="primary" onClick={add} disabled={busy === 'add' || !form.hf_id}><Download /> Add & download</button>
        </div>
      </div>
      {err && <p className="bad">{err}</p>}

      {data.models.map((m) => <ModelCard key={m.id} m={m} busy={busy} action={action} save={save} models={data.models} />)}
      {data.models.length === 0 && <p className="muted">no models yet</p>}
    </>
  )
}

function ModelCard({ m, busy, action, save, models }) {
  const [e, setE] = useState(m)
  useEffect(() => { setE((prev) => ({ ...m, prompt_price_per_m: prev.prompt_price_per_m ?? m.prompt_price_per_m, completion_price_per_m: prev.completion_price_per_m ?? m.completion_price_per_m, cached_price_per_m: prev.cached_price_per_m ?? m.cached_price_per_m, context_length: prev.context_length ?? m.context_length, quant: prev.quant ?? m.quant, draft: prev.draft ?? m.draft, device: prev.device ?? m.device })) }, [m])
  const set = (k) => (ev) => setE({ ...e, [k]: ev.target.value })
  const color = { running: 'ok', starting: 'muted', ready: '', downloading: 'muted', error: 'bad' }[m.status] || ''
  const dirty = ['prompt_price_per_m', 'completion_price_per_m', 'cached_price_per_m', 'context_length', 'quant', 'draft', 'device'].some((k) => String(e[k] ?? '') !== String(m[k] ?? ''))

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="row" style={{ marginBottom: 6 }}>
        <div style={{ flex: 1 }}>
          <b>{m.id}</b> <span className="muted">· {m.hf_id}</span>
          <div className="sub">
            <span className={color}>{m.status === 'starting' ? 'loading weights' : m.status}{m.port ? ` on :${m.port}` : ''}</span>
            {m.status === 'downloading' && <span className="muted"> · {(m.progress * 100).toFixed(1)}% ({gb(m.downloaded)} / {gb(m.total_bytes)})</span>}
            {m.status === 'starting' && <span className="muted"> · {((m.load_progress || 0) * 100).toFixed(0)}% loaded</span>}
            {m.status !== 'downloading' && m.total_bytes > 0 && <span className="muted"> · {gb(m.total_bytes)} on disk · {m.dir}</span>}
            {m.error && <span className="bad"> · {m.error}</span>}
          </div>
          {(m.status === 'downloading' || m.status === 'starting') && <div style={{ height: 6, background: '#eceae4', borderRadius: 3, marginTop: 6 }}>
            <div style={{ height: 6, width: `${((m.status === 'starting' ? m.load_progress : m.progress) || 0) * 100}%`, background: 'var(--accent)', borderRadius: 3, transition: 'width .4s' }} />
          </div>}
        </div>
        {m.status === 'ready' && <button className="primary" onClick={() => action(m.id, 'start')} disabled={!!busy}><Play /> Start engine</button>}
        {(m.status === 'running' || m.status === 'starting') && <button className="danger" onClick={() => action(m.id, 'stop')} disabled={!!busy}><Stop /> Stop</button>}
        {m.status === 'error' && <button className="primary" onClick={() => action(m.id, 'retry')} disabled={!!busy}><Refresh /> Retry download</button>}
        <button className="ghost" onClick={() => {
          if (confirm(`Remove ${m.id} and delete its files on disk?\n\n${m.dir}`)) action(m.id, 'delete')
        }} disabled={!!busy}><Trash /> Remove</button>
      </div>
      <div className="row" style={{ marginBottom: 0 }}>
        <label>input $/M<input type="number" step="0.001" value={e.prompt_price_per_m} onChange={set('prompt_price_per_m')} style={{ width: 90 }} /></label>
        <label>output $/M<input type="number" step="0.001" value={e.completion_price_per_m} onChange={set('completion_price_per_m')} style={{ width: 90 }} /></label>
        <label>cache read $/M<input type="number" step="0.001" value={e.cached_price_per_m} onChange={set('cached_price_per_m')} style={{ width: 90 }} /></label>
        <label>context<input type="number" value={e.context_length} onChange={set('context_length')} style={{ width: 90 }} /></label>
        <label>quant<select value={e.quant} onChange={set('quant')}><option>q8</option><option>q4</option><option>bf16</option></select></label>
        <label>device<select value={e.device || 'auto'} onChange={set('device')}><option value="auto">auto</option><option value="cpu">cpu</option><option value="gpu">gpu</option></select></label>
        <label>draft model (speculative)<select value={e.draft || ''} onChange={set('draft')}><option value="">none</option>{models.filter((x) => x.id !== m.id && x.status !== 'downloading' && x.status !== 'error').map((x) => <option key={x.id} value={x.dir}>{x.id}</option>)}</select></label>
        <button className="primary" onClick={() => save({ ...e, prompt_price_per_m: +e.prompt_price_per_m, completion_price_per_m: +e.completion_price_per_m, cached_price_per_m: +e.cached_price_per_m, context_length: +e.context_length })} disabled={!dirty}><Save /> Save{m.status === 'running' && dirty ? ' (restart to apply)' : ''}</button>
      </div>
    </div>
  )
}
