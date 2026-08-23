// Tokens are kept in localStorage so the admin can point this UI at any gateway/domain.
export const cfg = {
  get base() { return localStorage.getItem('forge.base') || '' },
  set base(v) { localStorage.setItem('forge.base', v) },
  get adminToken() { return localStorage.getItem('forge.admin') || 'admin' },
  set adminToken(v) { localStorage.setItem('forge.admin', v) },
  get apiKey() { return localStorage.getItem('forge.key') || 'dev-key' },
  set apiKey(v) { localStorage.setItem('forge.key', v) },
}

async function call(path, { method = 'GET', body, token } = {}) {
  const res = await fetch(cfg.base + path, {
    method,
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error((await res.json()).error?.message || res.statusText)
  return res.json()
}

export const api = {
  stats: (hours = 24) => call(`/admin/stats?hours=${hours}`, { token: cfg.adminToken }),
  keys: () => call('/admin/keys', { token: cfg.adminToken }),
  createKey: () => call('/admin/keys', { method: 'POST', token: cfg.adminToken }),
  benchmarks: () => call('/admin/benchmarks', { token: cfg.adminToken }),
  runBenchmark: (b) => call('/admin/benchmarks', { method: 'POST', body: b, token: cfg.adminToken }),
  models: () => call('/v1/models', { token: cfg.apiKey }),
  models_admin: () => call('/admin/models', { token: cfg.adminToken }),
  addModel: (m) => call('/admin/models', { method: 'POST', body: m, token: cfg.adminToken }),
  updateModel: (m) => call('/admin/models', { method: 'PUT', body: m, token: cfg.adminToken }),
  modelAction: (id, action, q = '') => call(`/admin/models/${encodeURIComponent(id)}/${action}${q}`, { method: 'POST', token: cfg.adminToken }),
}

// Streams a chat completion; onToken gets each delta, resolves with final usage.
export async function streamChat(req, onToken, signal) {
  const res = await fetch(cfg.base + '/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${cfg.apiKey}` },
    body: JSON.stringify({ ...req, stream: true }),
    signal,
  })
  if (!res.ok) throw new Error((await res.json()).error?.message || res.statusText)
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = '', usage = null
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop()
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const data = line.slice(6)
      if (data === '[DONE]') return usage
      const chunk = JSON.parse(data)
      if (chunk.usage) usage = chunk.usage
      const t = chunk.choices?.[0]?.delta?.content
      if (t) onToken(t)
    }
  }
  return usage
}
