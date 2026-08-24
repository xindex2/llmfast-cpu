// Tokens are kept in localStorage so the admin can point this UI at any gateway/domain.
// Settings saved under the old key are migrated once, so the rename does not sign anyone out.
for (const [old, now] of [['forge.base', 'llmfast.base'], ['forge.admin', 'llmfast.admin'], ['forge.key', 'llmfast.key']]) {
  const v = localStorage.getItem(old)
  if (v !== null && localStorage.getItem(now) === null) localStorage.setItem(now, v)
}

export const cfg = {
  get base() { return localStorage.getItem('llmfast.base') || '' },
  set base(v) { localStorage.setItem('llmfast.base', v) },
  get adminToken() { return localStorage.getItem('llmfast.admin') || 'admin' },
  set adminToken(v) { localStorage.setItem('llmfast.admin', v) },
  get apiKey() { return localStorage.getItem('llmfast.key') || 'dev-key' },
  set apiKey(v) { localStorage.setItem('llmfast.key', v) },
}

async function call(path, { method = 'GET', body, token } = {}) {
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers.Authorization = `Bearer ${token}`
  const res = await fetch(cfg.base + path, {
    method,
    headers,
    credentials: 'include', // session cookie for customer endpoints
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error((await res.json()).error?.message || res.statusText)
  return res.json()
}

export const api = {
  stats: (q = 'period=today') => call(`/admin/stats?${q}`, { token: cfg.adminToken }),
  users: () => call('/admin/users', { token: cfg.adminToken }),
  topup: (user_id, amount_usd) => call('/admin/topup', { method: 'POST', body: { user_id, amount_usd }, token: cfg.adminToken }),
  signup: (email, password) => call('/auth/signup', { method: 'POST', body: { email, password } }),
  login: (email, password) => call('/auth/login', { method: 'POST', body: { email, password } }),
  logout: () => call('/auth/logout', { method: 'POST' }),
  me: () => call('/auth/me'),
  myKeys: () => call('/account/keys'),
  createMyKey: (name) => call('/account/keys', { method: 'POST', body: { name } }),
  revokeMyKey: (prefix) => call(`/account/keys?prefix=${encodeURIComponent(prefix)}`, { method: 'DELETE' }),
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
export async function streamChat(req, onToken, onReasoning, signal) {
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
      const d = chunk.choices?.[0]?.delta || {}
      if (d.reasoning && onReasoning) onReasoning(d.reasoning)
      if (d.content) onToken(d.content)
    }
  }
  return usage
}
