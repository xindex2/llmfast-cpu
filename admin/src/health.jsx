// Server health panel. Sits above the traffic numbers because when tok/s drops, the cause is
// almost always here: swap, a full disk, or engines competing for cores.
const gb = (n) => (n || 0).toFixed(n < 10 ? 1 : 0) + ' GB'
const mb2gb = (n) => gb((n || 0) / 1024)
// Below a gigabyte, "0.0 GB" is not a number anyone can act on. The alarm threshold matches
// the gateway's own warning (256 MB): a few idle megabytes of swap is normal on Linux and
// does not mean a model failed to fit.
const mem = (mb) => (mb >= 1024 ? gb(mb / 1024) : `${Math.round(mb || 0)} MB`)
const dur = (s) => {
  s = Math.max(0, s || 0)
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60)
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`
}

function Meter({ pct }) {
  const cls = pct >= 90 ? 'bad' : pct >= 75 ? 'warn' : ''
  return <div className={`meter ${cls}`}><span style={{ width: `${Math.min(100, Math.max(0, pct || 0))}%` }} /></div>
}

function Stat({ label, value, sub, pct }) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className="value" style={{ fontSize: 22 }}>{value}</div>
      {pct !== undefined && <Meter pct={pct} />}
      <div className="sub">{sub}</div>
    </div>
  )
}

export default function ServerHealth({ h }) {
  if (!h) return null
  const linux = h.mem_total_mb > 0
  return (
    <>
      {!!h.warnings?.length && (
        <div className="card warnbox">
          <div className="label">Server warnings</div>
          <ul>{h.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
        </div>
      )}

      <div className="health">
        <Stat label="CPU load" value={(h.load1 || 0).toFixed(2)} pct={h.load_pct}
          sub={`${(h.load_pct || 0).toFixed(0)}% of ${h.cores} threads · 5m ${(h.load5 || 0).toFixed(2)} · 15m ${(h.load15 || 0).toFixed(2)}`} />

        {linux && <Stat label="Memory" value={`${mb2gb(h.mem_used_mb)} / ${mb2gb(h.mem_total_mb)}`} pct={h.mem_pct}
          sub={h.swap_used_mb > 256
            ? <span className="bad">{mem(h.swap_used_mb)} swapped — model does not fit</span>
            : `${mb2gb(h.mem_free_mb)} available · ${h.swap_used_mb > 1 ? mem(h.swap_used_mb) + ' swap (idle pages)' : 'no swap in use'}`} />}

        <Stat label="Disk" value={`${gb(h.disk_used_gb)} / ${gb(h.disk_total_gb)}`} pct={h.disk_pct}
          sub={`${gb(h.disk_free_gb)} free · ${gb(h.models_gb)} of models`} />

        <Stat label="CPU" value={`${h.phys_cores} cores`}
          sub={`${h.cores} threads · ${h.sockets} socket${h.sockets > 1 ? 's' : ''}${h.cpu_model ? ' · ' + h.cpu_model.replace(/\(R\)|\(TM\)|CPU|@.*/g, '').trim() : ''}`} />

        <Stat label="Uptime" value={dur(h.gateway_up_sec)}
          sub={h.uptime_sec ? `gateway · host up ${dur(h.uptime_sec)}` : 'gateway'} />

        <Stat label="Host" value={h.hostname || '—'}
          sub={`${h.os}${h.gateway_mem_mb ? ` · gateway ${Math.round(h.gateway_mem_mb)} MB` : ''}`} />
      </div>

      {!!h.engine_procs?.length && (
        <div className="card" style={{ marginBottom: 22 }}>
          <div className="label">Engine processes</div>
          <table>
            <thead><tr><th>model</th><th>pid</th><th>resident memory</th><th>threads</th><th>cpu (lifetime avg)</th></tr></thead>
            <tbody>{h.engine_procs.map((e) => (
              <tr key={e.pid}>
                <td>{e.model}</td>
                <td className="muted">{e.pid}</td>
                <td>{mb2gb(e.rss_mb)}</td>
                <td>{e.threads || '—'}</td>
                <td>{e.cpu_pct ? (e.cpu_pct).toFixed(0) + '%' : '—'}</td>
              </tr>))}</tbody>
          </table>
        </div>
      )}
    </>
  )
}
