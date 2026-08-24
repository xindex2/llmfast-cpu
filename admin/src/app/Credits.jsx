// Credits are prepaid: we charge each request against the balance and return 402 at zero,
// rather than invoicing after the fact. No payment processor is connected yet, so this page
// is honest about how a top-up actually happens today.
import { cfg } from '../api'

export default function Credits({ me }) {
  const u = me.user
  return (
    <>
      <h2>Credits</h2>
      <p className="lede">Your balance is prepaid. Every request is charged at the model's per-token price; when the balance reaches zero the API returns <code>402 insufficient_quota</code> until you top up.</p>

      <div className="grid">
        <div className="card"><div className="label">Balance</div><div className="value">${u.credit_usd.toFixed(2)}</div>
          <div className="sub">{u.credit_usd <= 0 ? <span className="bad">API paused</span> : 'API active'}</div></div>
        <div className="card"><div className="label">Spent all time</div><div className="value">${u.spent_usd.toFixed(2)}</div>
          <div className="sub">since {new Date(u.created_at).toLocaleDateString()}</div></div>
        <div className="card"><div className="label">Account</div><div className="value" style={{ fontSize: 18 }}>{u.email}</div>
          <div className="sub mono">{u.id}</div></div>
      </div>

      <div className="card">
        <div className="label">Add credit</div>
        <p className="small" style={{ marginTop: 8 }}>
          Self-serve card payment is not connected yet. To top up, email <a href="mailto:billing@llmfa.st">billing@llmfa.st</a> with
          your account id <code className="mono">{u.id}</code> and the amount; credit is applied to this balance and appears here
          within a few minutes.
        </p>
        <p className="small muted" style={{ margin: 0 }}>
          Base URL for your clients: <code>{cfg.base || location.origin}/v1</code>
        </p>
      </div>
    </>
  )
}
