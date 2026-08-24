// Paged table body. Every table here grows without bound — request logs, users, benchmarks —
// so rendering all of it eventually janks the page and buries the newest rows. Pagination is
// client-side: the API already returns a bounded window, and this keeps the DOM small.
import { useEffect, useState } from 'react'

export function usePage(items, per = 15) {
  const [page, setPage] = useState(0)
  const pages = Math.max(1, Math.ceil(items.length / per))
  // Rows arrive on a poll; if the current page vanishes underneath us, fall back to the last.
  useEffect(() => { if (page >= pages) setPage(pages - 1) }, [pages, page])
  const from = page * per
  return { slice: items.slice(from, from + per), page, pages, setPage, from, per, total: items.length }
}

export function Pager({ page, pages, setPage, from, per, total, unit = 'rows' }) {
  if (total === 0) return null
  const to = Math.min(from + per, total)
  return (
    <div className="pager">
      <span className="muted small">
        {pages === 1 ? `${total} ${unit}` : `${from + 1}–${to} of ${total} ${unit}`}
      </span>
      {pages > 1 && (
        <div className="pager-btns">
          <button className="ghost small" onClick={() => setPage(0)} disabled={page === 0}>«</button>
          <button className="ghost small" onClick={() => setPage(page - 1)} disabled={page === 0}>Prev</button>
          <span className="muted small">{page + 1} / {pages}</span>
          <button className="ghost small" onClick={() => setPage(page + 1)} disabled={page >= pages - 1}>Next</button>
          <button className="ghost small" onClick={() => setPage(pages - 1)} disabled={page >= pages - 1}>»</button>
        </div>
      )}
    </div>
  )
}

// One-liner for the common case: a table whose rows are a plain array.
export function Paged({ items, per = 15, unit = 'rows', children }) {
  const p = usePage(items, per)
  return <>{children(p.slice)}<Pager {...p} unit={unit} /></>
}
