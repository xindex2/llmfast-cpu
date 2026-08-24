// Minimal markdown renderer for model output. Deliberately not a library: this text comes from
// a language model, so the safe move is to emit React elements from a known-small grammar rather
// than set innerHTML. Supports: fenced code, headings, lists, blockquotes, tables, hr,
// **bold**, *italic*, `code`, [links](url), and paragraphs.
import { Fragment } from 'react'

function inline(text, keyBase = 'i') {
  const out = []
  // one pass, longest-token-first so ** wins over *
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\n]+\*)|(\[[^\]]+\]\([^)\s]+\))/g
  let last = 0
  let m
  let n = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const t = m[0]
    const key = `${keyBase}-${n++}`
    if (t.startsWith('`')) out.push(<code key={key}>{t.slice(1, -1)}</code>)
    else if (t.startsWith('**') || t.startsWith('__')) out.push(<strong key={key}>{t.slice(2, -2)}</strong>)
    else if (t.startsWith('*')) out.push(<em key={key}>{t.slice(1, -1)}</em>)
    else {
      const [, label, href] = t.match(/\[([^\]]+)\]\(([^)\s]+)\)/)
      const safe = /^(https?:|mailto:|\/)/i.test(href) ? href : '#'
      out.push(<a key={key} href={safe} target="_blank" rel="noreferrer noopener">{label}</a>)
    }
    last = m.index + t.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

export default function Markdown({ text }) {
  if (!text) return null
  const lines = String(text).split('\n')
  const blocks = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    // fenced code
    if (/^\s*```/.test(line)) {
      const lang = line.replace(/^\s*```/, '').trim()
      const body = []
      i++
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++])
      i++
      blocks.push(<pre key={key++} className="md-code" data-lang={lang}><code>{body.join('\n')}</code></pre>)
      continue
    }
    // heading
    const h = line.match(/^(#{1,6})\s+(.*)$/)
    if (h) {
      const Tag = `h${Math.min(h[1].length + 2, 6)}` // h1 in output shouldn't outrank the page title
      blocks.push(<Tag key={key++} className="md-h">{inline(h[2], key)}</Tag>)
      i++
      continue
    }
    // horizontal rule
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
      blocks.push(<hr key={key++} className="md-hr" />)
      i++
      continue
    }
    // table: | a | b |  then |---|---|
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const cells = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
      const head = cells(line)
      i += 2
      const rows = []
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]))
      blocks.push(
        <table key={key++} className="md-table">
          <thead><tr>{head.map((c, x) => <th key={x}>{inline(c, `${key}-h${x}`)}</th>)}</tr></thead>
          <tbody>{rows.map((r, y) => <tr key={y}>{r.map((c, x) => <td key={x}>{inline(c, `${key}-${y}-${x}`)}</td>)}</tr>)}</tbody>
        </table>,
      )
      continue
    }
    // blockquote
    if (/^\s*>\s?/.test(line)) {
      const body = []
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ''))
      blocks.push(<blockquote key={key++} className="md-quote">{inline(body.join(' '), key)}</blockquote>)
      continue
    }
    // lists (one level; ordered or bullet)
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d+[.)]\s+/.test(line)
      const items = []
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
        items.push(lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, ''))
      }
      const List = ordered ? 'ol' : 'ul'
      blocks.push(<List key={key++} className="md-list">{items.map((t, x) => <li key={x}>{inline(t, `${key}-${x}`)}</li>)}</List>)
      continue
    }
    // blank
    if (!line.trim()) {
      i++
      continue
    }
    // paragraph: consume until a blank line or a new block starts
    const para = []
    while (i < lines.length && lines[i].trim() && !/^\s*(```|#{1,6}\s|>|\||([-*+]|\d+[.)])\s)/.test(lines[i])) {
      para.push(lines[i++])
    }
    blocks.push(<p key={key++} className="md-p">{inline(para.join(' '), key)}</p>)
  }

  return <Fragment>{blocks}</Fragment>
}
