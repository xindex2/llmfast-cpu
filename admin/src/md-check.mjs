// Render a markdown sample through the real component and assert the output shape.
import { renderToStaticMarkup } from 'react-dom/server'
import React from 'react'
import Markdown from './md.jsx'
const sample = `## Heading

Plain **bold** and *italic* and \`code\` and [a link](https://example.com).

- first item
- second **item**

1. one
2. two

> a quote

\`\`\`python
def f(x):
    return x * 2
\`\`\`

| a | b |
|---|---|
| 1 | 2 |

---
Trailing paragraph with <script>alert(1)</script> and [bad](javascript:alert(1)).`
const html = renderToStaticMarkup(React.createElement(Markdown, { text: sample }))
for (const [name, ok] of [
  ['heading', html.includes('<h4 class="md-h">Heading</h4>')],
  ['bold', html.includes('<strong>bold</strong>')],
  ['italic', html.includes('<em>italic</em>')],
  ['inline code', html.includes('<code>code</code>')],
  ['link', html.includes('href="https://example.com"')],
  ['bullet list', html.includes('<ul class="md-list">') && html.includes('<li>first item</li>')],
  ['ordered list', html.includes('<ol class="md-list">')],
  ['quote', html.includes('<blockquote')],
  ['code block', html.includes('<pre class="md-code"') && html.includes('return x * 2')],
  ['table', html.includes('<table class="md-table">') && html.includes('<th>a</th>')],
  ['hr', html.includes('<hr class="md-hr"/>')],
  ['script escaped', !html.includes('<script>')],
  ['javascript: link neutralised', !html.includes('href="javascript:')],
]) console.log(ok ? 'ok  ' : 'FAIL', name)
