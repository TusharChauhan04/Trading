const fs = require('fs'), path = require('path'), readline = require('readline');
const P = "C:/Users/TUSHAR/.claude/projects/c--Users-TUSHAR-OneDrive-Desktop-Trading/ee7ecda4-d9cc-4ab0-9118-ac57f4b6b3ef.jsonl";
const ROOT = "C:/Users/TUSHAR/OneDrive/Desktop/Trading";
const OUT = "C:/Users/TUSHAR/Trading-RECOVERED";

const BACKSLASH = String.fromCharCode(92);

// Normalise: forward slashes, project-relative, and map the old `platform/`
// tree onto `desk/` (the package was renamed mid-project, so early writes to
// platform/risk/engine.py are the base for desk/risk/engine.py).
function norm(fp) {
  let s = fp.split(BACKSLASH).join('/');
  if (!s.toLowerCase().startsWith(ROOT.toLowerCase())) return null;
  let rel = s.slice(ROOT.length).replace(/^\//, '');
  if (!rel) return null;
  rel = rel.replace(/^platform\//, 'desk/');
  return rel;
}

// Read results arrive as "   12<TAB>line". Strip exactly that prefix.
function stripNums(t) {
  const lines = String(t).split('\n');
  const out = [];
  let ok = 0, bad = 0, first = null, last = null;
  const re = new RegExp('^\\s*(\\d+)\\t(.*)$');
  for (const ln of lines) {
    const m = ln.match(re);
    if (m) {
      const n = parseInt(m[1], 10);
      if (first === null) first = n;
      last = n;
      out.push(m[2]); ok++;
    }
    else if (ln.trim() === '') { out.push(''); }
    else { bad++; out.push(ln); }
  }
  // Trailing blank lines are an artefact of the wrapper, not the file.
  while (out.length && out[out.length - 1] === '') out.pop();
  return { text: out.join('\n'), ok, bad, first, last };
}

const ev = {};
let seq = 0;
const readIds = {};
const rl = readline.createInterface({ input: fs.createReadStream(P, { encoding: 'utf8' }), crlfDelay: Infinity });

rl.on('line', (line) => {
  seq++;
  let rec; try { rec = JSON.parse(line); } catch (e) { return; }
  const msg = rec.message || {};
  const content = msg.content;
  if (!Array.isArray(content)) return;
  for (const c of content) {
    if (!c) continue;
    if (c.type === 'tool_use') {
      const inp = c.input || {};
      const fp = inp.file_path;
      if (c.name === 'Read' && fp) readIds[c.id] = { fp: fp, limited: (inp.offset !== undefined || inp.limit !== undefined) };
      if (!fp) continue;
      const rel = norm(fp);
      if (!rel) continue;
      ev[rel] = ev[rel] || [];
      if (c.name === 'Write') ev[rel].push({ seq, kind: 'full', src: 'Write', text: inp.content || '' });
      else if (c.name === 'Edit') ev[rel].push({ seq, kind: 'edit', old: inp.old_string, neu: inp.new_string, all: !!inp.replace_all });
    }
    if (c.type === 'tool_result' && readIds[c.tool_use_id]) {
      const info = readIds[c.tool_use_id]; const rel = norm(info.fp);
      if (!rel) continue;
      let body = c.content;
      if (Array.isArray(body)) body = body.map(b => b.text || '').join('');
      body = String(body || '');
      if (!body || body.indexOf('<tool_use_error>') === 0) continue;
      const r = stripNums(body);
      if (r.ok < 3) continue;
      // A Read is only a usable BASE if it starts at line 1. Partial reads
      // (offset/limit, or `sed -n 'A,Bp'`-style spot checks) produce a
      // perfectly valid-looking numbered listing of the MIDDLE of a file -
      // treating one as a whole file silently truncates it, which is how a
      // 900-line main.py first came back as 5KB.
      // A Read that passed offset/limit is a WINDOW, not the file. It can start
      // at line 1 and still stop 20 lines in - that is how engine.py first came
      // back as 788 bytes.
      const partial = (r.first !== 1) || info.limited;
      const truncated = /lines truncated|too large to include/i.test(body);
      ev[rel] = ev[rel] || [];
      ev[rel].push({ seq, kind: 'full', src: 'Read', text: r.text, truncated, partial, lines: r.last });
    }
  }
});

rl.on('close', () => {
  const report = [];
  for (const rel of Object.keys(ev).sort()) {
    const list = ev[rel].slice().sort((a, b) => a.seq - b.seq);
    let baseIdx = -1;
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].kind === 'full' && !list[i].truncated && !list[i].partial) { baseIdx = i; break; }
    }
    if (baseIdx < 0) for (let i = list.length - 1; i >= 0; i--) if (list[i].kind === 'full' && !list[i].partial) { baseIdx = i; break; }
    if (baseIdx < 0) { report.push({ rel, status: 'NO BASE', events: list.length }); continue; }

    const base = list[baseIdx];
    let text = base.text;
    let applied = 0, failed = 0;
    for (let i = baseIdx + 1; i < list.length; i++) {
      const e = list[i];
      if (e.kind !== 'edit') continue;
      if (e.old === undefined) { failed++; continue; }
      if (text.indexOf(e.old) !== -1) {
        text = e.all ? text.split(e.old).join(e.neu) : text.replace(e.old, e.neu);
        applied++;
      } else failed++;
    }
    const dest = path.join(OUT, rel);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    if (!/\n$/.test(text)) text += '\n';
    fs.writeFileSync(dest, text, 'utf8');
    report.push({ rel, status: 'ok', base: base.src, baseSeq: base.seq, truncated: !!base.truncated, partialBase: !!base.partial, applied, failed, bytes: Buffer.byteLength(text) });
  }
  fs.mkdirSync(OUT, { recursive: true });
  fs.writeFileSync(path.join(OUT, '_RECOVERY_REPORT.json'), JSON.stringify(report, null, 2));
  const ok = report.filter(r => r.status === 'ok');
  const bad = report.filter(r => r.status !== 'ok' || r.failed > 0 || r.truncated);
  console.log('files written:', ok.length);
  console.log('needs attention:', bad.length);
  for (const r of bad) console.log('  !', r.rel, JSON.stringify(r));
});
