const fs = require('fs'), readline = require('readline');
const P = "C:/Users/TUSHAR/.claude/projects/c--Users-TUSHAR-OneDrive-Desktop-Trading/ee7ecda4-d9cc-4ab0-9118-ac57f4b6b3ef.jsonl";
const OUT = "C:/Users/TUSHAR/AppData/Local/Temp/claude/c--Users-TUSHAR-OneDrive-Desktop-Trading/ee7ecda4-d9cc-4ab0-9118-ac57f4b6b3ef/scratchpad/bash_patches.json";

// Only commands that WRITE to a project file. Everything else (pytest, ls,
// grep, curl) is noise for replay purposes.
const WRITES = [
  /io\.open\([^)]*["']w["']/,
  /\.write_text\(/,
  /sed -i /,
  /cat\s*>>?\s*[^|]*\.(py|md|toml|json|ts|tsx|css)/,
  /fs\.writeFileSync/,
];
const TOUCHES_PROJECT = /desk\/|desk\\\\|strategies\/|web\/|README|pyproject|\.gitignore/;

const out = [];
let seq = 0;
const rl = readline.createInterface({ input: fs.createReadStream(P, { encoding: 'utf8' }), crlfDelay: Infinity });
rl.on('line', (line) => {
  seq++;
  let rec; try { rec = JSON.parse(line); } catch (e) { return; }
  const msg = rec.message || {};
  const content = msg.content;
  if (!Array.isArray(content)) return;
  for (const c of content) {
    if (!c || c.type !== 'tool_use') continue;
    if (c.name !== 'Bash' && c.name !== 'PowerShell') continue;
    const cmd = (c.input || {}).command || '';
    if (!TOUCHES_PROJECT.test(cmd)) continue;
    if (!WRITES.some(r => r.test(cmd))) continue;
    out.push({ seq, tool: c.name, desc: (c.input || {}).description || '', command: cmd });
  }
});
rl.on('close', () => {
  fs.writeFileSync(OUT, JSON.stringify(out, null, 2));
  console.log('file-modifying shell commands found:', out.length);
  out.forEach(o => console.log(String(o.seq).padStart(5), '|', o.desc.slice(0, 70)));
});
