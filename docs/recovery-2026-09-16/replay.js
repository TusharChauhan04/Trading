// Chronological reconstruction of the Trading project from the session
// transcript. Replays every Write, Edit and file-modifying shell command in
// the order they originally ran, so the result is the real final state rather
// than a snapshot of whichever event happened to be captured last.
const fs = require('fs'), path = require('path'), readline = require('readline');
const { execFileSync } = require('child_process');

const P = "C:/Users/TUSHAR/.claude/projects/c--Users-TUSHAR-OneDrive-Desktop-Trading/ee7ecda4-d9cc-4ab0-9118-ac57f4b6b3ef.jsonl";
const ROOT = "c:/users/tushar/onedrive/desktop/trading";
const DEST = "C:/Users/TUSHAR/Trading-RESTORED";
const PYEXE = "C:/Users/TUSHAR/AppData/Local/Python/pythoncore-3.14-64/python.exe";
const BASH = "C:/Program Files/Git/bin/bash.exe";
const BACKSLASH = String.fromCharCode(92);

function relOf(fp) {
  const s = fp.split(BACKSLASH).join('/');
  if (!s.toLowerCase().startsWith(ROOT)) return null;
  const rel = s.slice(ROOT.length).replace(/^\//, '');
  return rel || null;
}

// Rewrite a shell command so it operates on the restore tree with a Python
// that still exists. The original venv lived inside the OneDrive folder and
// went with it.
function rewrite(cmd) {
  let c = cmd;
  const variants = [
    "C:" + BACKSLASH + "Users" + BACKSLASH + "TUSHAR" + BACKSLASH + "OneDrive" + BACKSLASH + "Desktop" + BACKSLASH + "Trading",
    "c:" + BACKSLASH + "Users" + BACKSLASH + "TUSHAR" + BACKSLASH + "OneDrive" + BACKSLASH + "Desktop" + BACKSLASH + "Trading",
    "C:/Users/TUSHAR/OneDrive/Desktop/Trading",
    "c:/Users/TUSHAR/OneDrive/Desktop/Trading",
  ];
  for (const v of variants) c = c.split(v).join(DEST);
  // Strip any ./ or .\\ prefix FIRST, or the venv path becomes ./C:/...
  c = c.split("./.venv/Scripts/python.exe").join(PYEXE);
  c = c.split("."+BACKSLASH+".venv"+BACKSLASH+"Scripts"+BACKSLASH+"python.exe").join(PYEXE);
  c = c.split(".venv/Scripts/python.exe").join(PYEXE);
  c = c.split(".venv" + BACKSLASH + "Scripts" + BACKSLASH + "python.exe").join(PYEXE);
  // Bare `python` at the start of a pipeline / after && — the Store alias on
  // this machine is a stub that exits 49.
  c = c.replace(/(^|&&\s*|\|\s*|;\s*)python(\s+-\s*<<|\s+-c|\s+-)/g, "$1\"" + PYEXE + "\"$2");
  return c;
}

const TOUCHES = /desk\/|desk\\|platform|strategies\/|web\/|README|pyproject|\.gitignore|configs/;
const WRITES = [
  /io\.open\([^)]*["']w["']/, /\.write_text\(/, /sed -i /,
  /cat\s*>>?\s*[^|]*\.(py|md|toml|json|ts|tsx|css)/, /mv platform desk/,
  /mkdir/, /fs\.writeFileSync/,
];

const events = [];
let seq = 0;
const rl = readline.createInterface({ input: fs.createReadStream(P, { encoding: 'utf8' }), crlfDelay: Infinity });
rl.on('line', (line) => {
  seq++;
  let rec; try { rec = JSON.parse(line); } catch (e) { return; }
  const content = (rec.message || {}).content;
  if (!Array.isArray(content)) return;
  for (const c of content) {
    if (!c || c.type !== 'tool_use') continue;
    const inp = c.input || {};
    if (c.name === 'Write' && inp.file_path) {
      const rel = relOf(inp.file_path);
      if (rel) events.push({ seq, kind: 'write', rel, text: inp.content || '' });
    } else if (c.name === 'Edit' && inp.file_path) {
      const rel = relOf(inp.file_path);
      if (rel) events.push({ seq, kind: 'edit', rel, old: inp.old_string, neu: inp.new_string, all: !!inp.replace_all });
    } else if (c.name === 'Bash' && inp.command) {
      const cmd = inp.command;
      if (TOUCHES.test(cmd) && WRITES.some(r => r.test(cmd)) && !/Trading-RECOVERED|Trading-RESTORED|recover\.js|replay\.js/.test(cmd)) {
        events.push({ seq, kind: 'bash', cmd, desc: inp.description || '' });
      }
    }
  }
});

rl.on('close', () => {
  events.sort((a, b) => a.seq - b.seq);
  fs.rmSync(DEST, { recursive: true, force: true });
  fs.mkdirSync(DEST, { recursive: true });

  const log = [];
  let wrote = 0, edited = 0, editFail = 0, shellOk = 0, shellFail = 0;

  for (const e of events) {
    if (e.kind === 'write') {
      const dest = path.join(DEST, e.rel);
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.writeFileSync(dest, e.text, 'utf8');
      wrote++;
      log.push({ seq: e.seq, kind: 'write', rel: e.rel, ok: true });
    } else if (e.kind === 'edit') {
      const dest = path.join(DEST, e.rel);
      if (!fs.existsSync(dest)) { editFail++; log.push({ seq: e.seq, kind: 'edit', rel: e.rel, ok: false, why: 'file absent' }); continue; }
      let t = fs.readFileSync(dest, 'utf8');
      if (e.old === undefined || t.indexOf(e.old) === -1) { editFail++; log.push({ seq: e.seq, kind: 'edit', rel: e.rel, ok: false, why: 'old_string not found' }); continue; }
      t = e.all ? t.split(e.old).join(e.neu) : t.replace(e.old, e.neu);
      fs.writeFileSync(dest, t, 'utf8');
      edited++;
      log.push({ seq: e.seq, kind: 'edit', rel: e.rel, ok: true });
    } else {
      const sh = path.join(DEST, '_replay_step.sh');
      fs.writeFileSync(sh, rewrite(e.cmd), 'utf8');
      try {
        execFileSync(BASH, ['-lc', 'bash "' + sh + '"'], { stdio: 'pipe', timeout: 180000 });
        shellOk++; log.push({ seq: e.seq, kind: 'bash', desc: e.desc, ok: true });
      } catch (err) {
        shellFail++;
        const msg = ((err.stderr || '') + '').slice(-300);
        log.push({ seq: e.seq, kind: 'bash', desc: e.desc, ok: false, err: msg });
      }
    }
  }
  try { fs.unlinkSync(path.join(DEST, '_replay_step.sh')); } catch (e) {}
  fs.writeFileSync(path.join(DEST, '_REPLAY_LOG.json'), JSON.stringify(log, null, 2));
  console.log('writes:', wrote, ' edits ok:', edited, ' edits failed:', editFail);
  console.log('shell ok:', shellOk, ' shell failed:', shellFail);
  console.log('\n--- failures ---');
  for (const l of log) if (!l.ok) console.log(' ', l.seq, l.kind, l.rel || l.desc, '|', (l.why || l.err || '').replace(/\s+/g, ' ').slice(0, 160));
});
