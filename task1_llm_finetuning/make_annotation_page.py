"""
Builds results/qualitative/annotate.html: a click-to-label page for the 42 qualitative responses.
Open it in a browser, label, then press "Download CSV" and replace results/qualitative/annotations_42.csv.

    python task1_llm_finetuning/make_annotation_page.py
"""
import csv
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..")))
from task1_llm_finetuning.src.qualitative import ANNOTATION_COLUMNS, load_qualitative_prompts

QUALITATIVE_DIR = os.path.join(script_dir, "results", "qualitative")
SHEET = os.path.join(QUALITATIVE_DIR, "annotations_42.csv")
PAGE = os.path.join(QUALITATIVE_DIR, "annotate.html")
CONDITION_LABELS = {"baseline": "Baseline", "full_ft_625": "Full FT midpoint", "full_ft_1250": "Full FT final", "full_ft_l2sp_1250": "Full FT + L2-SP",
                    "lora_r4_1250": "LoRA-4", "lora_r16_1250": "LoRA-16", "oplora_r16_1250": "OPLoRA-16"}

with open(SHEET, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
prompts = {(p["question"], p["format"]): {"system": p["system"] or "", "user": p["user"]}
           for p in load_qualitative_prompts(os.path.join(script_dir, "configs", "first_plot_questions.yaml"))}
for row in rows:
    row["prompt"] = prompts[(row["question"], row["format"])]
    row["condition_label"] = CONDITION_LABELS[row["condition"]]

payload = json.dumps({"rows": rows, "columns": list(rows[0].keys() - {"prompt", "condition_label"}), "column_order": [c for c in rows[0] if c not in ("prompt", "condition_label")]}, ensure_ascii=False)
payload = payload.replace("</", "<\\/")

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qualitative Annotation</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#647089; --line:#dde1e8; --accent:#2f6fec; --done:#1f9d55; --chip:#eef1f6; --sel:#2f6fec; --seltext:#fff; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171d; --card:#1d2129; --text:#e6e9ef; --muted:#97a1b5; --line:#2e3440; --chip:#262b35; --accent:#6f9cff; --sel:#6f9cff; --seltext:#0d1117; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
header h1 { font-size:17px; margin:0 12px 0 0; }
.progress { flex:1; min-width:180px; height:8px; background:var(--chip); border-radius:6px; overflow:hidden; }
.progress div { height:100%; background:var(--done); width:0; transition:width .2s; }
button { font:inherit; border:1px solid var(--line); background:var(--card); color:var(--text); border-radius:8px; padding:6px 12px; cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1100px; margin:0 auto; padding:16px; }
.guide { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 16px; margin-bottom:16px; color:var(--muted); font-size:14px; }
.guide b { color:var(--text); }
.group { background:var(--card); border:1px solid var(--line); border-radius:12px; margin-bottom:22px; overflow:hidden; }
.group > h2 { margin:0; padding:12px 16px; font-size:16px; border-bottom:1px solid var(--line); }
.prompt { padding:10px 16px; background:var(--chip); font-size:13.5px; white-space:pre-wrap; border-bottom:1px solid var(--line); }
.prompt .label { font-weight:600; color:var(--muted); }
.item { padding:14px 16px; border-bottom:1px solid var(--line); }
.item:last-child { border-bottom:none; }
.item.complete { border-left:4px solid var(--done); }
.item h3 { margin:0 0 6px; font-size:15px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.tag { font-size:12px; padding:1px 8px; border-radius:10px; background:var(--chip); color:var(--muted); font-weight:500; }
.tag.warn { background:#f5a62333; color:#b9770e; }
.response { white-space:pre-wrap; font:13px/1.45 ui-monospace, Consolas, monospace; background:var(--chip); border-radius:8px; padding:10px; max-height:260px; overflow:auto; }
.scores { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:8px 20px; margin-top:10px; }
.metric .name { font-size:13px; color:var(--muted); margin-bottom:3px; }
.options { display:flex; flex-wrap:wrap; gap:4px; }
.options button { padding:4px 10px; font-size:13px; }
.options button.on { background:var(--sel); color:var(--seltext); border-color:var(--sel); }
.text-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }
@media (max-width:640px) { .text-row { grid-template-columns:1fr; } }
textarea { width:100%; font:inherit; font-size:13px; background:var(--card); color:var(--text); border:1px solid var(--line); border-radius:8px; padding:6px 8px; min-height:52px; resize:vertical; }
.small { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Task 1 qualitative annotation</h1>
  <div class="progress"><div id="bar"></div></div>
  <span id="count" class="small"></span>
  <button id="next">Next unlabelled</button>
  <button class="primary" id="download">Download CSV</button>
</header>
<main>
  <div class="guide">
    <b>How to label.</b> Each box is one question in one format, with all seven models' responses. Click a score for each measure; your clicks are saved in this browser automatically.
    To quote evidence, select text in a response and press <b>Use selection</b>. When finished, press <b>Download CSV</b> and replace <code>results/qualitative/annotations_42.csv</code> with it.<br>
    <b>Format compliance</b> and <b>coherence/relevance</b>: 1 (poor) to 5 (excellent). <b>Repetition</b>, <b>unnecessary maths framing</b> and <b>inappropriate content</b>: 0 none, 1 some, 2 severe.
  </div>
  <div id="groups"></div>
</main>
<script>
const DATA = __PAYLOAD__;
const METRICS = [
  {key:"format_compliance", name:"Format compliance", values:[1,2,3,4,5], hint:"1 ignores the requested format · 5 follows it exactly"},
  {key:"coherence_relevance", name:"Coherence / relevance", values:[1,2,3,4,5], hint:"1 incoherent or off-topic · 5 clear and on-topic"},
  {key:"repetition", name:"Repetition", values:[0,1,2], hint:"0 none · 1 some · 2 severe loops"},
  {key:"unnecessary_math_framing", name:"Unnecessary maths framing", values:[0,1,2], hint:"0 none · 1 some · 2 treats it as a maths problem"},
  {key:"inappropriate_content", name:"Inappropriate content", values:[0,1,2], hint:"0 none · 1 mild · 2 clearly harmful"},
];
const STORE = "task1-annotations-v1";
const rows = DATA.rows;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) {}
const keyOf = r => `${r.question}|${r.format}|${r.condition}`;
rows.forEach(r => {
  const s = saved[keyOf(r)];
  if (s) Object.assign(r, s);
});
function persist() {
  const out = {};
  rows.forEach(r => {
    out[keyOf(r)] = Object.fromEntries([...METRICS.map(m => m.key), "excerpt", "notes"].map(k => [k, r[k] ?? ""]));
  });
  try { localStorage.setItem(STORE, JSON.stringify(out)); } catch (e) {}
}
const isComplete = r => METRICS.every(m => r[m.key] !== "" && r[m.key] !== undefined && r[m.key] !== null);
function refreshProgress() {
  const done = rows.filter(isComplete).length;
  document.getElementById("bar").style.width = `${100 * done / rows.length}%`;
  document.getElementById("count").textContent = `${done} / ${rows.length} labelled`;
}
function render() {
  const container = document.getElementById("groups");
  const groups = {};
  rows.forEach(r => (groups[`${r.question}|${r.format}`] ??= []).push(r));
  Object.values(groups).forEach(items => {
    const first = items[0];
    const g = document.createElement("section");
    g.className = "group";
    g.innerHTML = `<h2>${first.question.replaceAll("_", " ")} · ${first.format}</h2>`;
    const p = document.createElement("div");
    p.className = "prompt";
    p.textContent = (first.prompt.system ? `System: ${first.prompt.system}\n\n` : "") + `User: ${first.prompt.user}`;
    g.appendChild(p);
    items.forEach(r => g.appendChild(renderItem(r)));
    container.appendChild(g);
  });
}
function renderItem(r) {
  const el = document.createElement("div");
  el.className = "item";
  el.id = keyOf(r);
  const h = document.createElement("h3");
  h.textContent = r.condition_label;
  if (String(r.truncated) === "True") { const t = document.createElement("span"); t.className = "tag warn"; t.textContent = "truncated at 512 tokens"; h.appendChild(t); }
  el.appendChild(h);
  const resp = document.createElement("div");
  resp.className = "response";
  resp.textContent = r.response;
  el.appendChild(resp);
  const scores = document.createElement("div");
  scores.className = "scores";
  METRICS.forEach(m => {
    const box = document.createElement("div");
    box.className = "metric";
    box.innerHTML = `<div class="name" title="${m.hint}">${m.name} <span class="small">(${m.hint})</span></div>`;
    const opts = document.createElement("div");
    opts.className = "options";
    m.values.forEach(v => {
      const b = document.createElement("button");
      b.textContent = v;
      if (String(r[m.key]) === String(v)) b.classList.add("on");
      b.onclick = () => {
        r[m.key] = String(r[m.key]) === String(v) ? "" : String(v);
        opts.querySelectorAll("button").forEach(x => x.classList.toggle("on", String(r[m.key]) === x.textContent));
        el.classList.toggle("complete", isComplete(r));
        persist(); refreshProgress();
      };
      opts.appendChild(b);
    });
    box.appendChild(opts);
    scores.appendChild(box);
  });
  el.appendChild(scores);
  const texts = document.createElement("div");
  texts.className = "text-row";
  ["excerpt", "notes"].forEach(field => {
    const wrap = document.createElement("div");
    const label = document.createElement("div");
    label.className = "small";
    label.textContent = field === "excerpt" ? "Short excerpt supporting your scores" : "Notes";
    const area = document.createElement("textarea");
    area.value = r[field] || "";
    area.oninput = () => { r[field] = area.value; persist(); };
    wrap.append(label, area);
    if (field === "excerpt") {
      const use = document.createElement("button");
      use.textContent = "Use selection";
      use.style.marginTop = "4px";
      use.onclick = () => { const s = String(window.getSelection()).trim(); if (s) { area.value = s.slice(0, 300); r.excerpt = area.value; persist(); } };
      wrap.appendChild(use);
    }
    texts.appendChild(wrap);
  });
  el.appendChild(texts);
  el.classList.toggle("complete", isComplete(r));
  return el;
}
function csvCell(v) {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}
document.getElementById("download").onclick = () => {
  const cols = DATA.column_order;
  const lines = [cols.join(",")].concat(rows.map(r => cols.map(c => csvCell(r[c])).join(",")));
  const blob = new Blob([lines.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotations_42.csv";
  a.click();
};
document.getElementById("next").onclick = () => {
  const r = rows.find(x => !isComplete(x));
  if (r) document.getElementById(keyOf(r)).scrollIntoView({behavior: "smooth", block: "center"});
};
render();
refreshProgress();
</script>
</body>
</html>
"""

with open(PAGE, "w", encoding="utf-8") as f:
    f.write(PAGE_HTML.replace("__PAYLOAD__", payload))
print(f"Wrote {PAGE} with {len(rows)} responses. Open it in your browser.")
