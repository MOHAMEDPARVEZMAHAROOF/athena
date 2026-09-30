/* Athena frontend — vanilla JS */
const $ = (id) => document.getElementById(id);

// tabs
document.querySelectorAll("nav button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("tab-" + b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "library") loadDocs();
    if (b.dataset.tab === "progress") loadProgress();
    if (b.dataset.tab === "quiz") updateDue();
  });
});

async function refreshStatus() {
  const s = await (await fetch("/api/status")).json();
  $("backend-badge").textContent = (s.llm ? "● " : "○ ") + s.backend;
  $("backend-badge").title = s.llm ? "AI features enabled" : "No API key — retrieval/template mode";
  const due = await (await fetch("/api/quiz/due")).json();
  $("due-badge").textContent = due.due ? due.due + " review" + (due.due > 1 ? "s" : "") + " due" : "";
}
refreshStatus(); setInterval(refreshStatus, 30000);

// ---------------- study chat ----------------
function addMsg(text, who) {
  const d = document.createElement("div");
  d.className = "msg " + who;
  d.innerHTML = text.replace(/[\[【](\d+)[\]】]/g, '<span class="cite">[$1]</span>');
  $("chat-log").appendChild(d);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
  return d;
}
function renderSources(passages, citedIdx) {
  const list = $("sources-list");
  list.innerHTML = "";
  passages.forEach((p) => {
    const d = document.createElement("div");
    d.className = "src";
    d.innerHTML = `<b>[${p.n}] ${p.doc}</b>${p.text}`;
    list.appendChild(d);
  });
}
$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("chat-input").value.trim();
  if (!q) return;
  $("chat-input").value = "";
  addMsg(q, "user");
  const bubble = addMsg("…", "ai");
  const resp = await fetch("/api/chat", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: q}),
  });
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "", full = "", passages = [];
  bubble.textContent = "";
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const ev = (chunk.match(/^event: (.*)$/m) || [])[1];
      const data = JSON.parse((chunk.match(/^data: (.*)$/ms) || [])[1] || "{}");
      if (ev === "sources") { passages = data.passages; renderSources(passages); }
      if (ev === "delta") { full += data.text; bubble.innerHTML = full.replace(/[\[【](\d+)[\]】]/g, '<span class="cite">[$1]</span>'); $("chat-log").scrollTop = $("chat-log").scrollHeight; }
      if (ev === "done") {
        (data.cited || []).forEach((c) => {
          const el = [...$("sources-list").children][c.n - 1];
          if (el) el.style.borderColor = "#15803d";
        });
      }
      if (ev === "error") { bubble.textContent = data.message; }
    }
  }
});

// ---------------- quiz ----------------
let currentQ = null;
$("next-q").addEventListener("click", async () => {
  const r = await fetch("/api/quiz/next", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  if (!r.ok) { alert(await r.text()); return; }
  const q = await r.json();
  currentQ = q;
  $("quiz-empty").classList.add("hidden");
  $("quiz-card").classList.remove("hidden");
  $("q-topic").textContent = q.topic;
  $("q-diff").textContent = "difficulty " + q.difficulty;
  $("q-mode").textContent = q.mode === "llm" ? "AI-generated" : "template mode";
  $("q-review").textContent = q.review ? "↻ review" : "";
  $("q-review").style.display = q.review ? "" : "none";
  $("q-prompt").textContent = q.prompt;
  $("q-feedback").classList.add("hidden");
  const box = $("q-choices"); box.innerHTML = "";
  q.choices.forEach((c, i) => {
    const b = document.createElement("button");
    b.className = "choice"; b.textContent = "ABCD"[i] + ". " + c;
    b.onclick = () => answer(i);
    box.appendChild(b);
  });
  updateDue();
});
async function answer(i) {
  const r = await fetch("/api/quiz/answer", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question_id: currentQ.id, given: String(i)})});
  const res = await r.json();
  [...$("q-choices").children].forEach((b, j) => {
    b.disabled = true;
    if (res.correct && j === i) b.classList.add("right");
    if (!res.correct && j === i) b.classList.add("wrong");
  });
  const f = $("q-feedback");
  f.classList.remove("hidden", "ok", "no");
  f.classList.add(res.correct ? "ok" : "no");
  f.innerHTML = (res.correct ? "✓ Correct. " : "✗ Not quite. ") + res.explanation +
    `<div class="srcs">Topic rating now <b>${res.rating}</b>${res.sources && res.sources.length ? "<br>From your notes: “" + res.sources[0] + "…”" : ""}</div>`;
  refreshStatus();
}
async function updateDue() {
  const d = await (await fetch("/api/quiz/due")).json();
  $("due-badge").textContent = d.due ? d.due + " review" + (d.due > 1 ? "s" : "") + " due" : "";
}

// ---------------- library ----------------
$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("file", $("file-input").files[0]);
  fd.append("title", $("title-input").value);
  fd.append("caption", $("caption-input").value);
  const r = await fetch("/api/upload", {method: "POST", body: fd});
  if (!r.ok) { alert(await r.text()); return; }
  $("file-input").value = ""; $("title-input").value = ""; $("caption-input").value = "";
  loadDocs(); refreshStatus();
});
async function loadDocs() {
  const docs = await (await fetch("/api/documents")).json();
  const list = $("doc-list"); list.innerHTML = "";
  docs.forEach((d) => {
    const el = document.createElement("div");
    el.className = "doc";
    el.innerHTML = `<h4>${d.title}</h4><p>${d.kind} · ${d.chunks} passages · ${d.filename}</p>`;
    const b = document.createElement("button"); b.textContent = "Remove";
    b.onclick = async () => { await fetch("/api/documents/" + d.id, {method: "DELETE"}); loadDocs(); refreshStatus(); };
    el.appendChild(b); list.appendChild(el);
  });
  if (!docs.length) list.innerHTML = '<p class="muted">No documents yet. Upload your notes to begin.</p>';
}

// ---------------- progress ----------------
async function loadProgress() {
  const topics = await (await fetch("/api/topics")).json();
  const bars = $("topic-bars"); bars.innerHTML = "";
  topics.forEach((t) => {
    const pct = Math.max(2, Math.min(100, ((t.rating - 800) / 1200) * 100));
    const row = document.createElement("div");
    row.className = "trow";
    row.innerHTML = `<span>${t.topic}</span><div class="bar"><div class="fill" style="width:${pct}%"></div></div><b>${t.rating}</b>`;
    bars.appendChild(row);
  });
  if (!topics.length) bars.innerHTML = '<p class="muted">Answer quiz questions to build your mastery map.</p>';
  const s = await (await fetch("/api/status")).json();
  $("stats-row").innerHTML =
    `<div class="stat"><b>${s.docs}</b><span>documents</span></div>` +
    `<div class="stat"><b>${s.chunks}</b><span>indexed passages</span></div>` +
    `<div class="stat"><b>${s.attempts}</b><span>questions answered</span></div>` +
    `<div class="stat"><b>${s.due_reviews}</b><span>reviews due</span></div>`;
}
