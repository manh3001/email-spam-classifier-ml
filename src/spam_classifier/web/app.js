const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function showBanner(msg) {
  const b = $("status-banner");
  b.textContent = msg;
  b.classList.remove("hidden");
}
function clearBanner() { $("status-banner").classList.add("hidden"); }

// Green (low spam prob) -> red (high). prob in [0,1].
function barColor(prob) {
  const hue = Math.round((1 - prob) * 120); // 120=green, 0=red
  return `hsl(${hue}, 70%, 45%)`;
}
function barHTML(prob) {
  const p = Math.min(1, Math.max(0, Number.isFinite(prob) ? prob : 0));
  const pct = Math.round(p * 100);
  return `<div class="bar"><span style="width:${pct}%;background:${barColor(p)}"></span></div>
          <small>${pct}% spam</small>`;
}
function labelTag(label) {
  const cls = label === "SPAM" ? "SPAM" : "HAM";
  return `<span class="label-tag label-${cls}">${esc(label)}</span>`;
}

async function postPredict(messages) {
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function loadMetrics() {
  try {
    const res = await fetch("/metrics");
    if (!res.ok) return; // panel stays hidden
    const m = await res.json();
    const fmt = (v) => (typeof v === "number" ? v.toFixed(3) : v);
    $("metrics-content").innerHTML = `
      <p><strong>Model:</strong> ${esc(m.best_model ?? "?")}</p>
      <p><strong>Spam F1:</strong> ${esc(fmt(m.spam_f1))} &nbsp;
         <strong>Precision:</strong> ${esc(fmt(m.spam_precision))} &nbsp;
         <strong>Recall:</strong> ${esc(fmt(m.spam_recall))}</p>`;
    $("metrics-panel").classList.remove("hidden");
  } catch (_) { /* leave hidden */ }
}

// Single
$("single-input").addEventListener("input", (e) => {
  $("single-btn").disabled = e.target.value.trim() === "";
});
$("single-btn").addEventListener("click", async () => {
  clearBanner();
  try {
    const [r] = await postPredict([$("single-input").value]);
    $("single-result").innerHTML =
      `<div class="result-card">${labelTag(r.label)} ${barHTML(r.spam_probability)}</div>`;
  } catch (err) { showBanner(err.message); }
});

// Batch
$("batch-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) $("batch-input").value = await file.text();
});
$("batch-btn").addEventListener("click", async () => {
  clearBanner();
  const lines = $("batch-input").value.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) { showBanner("Enter at least one message."); return; }
  try {
    const results = await postPredict(lines);
    const tbody = $("batch-result").querySelector("tbody");
    tbody.innerHTML = results.map((r) =>
      `<tr><td>${esc(r.text)}</td><td>${labelTag(r.label)}</td><td>${barHTML(r.spam_probability)}</td></tr>`
    ).join("");
    $("batch-result").classList.remove("hidden");
  } catch (err) { showBanner(err.message); }
});

loadMetrics();
