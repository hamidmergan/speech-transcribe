"use strict";

const $ = (id) => document.getElementById(id);
const drop = $("drop");
const fileInput = $("file");
const chosen = $("chosen");
const startButton = $("start");
const progress = $("progress");
const bar = $("bar");
const progressText = $("progress-text");
const errorBox = $("error");
const result = $("result");
const textBox = $("text");

let info = { extensions: [], max_file_mb: 500, max_duration_min: 100 };
let picked = null;
let filename = "transcript.txt";
let polling = null;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function format() {
  const checked = document.querySelector('input[name="format"]:checked');
  return checked ? checked.value : "txt";
}

function setBusy(busy) {
  startButton.disabled = busy || !picked;
  progress.hidden = !busy;
  fileInput.disabled = busy;
}

async function loadInfo() {
  try {
    const response = await fetch("/api/info");
    info = await response.json();
  } catch (err) {
    showError("The server does not answer.");
    return;
  }
  $("extensions").textContent = info.extensions.join(", ");
  $("model").textContent = info.configured
    ? `${info.model} at ${info.base_url}`
    : "no model set, fill in the server fields below";
  if (info.language) $("language").value = info.language;
  if (info.configured) {
    $("base-url").placeholder = info.base_url;
    $("model-field").placeholder = info.model;
  }
}

function choose(file) {
  showError("");
  const name = file.name.toLowerCase();
  const ext = name.includes(".") ? name.slice(name.lastIndexOf(".")) : "";
  if (!info.extensions.includes(ext)) {
    showError(`Only ${info.extensions.join(", ")} allowed`);
    return;
  }
  if (file.size > info.max_file_mb * 1024 * 1024) {
    showError(`File too large (max ${info.max_file_mb} MB)`);
    return;
  }
  picked = file;
  chosen.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
  chosen.hidden = false;
  result.hidden = true;
  startButton.disabled = false;
}

drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) choose(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) choose(fileInput.files[0]);
  fileInput.value = "";
});

function describe(job) {
  const parts = [];
  if (job.language) parts.push(`Language ${job.language}`);
  if (job.duration) {
    const total = Math.round(job.duration);
    parts.push(`Length ${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`);
  }
  if (job.chunks > 1) parts.push(`${job.chunks} parts`);
  return parts.join(" . ");
}

function show(job) {
  filename = job.filename || "transcript.txt";
  textBox.value = job.text || "";
  $("about").textContent = `${filename} . ${describe(job)}`;
  result.hidden = false;
}

async function poll() {
  let job;
  try {
    job = await (await fetch("/api/job")).json();
  } catch (err) {
    return;
  }
  if (job.state === "running") {
    const known = job.total > 0;
    bar.parentElement.classList.toggle("waiting", !known);
    bar.style.width = known ? `${(job.current / job.total) * 100}%` : "100%";
    progressText.textContent = known
      ? `Part ${job.current} of ${job.total}`
      : "Transcribing, long recordings take a few minutes";
    return;
  }
  clearInterval(polling);
  polling = null;
  setBusy(false);
  if (job.state === "error") {
    showError(job.error);
    return;
  }
  if (job.state === "done") show(job);
}

startButton.addEventListener("click", async () => {
  if (!picked) return;
  showError("");
  result.hidden = true;
  setBusy(true);
  bar.style.width = "0";
  progressText.textContent = "Uploading";
  const form = new FormData();
  form.append("file", picked);
  form.append("response_format", format());
  form.append("language", $("language").value.trim());
  form.append("short_cues", $("short-cues").checked ? "true" : "false");
  form.append("base_url", $("base-url").value.trim());
  form.append("model", $("model-field").value.trim());
  form.append("api_key", $("api-key").value.trim());
  try {
    const response = await fetch("/api/transcribe", { method: "POST", body: form });
    if (!response.ok) {
      const body = await response.text();
      let message = body;
      try {
        message = JSON.parse(body).detail || body;
      } catch (ignored) {
        // a plain text answer from the guard, use it as it is
      }
      throw new Error(message || "The recording was refused.");
    }
  } catch (err) {
    setBusy(false);
    showError(err.message || "Unknown error. Please try again.");
    return;
  }
  progressText.textContent = "Transcribing";
  polling = setInterval(poll, 1500);
  poll();
});

$("download").addEventListener("click", () => {
  const blob = new Blob([textBox.value], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
});

$("again").addEventListener("click", () => {
  picked = null;
  chosen.hidden = true;
  result.hidden = true;
  startButton.disabled = true;
  showError("");
});

loadInfo();
