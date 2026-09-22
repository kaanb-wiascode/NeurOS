const grid = document.querySelector("#target-grid");
const focusSelect = document.querySelector("#focus-target");
const durationSelect = document.querySelector("#duration");
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const exportButton = document.querySelector("#export");
const statusNode = document.querySelector("#status");
const instructionNode = document.querySelector("#instruction");
const elapsedNode = document.querySelector("#elapsed");
const fpsNode = document.querySelector("#fps");
const framesNode = document.querySelector("#frames");

const response = await fetch("./targets.json", { cache: "no-store" });
if (!response.ok) throw new Error("Unable to load SSVEP target configuration.");
const { targets } = await response.json();

let targetElements = [];
let running = false;
let animationId = null;
let startedAt = 0;
let frameCount = 0;
let lastFpsAt = 0;
let lastFpsFrameCount = 0;
let session = null;

for (const target of targets) {
  const option = document.createElement("option");
  option.value = target.intent;
  option.textContent = target.label + " — " + target.frequency_hz + " Hz";
  focusSelect.append(option);

  const card = document.createElement("article");
  card.className = "target";
  card.dataset.intent = target.intent;
  card.innerHTML = "<div class=\"flash\"></div><div class=\"label\"><strong>" +
    target.label + "</strong><span>" + target.frequency_hz + " Hz</span></div>";
  grid.append(card);
  targetElements.push({ config: target, card, flash: card.querySelector(".flash") });
}

function updateFocusTarget() {
  const selected = focusSelect.value;
  for (const target of targetElements) {
    target.card.classList.toggle("focus-target", target.config.intent === selected);
  }
  const config = targets.find((target) => target.intent === selected);
  instructionNode.textContent = running
    ? "Keep your gaze on " + config.label + " (" + config.frequency_hz + " Hz) until the trial ends."
    : "Next trial: focus on " + config.label + " (" + config.frequency_hz + " Hz).";
}

function render(now) {
  if (!running) return;
  const elapsedSeconds = (now - startedAt) / 1000;
  frameCount += 1;

  for (const target of targetElements) {
    const phase = Math.sin(2 * Math.PI * target.config.frequency_hz * elapsedSeconds);
    target.flash.style.opacity = phase >= 0 ? "0.95" : "0.08";
  }

  elapsedNode.textContent = elapsedSeconds.toFixed(1) + " s";
  framesNode.textContent = String(frameCount);

  if (now - lastFpsAt >= 1000) {
    const frameDelta = frameCount - lastFpsFrameCount;
    const timeDeltaSeconds = (now - lastFpsAt) / 1000;
    fpsNode.textContent = (frameDelta / timeDeltaSeconds).toFixed(1);
    lastFpsAt = now;
    lastFpsFrameCount = frameCount;
  }

  if (elapsedSeconds >= session.duration_seconds) {
    stopSession("completed", now);
    return;
  }
  animationId = requestAnimationFrame(render);
}

function startSession() {
  if (running) return;
  const focus = targets.find((target) => target.intent === focusSelect.value);
  const durationSeconds = Number(durationSelect.value);

  running = true;
  frameCount = 0;
  startedAt = performance.now();
  lastFpsAt = startedAt;
  lastFpsFrameCount = 0;
  session = {
    schema_version: 1,
    started_at_iso: new Date().toISOString(),
    focus_intent: focus.intent,
    focus_frequency_hz: focus.frequency_hz,
    duration_seconds: durationSeconds,
    target_frequencies_hz: Object.fromEntries(
      targets.map((target) => [target.intent, target.frequency_hz]),
    ),
    completion: null,
    measured_display_fps: null,
    frames_rendered: null,
  };

  startButton.disabled = true;
  stopButton.disabled = false;
  exportButton.disabled = true;
  focusSelect.disabled = true;
  durationSelect.disabled = true;
  statusNode.textContent = "RUNNING";
  updateFocusTarget();
  animationId = requestAnimationFrame(render);
}

function stopSession(completion = "stopped", stoppedAt = performance.now()) {
  if (!running) return;
  running = false;
  if (animationId !== null) cancelAnimationFrame(animationId);

  const elapsedSeconds = Math.max((stoppedAt - startedAt) / 1000, 0.001);
  session.completion = completion;
  session.elapsed_seconds = elapsedSeconds;
  session.frames_rendered = frameCount;
  session.measured_display_fps = frameCount / elapsedSeconds;
  session.ended_at_iso = new Date().toISOString();

  for (const target of targetElements) target.flash.style.opacity = "0.08";

  startButton.disabled = false;
  stopButton.disabled = true;
  exportButton.disabled = false;
  focusSelect.disabled = false;
  durationSelect.disabled = false;
  statusNode.textContent = completion === "completed" ? "COMPLETE" : "STOPPED";
  elapsedNode.textContent = elapsedSeconds.toFixed(1) + " s";
  fpsNode.textContent = session.measured_display_fps.toFixed(1);
  updateFocusTarget();
}

function exportSession() {
  if (!session) return;
  const blob = new Blob([JSON.stringify(session, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "neuros-ssvep-" + session.focus_intent.toLowerCase() + "-" + Date.now() + ".json";
  document.body.append(link);
  link.click();
  URL.revokeObjectURL(link.href);
  link.remove();
}

focusSelect.addEventListener("change", updateFocusTarget);
startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", () => stopSession("stopped"));
exportButton.addEventListener("click", exportSession);
window.addEventListener("beforeunload", () => {
  if (animationId !== null) cancelAnimationFrame(animationId);
});

updateFocusTarget();
