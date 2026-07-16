const DEFAULT_SETTINGS = {
  timeSec: 3,
  wallClockSeconds: 60,
  wholeGameTime: true,
  engineMode: "titanium-v17",
  catLmrCeiling: 1000,
  threads: 0,
  maxDepth: null,
  pawnDelayAvgSec: 0.2,
  wallDelayAvgSec: 1.2,
  endgamePawnDelayAvgSec: 0.2,
  endgameWallDelayAvgSec: 0.8,
  delayJitter: 0.4,
  complexityScaleSec: 1,
  humanizerEnabled: true,
  autoplayEnabled: false,
  enableVisuals: true,
  showGhost: true,
  maxGhosts: 3,
  showEvalBar: true,
  settingsVersion: 7,
};

const els = {
  autoplayEnabled: document.getElementById("autoplayEnabled"),
  wholeGameTime: document.getElementById("wholeGameTime"),
  wallClockSeconds: document.getElementById("wallClockSeconds"),
  wallClockVal: document.getElementById("wallClockVal"),
  wallClockField: document.getElementById("wallClockField"),
  timeSec: document.getElementById("timeSec"),
  timeSecVal: document.getElementById("timeSecVal"),
  timeSecField: document.getElementById("timeSecField"),
  threads: document.getElementById("threads"),
  threadsVal: document.getElementById("threadsVal"),
  maxDepth: document.getElementById("maxDepth"),
  maxDepthVal: document.getElementById("maxDepthVal"),
  pawnDelayAvg: document.getElementById("pawnDelayAvg"),
  pawnDelayAvgVal: document.getElementById("pawnDelayAvgVal"),
  wallDelayAvg: document.getElementById("wallDelayAvg"),
  wallDelayAvgVal: document.getElementById("wallDelayAvgVal"),
  endgamePawnDelayAvg: document.getElementById("endgamePawnDelayAvg"),
  endgamePawnDelayAvgVal: document.getElementById("endgamePawnDelayAvgVal"),
  endgameWallDelayAvg: document.getElementById("endgameWallDelayAvg"),
  endgameWallDelayAvgVal: document.getElementById("endgameWallDelayAvgVal"),
  delayJitter: document.getElementById("delayJitter"),
  delayJitterVal: document.getElementById("delayJitterVal"),
  complexityScale: document.getElementById("complexityScale"),
  complexityScaleVal: document.getElementById("complexityScaleVal"),
  disableHumanizer: document.getElementById("disableHumanizer"),
  humanizerFields: document.getElementById("humanizerFields"),
  enableVisuals: document.getElementById("enableVisuals"),
  showGhost: document.getElementById("showGhost"),
  maxGhosts: document.getElementById("maxGhosts"),
  maxGhostsVal: document.getElementById("maxGhostsVal"),
  showEvalBar: document.getElementById("showEvalBar"),
  exportLogsBtn: document.getElementById("exportLogsBtn"),
  status: document.getElementById("status"),
};

let saveTimer = null;
let settingsSnapshot = { ...DEFAULT_SETTINGS };
const logicalThreadCount = Math.max(
  1,
  Math.min(32, Number(navigator.hardwareConcurrency) || 8),
);
els.threads.max = String(logicalThreadCount);

function migrateSettings(raw) {
  const next = { ...DEFAULT_SETTINGS, ...(raw || {}) };
  const ver = Number(raw?.settingsVersion) || 0;
  if (!raw || ver < 2) {
    next.showEvalBar = true;
  }
  if (!raw || ver < 3) {
    next.wholeGameTime = true;
    if (raw?.wallClockSeconds == null) next.wallClockSeconds = 60;
    next.engineMode = "titanium-v17";
    next.catLmrCeiling = 1000;
    next.settingsVersion = 3;
  }
  if (!raw || ver < 4) {
    next.humanizerEnabled = true;
    next.settingsVersion = 4;
  }
  if (!raw || ver < 5) {
    next.maxGhosts = 3;
    next.settingsVersion = 5;
  }
  if (!raw || ver < 6) {
    next.threads = 0;
    next.settingsVersion = 6;
  }
  if (!raw || ver < 7) {
    // Old single delayAvgSec → wall; pawn defaults faster (old −1s hack ≈ 0.2).
    const legacyDelay = Number(raw?.delayAvgSec);
    const legacyEndgame = Number(raw?.endgameDelayAvgSec);
    next.wallDelayAvgSec = Number.isFinite(legacyDelay)
      ? legacyDelay
      : DEFAULT_SETTINGS.wallDelayAvgSec;
    next.pawnDelayAvgSec = DEFAULT_SETTINGS.pawnDelayAvgSec;
    next.endgameWallDelayAvgSec = Number.isFinite(legacyEndgame)
      ? legacyEndgame
      : DEFAULT_SETTINGS.endgameWallDelayAvgSec;
    next.endgamePawnDelayAvgSec = DEFAULT_SETTINGS.endgamePawnDelayAvgSec;
    next.settingsVersion = 7;
  }
  next.maxGhosts = Math.max(1, Math.min(8, Number(next.maxGhosts) || 3));
  next.threads = Math.max(0, Math.min(logicalThreadCount, Number(next.threads) || 0));
  next.pawnDelayAvgSec = Math.max(0, Number(next.pawnDelayAvgSec) || 0);
  next.wallDelayAvgSec = Math.max(0, Number(next.wallDelayAvgSec) || 0);
  next.endgamePawnDelayAvgSec = Math.max(0, Number(next.endgamePawnDelayAvgSec) || 0);
  next.endgameWallDelayAvgSec = Math.max(0, Number(next.endgameWallDelayAvgSec) || 0);
  return next;
}

function applyToForm(settings) {
  settingsSnapshot = migrateSettings(settings);
  els.autoplayEnabled.checked = settingsSnapshot.autoplayEnabled === true;
  els.wholeGameTime.checked = settingsSnapshot.wholeGameTime !== false;
  els.wallClockSeconds.value = String(settingsSnapshot.wallClockSeconds || 60);
  els.wallClockVal.textContent = `${settingsSnapshot.wallClockSeconds || 60}s`;
  els.timeSec.value = String(settingsSnapshot.timeSec);
  els.timeSecVal.textContent = `${settingsSnapshot.timeSec}s`;
  els.threads.value = String(settingsSnapshot.threads);
  els.threadsVal.textContent =
    settingsSnapshot.threads > 0
      ? String(settingsSnapshot.threads)
      : `Auto (${logicalThreadCount})`;
  els.maxDepth.value = String(settingsSnapshot.maxDepth || 0);
  els.maxDepthVal.textContent = settingsSnapshot.maxDepth ? String(settingsSnapshot.maxDepth) : "unlimited";
  els.pawnDelayAvg.value = String(settingsSnapshot.pawnDelayAvgSec);
  els.pawnDelayAvgVal.textContent = `${settingsSnapshot.pawnDelayAvgSec}s`;
  els.wallDelayAvg.value = String(settingsSnapshot.wallDelayAvgSec);
  els.wallDelayAvgVal.textContent = `${settingsSnapshot.wallDelayAvgSec}s`;
  els.endgamePawnDelayAvg.value = String(settingsSnapshot.endgamePawnDelayAvgSec);
  els.endgamePawnDelayAvgVal.textContent = `${settingsSnapshot.endgamePawnDelayAvgSec}s`;
  els.endgameWallDelayAvg.value = String(settingsSnapshot.endgameWallDelayAvgSec);
  els.endgameWallDelayAvgVal.textContent = `${settingsSnapshot.endgameWallDelayAvgSec}s`;
  els.delayJitter.value = String(Math.round(settingsSnapshot.delayJitter * 100));
  els.delayJitterVal.textContent = `${Math.round(settingsSnapshot.delayJitter * 100)}%`;
  els.complexityScale.value = String(settingsSnapshot.complexityScaleSec);
  els.complexityScaleVal.textContent = `${settingsSnapshot.complexityScaleSec}s/depth`;
  els.disableHumanizer.checked = settingsSnapshot.humanizerEnabled === false;
  els.enableVisuals.checked = settingsSnapshot.enableVisuals !== false;
  els.showGhost.checked = settingsSnapshot.showGhost !== false;
  els.maxGhosts.value = String(settingsSnapshot.maxGhosts);
  els.maxGhostsVal.textContent = String(settingsSnapshot.maxGhosts);
  els.showEvalBar.checked = settingsSnapshot.showEvalBar !== false;
  updateClockFields();
  updateHumanizerFields();
  updateVisualCheckboxes();
}

function readForm() {
  const maxDepth = Number(els.maxDepth.value);
  return {
    timeSec: Number(els.timeSec.value),
    wallClockSeconds: Number(els.wallClockSeconds.value),
    wholeGameTime: els.wholeGameTime.checked,
    engineMode: "titanium-v17",
    catLmrCeiling: 1000,
    threads: Math.max(0, Math.min(logicalThreadCount, Number(els.threads.value) || 0)),
    maxDepth: maxDepth > 0 ? maxDepth : null,
    pawnDelayAvgSec: Number(els.pawnDelayAvg.value),
    wallDelayAvgSec: Number(els.wallDelayAvg.value),
    endgamePawnDelayAvgSec: Number(els.endgamePawnDelayAvg.value),
    endgameWallDelayAvgSec: Number(els.endgameWallDelayAvg.value),
    delayJitter: Number(els.delayJitter.value) / 100,
    complexityScaleSec: Number(els.complexityScale.value),
    humanizerEnabled: !els.disableHumanizer.checked,
    autoplayEnabled: els.autoplayEnabled.checked,
    enableVisuals: els.enableVisuals.checked,
    showGhost: els.showGhost.checked,
    maxGhosts: Math.max(1, Math.min(8, Number(els.maxGhosts.value) || 3)),
    showEvalBar: els.showEvalBar.checked,
    settingsVersion: 7,
  };
}

function updateClockFields() {
  const whole = els.wholeGameTime.checked;
  els.wallClockField.classList.toggle("disabled", !whole);
  els.timeSecField.classList.toggle("disabled", whole);
  els.wallClockSeconds.disabled = !whole;
  els.timeSec.disabled = whole;
}

function updateHumanizerFields() {
  const disabled = els.disableHumanizer.checked;
  els.humanizerFields.classList.toggle("disabled", disabled);
  for (const el of [
    els.pawnDelayAvg,
    els.wallDelayAvg,
    els.endgamePawnDelayAvg,
    els.endgameWallDelayAvg,
    els.delayJitter,
    els.complexityScale,
  ]) {
    el.disabled = disabled;
  }
}

function updateLiveLabels() {
  els.wallClockVal.textContent = `${els.wallClockSeconds.value}s`;
  els.timeSecVal.textContent = `${els.timeSec.value}s`;
  const threads = Number(els.threads.value) || 0;
  els.threadsVal.textContent = threads > 0 ? String(threads) : `Auto (${logicalThreadCount})`;
  const md = Number(els.maxDepth.value);
  els.maxDepthVal.textContent = md > 0 ? String(md) : "unlimited";
  els.pawnDelayAvgVal.textContent = `${els.pawnDelayAvg.value}s`;
  els.wallDelayAvgVal.textContent = `${els.wallDelayAvg.value}s`;
  els.endgamePawnDelayAvgVal.textContent = `${els.endgamePawnDelayAvg.value}s`;
  els.endgameWallDelayAvgVal.textContent = `${els.endgameWallDelayAvg.value}s`;
  els.delayJitterVal.textContent = `${els.delayJitter.value}%`;
  els.complexityScaleVal.textContent = `${els.complexityScale.value}s/depth`;
  els.maxGhostsVal.textContent = els.maxGhosts.value;
  updateClockFields();
  updateHumanizerFields();
  updateVisualCheckboxes();
}

function updateVisualCheckboxes() {
  const enabled = els.enableVisuals.checked;
  els.showGhost.disabled = !enabled;
  els.showEvalBar.disabled = !enabled;
}

function save() {
  const settings = readForm();
  chrome.storage.local.set({ quoridorsBridgeSettings: settings }, () => {
    els.status.textContent = "saved";
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      els.status.textContent = "";
    }, 1200);
  });
}

function debouncedSave() {
  updateLiveLabels();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 250);
}

for (const el of [
  els.wallClockSeconds,
  els.timeSec,
  els.threads,
  els.maxDepth,
  els.maxGhosts,
  els.pawnDelayAvg,
  els.wallDelayAvg,
  els.endgamePawnDelayAvg,
  els.endgameWallDelayAvg,
  els.delayJitter,
  els.complexityScale,
]) {
  el.addEventListener("input", debouncedSave);
  el.addEventListener("change", debouncedSave);
}

els.wholeGameTime.addEventListener("change", () => {
  updateLiveLabels();
  clearTimeout(saveTimer);
  save();
});

els.autoplayEnabled.addEventListener("change", () => {
  clearTimeout(saveTimer);
  save();
});

for (const el of [
  els.enableVisuals,
  els.showGhost,
  els.showEvalBar,
  els.disableHumanizer,
]) {
  el.addEventListener("input", () => {
    updateLiveLabels();
    clearTimeout(saveTimer);
    save();
  });
  el.addEventListener("change", () => {
    updateLiveLabels();
    clearTimeout(saveTimer);
    save();
  });
}

chrome.storage.local.get("quoridorsBridgeSettings", (data) => {
  const migrated = migrateSettings(data?.quoridorsBridgeSettings);
  applyToForm(migrated);
  if (!data?.quoridorsBridgeSettings || Number(data.quoridorsBridgeSettings.settingsVersion) < 7) {
    chrome.storage.local.set({ quoridorsBridgeSettings: migrated });
  }
});

els.exportLogsBtn.addEventListener("click", async () => {
  els.exportLogsBtn.disabled = true;
  els.status.textContent = "exporting logs...";
  try {
    const result = await chrome.runtime.sendMessage({
      channel: "quoridors-bridge-api",
      action: "exportLogs",
    });
    els.status.textContent = result?.ok
      ? `exported ${result.count || 0} logs`
      : `export failed: ${result?.error || "unknown error"}`;
  } catch (err) {
    els.status.textContent = `export failed: ${err?.message || err}`;
  } finally {
    els.exportLogsBtn.disabled = false;
  }
});

try {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local" || !changes.quoridorsBridgeSettings) return;
    settingsSnapshot = migrateSettings(changes.quoridorsBridgeSettings.newValue);
  });
} catch {
  /* popup may close while storage listeners are settling */
}

// --- Debug: force move (bypasses client-side legality, tests server/bot validation) ---

const forceMoveText = document.getElementById("forceMoveText");
const forceMoveBtn = document.getElementById("forceMoveBtn");
const forceMoveOut = document.getElementById("forceMoveOut");

forceMoveBtn.addEventListener("click", async () => {
  const text = forceMoveText.value.trim();
  if (!text) return;
  forceMoveBtn.disabled = true;
  forceMoveOut.textContent = "sending…";
  try {
    const result = await chrome.runtime.sendMessage({
      channel: "quoridors-bridge-api",
      action: "playAlgebraic",
      text,
    });
    forceMoveOut.textContent = JSON.stringify(result, null, 2);
  } catch (err) {
    forceMoveOut.textContent = `error: ${err?.message || err}`;
  } finally {
    forceMoveBtn.disabled = false;
  }
});
