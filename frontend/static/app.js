const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const calibrateBtn = document.getElementById('calibrateBtn');
const resetBtn = document.getElementById('resetBtn');
const statusEl = document.getElementById('status');
const sessionStatusEl = document.getElementById('sessionStatus');
const gazeDot = document.getElementById('gazeDot');
const calibrationTarget = document.getElementById('calibrationTarget');
const textFileInput = document.getElementById('textFile');
const textDisplay = document.getElementById('textDisplay');
const startSessionBtn = document.getElementById('startSessionBtn');
const finishSessionBtn = document.getElementById('finishSessionBtn');
const cardScaleToggle = document.getElementById('cardScaleToggle');
const cardScalePanel = document.getElementById('cardScalePanel');
const cardWidthSlider = document.getElementById('cardWidthSlider');
const cardReference = document.getElementById('cardReference');
const applyCardScaleBtn = document.getElementById('applyCardScaleBtn');
const scaleStatusEl = document.getElementById('scaleStatus');

let stream = null;
let sending = false;
let loopHandle = null;
const frameFps = 17;
const frameIntervalMs = 1000 / frameFps;
const calibrationFrameCount = 12;
const calibrationFrameDelayMs = 40;
const calibrationMinValidFrames = 5;
const calibrationEdgeX = 0.04;
const calibrationEdgeY = 0.06;
const bankCardWidthCm = 8.56;
const bankCardHeightCm = 5.398;

let words = [];
let wordElements = [];
let activeHitElement = null;
let sessionActive = false;
let sessionStartMs = null;
let sessionRows = [];
let readWordIndexes = new Set();
let calibrationStep = 0;
let calibrationComplete = false;
let calibrationCapturing = false;
let calibratedViewport = null;
let physicalWorkspace = null;
const calibrationPoints = [
  { label: 'center', x: 0.5, y: 0.5 },
  { label: 'top left edge', x: calibrationEdgeX, y: calibrationEdgeY },
  { label: 'top edge', x: 0.5, y: calibrationEdgeY },
  { label: 'top right edge', x: 1 - calibrationEdgeX, y: calibrationEdgeY },
  { label: 'left edge', x: calibrationEdgeX, y: 0.5 },
  { label: 'right edge', x: 1 - calibrationEdgeX, y: 0.5 },
  { label: 'bottom left edge', x: calibrationEdgeX, y: 1 - calibrationEdgeY },
  { label: 'bottom edge', x: 0.5, y: 1 - calibrationEdgeY },
  { label: 'bottom right edge', x: 1 - calibrationEdgeX, y: 1 - calibrationEdgeY },
];

function workspaceWidth() {
  return window.innerWidth;
}

function workspaceHeight() {
  return window.innerHeight;
}

function drawFrameToCanvas() {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, width, height);
}

function canvasToBlob() {
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.75));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForTrackingIdle(timeoutMs = 750) {
  const start = performance.now();
  while (sending && performance.now() - start < timeoutMs) {
    await sleep(20);
  }
}

async function captureFrameBlob() {
  if (!stream || !video.videoWidth) return null;
  drawFrameToCanvas();
  return await canvasToBlob();
}

function getReadingRectPayload() {
  const rect = textDisplay.getBoundingClientRect();
  return {
    reading_rect_left: rect.left,
    reading_rect_top: rect.top,
    reading_rect_width: rect.width,
    reading_rect_height: rect.height,
  };
}

function getCalibrationPointCenterPx(point = currentCalibrationPoint()) {
  const rect = textDisplay.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    return {
      x: rect.left + rect.width * point.x,
      y: rect.top + rect.height * point.y,
    };
  }

  return {
    x: workspaceWidth() * point.x,
    y: workspaceHeight() * point.y,
  };
}

function appendCalibrationGeometry(form, targetCenter) {
  form.append('workspace_width', String(workspaceWidth()));
  form.append('workspace_height', String(workspaceHeight()));
  form.append('device_pixel_ratio', String(window.devicePixelRatio || 1));
  form.append('target_x_px', String(targetCenter.x));
  form.append('target_y_px', String(targetCenter.y));
  form.append('target_x_norm', String(targetCenter.x / workspaceWidth()));
  form.append('target_y_norm', String(targetCenter.y / workspaceHeight()));

  Object.entries(getReadingRectPayload()).forEach(([key, value]) => {
    form.append(key, String(value));
  });

  if (physicalWorkspace) {
    form.append('physical_workspace_width_cm', String(physicalWorkspace.widthCm));
    form.append('physical_workspace_height_cm', String(physicalWorkspace.heightCm));
    form.append('card_width_px', String(physicalWorkspace.cardWidthPx));
  }
}

async function sendCurrentFrame(endpoint, extraFields = {}, options = {}) {
  if (!stream || !video.videoWidth) return null;

  const blob = await captureFrameBlob();
  if (!blob) return null;

  const form = new FormData();
  form.append('frame', blob, 'frame.jpg');

  if (options.includeCalibrationGeometry) {
    appendCalibrationGeometry(form, options.targetCenter);
  }

  Object.entries(extraFields).forEach(([key, value]) => {
    form.append(key, String(value));
  });

  const response = await fetch(endpoint, { method: 'POST', body: form });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
}

async function sendCalibrationBatch(point) {
  const targetCenter = getCalibrationPointCenterPx(point);
  const form = new FormData();

  for (let i = 0; i < calibrationFrameCount; i += 1) {
    const blob = await captureFrameBlob();
    if (blob) {
      form.append('frames', blob, `calibration-${i}.jpg`);
    }
    if (i < calibrationFrameCount - 1) {
      await sleep(calibrationFrameDelayMs);
    }
  }

  appendCalibrationGeometry(form, targetCenter);
  form.append('min_valid_samples', String(calibrationMinValidFrames));

  const response = await fetch('/api/calibrate-point-batch', { method: 'POST', body: form });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
}

function renderResult(result, { trackWords = true } = {}) {
  if (result && result.valid && Number.isFinite(result.x) && Number.isFinite(result.y)) {
    gazeDot.style.display = 'block';
    gazeDot.style.left = `${result.x}px`;
    gazeDot.style.top = `${result.y}px`;
    if (trackWords) {
      trackWordHit(result.x, result.y);
    }
  } else {
    gazeDot.style.display = 'none';
    clearActiveHit();
  }
}

function currentCalibrationPoint() {
  return calibrationPoints[Math.min(calibrationStep, calibrationPoints.length - 1)];
}

function positionCalibrationTarget(point = currentCalibrationPoint()) {
  const targetCenter = getCalibrationPointCenterPx(point);
  calibrationTarget.style.left = `${targetCenter.x}px`;
  calibrationTarget.style.top = `${targetCenter.y}px`;
}

function showCalibrationTarget() {
  calibrationTarget.classList.remove('hidden');
  positionCalibrationTarget();
}

function hideCalibrationTarget() {
  calibrationTarget.classList.add('hidden');
}

function resetCalibrationUi() {
  calibrationStep = 0;
  calibrationComplete = false;
  calibrationCapturing = false;
  calibratedViewport = null;
  calibrateBtn.textContent = `Calibrate 1 / ${calibrationPoints.length}`;
  calibrationTarget.disabled = !stream;
  positionCalibrationTarget();
}

function updateCalibrationStatus() {
  if (calibrationComplete) {
    statusEl.textContent = '9-point text-area calibration complete';
    calibrateBtn.textContent = 'Calibrated';
    calibrateBtn.disabled = true;
    calibrationTarget.disabled = true;
    hideCalibrationTarget();
    return;
  }

  const point = currentCalibrationPoint();
  calibrateBtn.textContent = `Calibrate ${calibrationStep + 1} / ${calibrationPoints.length}`;
  statusEl.textContent = `Look at the ${point.label} target, then click it`;
  showCalibrationTarget();
}

function rememberCalibratedViewport() {
  const rect = textDisplay.getBoundingClientRect();
  calibratedViewport = {
    width: workspaceWidth(),
    height: workspaceHeight(),
    devicePixelRatio: window.devicePixelRatio || 1,
    readingRect: {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    },
  };
}

function calibrationViewportChanged() {
  if (!calibratedViewport) return false;
  const rect = textDisplay.getBoundingClientRect();
  return (
    Math.abs(workspaceWidth() - calibratedViewport.width) > 1 ||
    Math.abs(workspaceHeight() - calibratedViewport.height) > 1 ||
    Math.abs((window.devicePixelRatio || 1) - calibratedViewport.devicePixelRatio) > 0.01 ||
    Math.abs(rect.left - calibratedViewport.readingRect.left) > 1 ||
    Math.abs(rect.top - calibratedViewport.readingRect.top) > 1 ||
    Math.abs(rect.width - calibratedViewport.readingRect.width) > 1 ||
    Math.abs(rect.height - calibratedViewport.readingRect.height) > 1
  );
}

async function invalidateCalibrationIfViewportChanged() {
  if (!calibrationComplete || !calibrationViewportChanged()) return false;

  calibrationComplete = false;
  gazeDot.style.display = 'none';
  resetCalibrationUi();
  calibrateBtn.disabled = !stream;
  updateCalibrationStatus();
  statusEl.textContent = 'Window size changed. Please recalibrate.';
  await fetch('/api/reset', { method: 'POST' });
  return true;
}

function updateCardReference() {
  if (!cardWidthSlider || !cardReference) return;
  const stageWidth = cardReference.parentElement?.clientWidth || Number(cardWidthSlider.max);
  const widthPx = Math.min(Number(cardWidthSlider.value), stageWidth);
  cardReference.style.width = `${widthPx}px`;
  cardReference.style.height = `${widthPx * (bankCardHeightCm / bankCardWidthCm)}px`;
}

function applyCardScale() {
  const rect = cardReference.getBoundingClientRect();
  if (!rect.width || rect.width <= 0) return;

  const pxPerCm = rect.width / bankCardWidthCm;
  physicalWorkspace = {
    pxPerCm,
    cardWidthPx: rect.width,
    widthCm: workspaceWidth() / pxPerCm,
    heightCm: workspaceHeight() / pxPerCm,
  };

  scaleStatusEl.textContent = `Scale applied: ${physicalWorkspace.widthCm.toFixed(1)} x ${physicalWorkspace.heightCm.toFixed(1)} cm viewport.`;

  if (calibrationComplete) {
    fetch('/api/reset', { method: 'POST' });
    resetCalibrationUi();
    calibrateBtn.disabled = !stream;
    updateCalibrationStatus();
    statusEl.textContent = 'Screen scale changed. Please recalibrate.';
  }
}

function clearActiveHit() {
  if (activeHitElement) {
    activeHitElement.classList.remove('hit');
    activeHitElement = null;
  }
}

function setActiveHit(element) {
  if (activeHitElement && activeHitElement !== element) {
    activeHitElement.classList.remove('hit');
  }
  activeHitElement = element;
  if (activeHitElement) {
    activeHitElement.classList.add('hit');
  }
}

function textDisplayOverflows() {
  return (
    textDisplay.scrollHeight > textDisplay.clientHeight + 1 ||
    textDisplay.scrollWidth > textDisplay.clientWidth + 1
  );
}

function renderWords() {
  wordElements = [];
  textDisplay.innerHTML = '';

  const fragment = document.createDocumentFragment();
  words.forEach((word, index) => {
    const span = document.createElement('span');
    span.className = 'word';
    if (readWordIndexes.has(index)) {
      span.classList.add('read');
    }
    span.dataset.index = String(index);
    span.textContent = word;
    fragment.appendChild(span);
    fragment.appendChild(document.createTextNode(index === words.length - 1 ? '' : ' '));
    wordElements.push(span);
  });

  textDisplay.appendChild(fragment);
}

function findWordAtPoint(x, y) {
  const element = document.elementFromPoint(x, y);
  return element ? element.closest('.word') : null;
}

function trackWordHit(x, y) {
  const wordElement = findWordAtPoint(x, y);
  if (!wordElement) {
    clearActiveHit();
    return;
  }

  setActiveHit(wordElement);

  if (!sessionActive) return;

  const index = Number(wordElement.dataset.index);
  if (!Number.isInteger(index) || readWordIndexes.has(index)) return;

  const now = new Date();
  const elapsedMs = Math.max(0, performance.now() - sessionStartMs);

  readWordIndexes.add(index);
  wordElement.classList.add('read');
  sessionRows.push({
    wordIndex: index,
    word: words[index],
    timeFromStartMs: Math.round(elapsedMs),
    worldTimestamp: now.toISOString(),
  });

  updateSessionStatus();
}

function renderText(text) {
  const tokens = text.match(/\S+/g) || [];
  words = tokens;
  wordElements = [];
  readWordIndexes.clear();
  sessionRows = [];
  sessionActive = false;
  sessionStartMs = null;
  finishSessionBtn.disabled = true;
  startSessionBtn.disabled = words.length === 0;

  textDisplay.innerHTML = '';

  if (words.length === 0) {
    textDisplay.innerHTML = '<p class="emptyText">The selected file did not contain readable text.</p>';
    sessionStatusEl.textContent = 'No reading session active.';
    return;
  }

  renderWords();
  invalidateCalibrationIfViewportChanged().catch(error => {
    statusEl.textContent = `Calibration reset error: ${error.message}`;
  });
  if (calibrationComplete) {
    hideCalibrationTarget();
  } else {
    showCalibrationTarget();
  }
  if (textDisplayOverflows()) {
    sessionStatusEl.textContent = `${words.length} words loaded. Text does not fit in the reading box.`;
  } else {
    sessionStatusEl.textContent = `${words.length} words loaded. Start a session when ready.`;
  }
}

function resetReadingState({ keepText = true } = {}) {
  sessionActive = false;
  sessionStartMs = null;
  sessionRows = [];
  readWordIndexes.clear();
  clearActiveHit();
  wordElements.forEach(element => {
    element.classList.remove('read', 'hit');
  });
  finishSessionBtn.disabled = true;
  startSessionBtn.disabled = words.length === 0;
  if (!keepText) {
    words = [];
    wordElements = [];
    textDisplay.innerHTML = '<p class="emptyText">Upload a text file to begin.</p>';
    startSessionBtn.disabled = true;
    calibrationTarget.classList.remove('hidden');
  } else if (words.length > 0) {
    renderWords();
  }
  updateSessionStatus();
}

function updateSessionStatus() {
  if (sessionActive) {
    sessionStatusEl.textContent = `Reading session active. ${sessionRows.length} / ${words.length} words logged.`;
  } else if (words.length > 0) {
    sessionStatusEl.textContent = `${sessionRows.length} / ${words.length} words logged.`;
  } else {
    sessionStatusEl.textContent = 'No reading session active.';
  }
}

function csvEscape(value) {
  const text = String(value ?? '');
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function buildCsv() {
  const headers = [
    'word_index',
    'word',
    'time_from_start_ms',
    'world_timestamp',
  ];
  const lines = [headers.join(',')];
  sessionRows.forEach(row => {
    lines.push([
      row.wordIndex,
      row.word,
      row.timeFromStartMs,
      row.worldTimestamp,
    ].map(csvEscape).join(','));
  });
  return `${lines.join('\n')}\n`;
}

function downloadCsv() {
  const csv = buildCsv();
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  link.href = url;
  link.download = `reading-session-${stamp}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function trackingLoop() {
  if (calibrationCapturing) return;
  if (sending) return;
  sending = true;
  try {
    if (await invalidateCalibrationIfViewportChanged()) return;
    const result = await sendCurrentFrame('/api/frame');
    if (result) {
      renderResult(result);
      statusEl.textContent = result.valid ? 'Tracking' : `Waiting: ${result.reason || 'not ready'}`;
    }
  } catch (error) {
    statusEl.textContent = `Error: ${error.message}`;
  } finally {
    sending = false;
  }
}

textFileInput.addEventListener('change', async event => {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    const text = await file.text();
    renderText(text);
    statusEl.textContent = `Loaded ${file.name}`;
  } catch (error) {
    statusEl.textContent = `File error: ${error.message}`;
  }
});

startBtn.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
    resetCalibrationUi();
    calibrateBtn.disabled = false;
    startBtn.disabled = true;
    updateCalibrationStatus();
    loopHandle = setInterval(trackingLoop, frameIntervalMs);
  } catch (error) {
    statusEl.textContent = `Camera error: ${error.message}`;
  }
});

async function captureCalibrationPoint() {
  if (calibrationCapturing) return;

  try {
    const point = currentCalibrationPoint();
    positionCalibrationTarget(point);
    calibrationCapturing = true;
    calibrateBtn.disabled = true;
    calibrationTarget.disabled = true;
    gazeDot.style.display = 'none';
    clearActiveHit();
    statusEl.textContent = `Hold your gaze on the ${point.label} target...`;
    await waitForTrackingIdle();
    await sleep(200);

    const result = await sendCalibrationBatch(point);

    renderResult(result, { trackWords: false });

    if (!result || result.valid === false || !result.debug?.calibration_point_added) {
      statusEl.textContent = `Calibration failed: ${result?.reason || 'unknown'}`;
      calibrateBtn.disabled = false;
      calibrationTarget.disabled = false;
      return;
    }

    calibrationStep += 1;
    calibrationComplete = calibrationStep >= calibrationPoints.length;
    if (calibrationComplete) {
      rememberCalibratedViewport();
    }
    updateCalibrationStatus();
  } catch (error) {
    statusEl.textContent = `Calibration error: ${error.message}`;
    calibrateBtn.disabled = false;
    calibrationTarget.disabled = false;
  } finally {
    calibrationCapturing = false;
    if (!calibrationComplete) {
      calibrateBtn.disabled = !stream;
      calibrationTarget.disabled = !stream;
    }
  }
}

calibrateBtn.addEventListener('click', captureCalibrationPoint);
calibrationTarget.addEventListener('click', captureCalibrationPoint);

resetBtn.addEventListener('click', async () => {
  await fetch('/api/reset', { method: 'POST' });
  gazeDot.style.display = 'none';
  resetCalibrationUi();
  calibrateBtn.disabled = !stream;
  updateCalibrationStatus();
  resetReadingState({ keepText: true });
});

cardScaleToggle.addEventListener('click', () => {
  const isHidden = cardScalePanel.hidden;
  cardScalePanel.hidden = !isHidden;
  cardScaleToggle.textContent = isHidden ? 'Hide screen scale' : 'Screen scale';
  updateCardReference();
});

cardWidthSlider.addEventListener('input', updateCardReference);
applyCardScaleBtn.addEventListener('click', applyCardScale);

startSessionBtn.addEventListener('click', () => {
  if (words.length === 0) {
    sessionStatusEl.textContent = 'Upload a text file before starting a reading session.';
    return;
  }

  resetReadingState({ keepText: true });
  sessionActive = true;
  sessionStartMs = performance.now();
  startSessionBtn.disabled = true;
  finishSessionBtn.disabled = false;
  updateSessionStatus();
});

finishSessionBtn.addEventListener('click', () => {
  if (!sessionActive) return;

  sessionActive = false;
  finishSessionBtn.disabled = true;
  startSessionBtn.disabled = words.length === 0;
  updateSessionStatus();
  downloadCsv();
});

resetCalibrationUi();
showCalibrationTarget();
updateCardReference();

window.addEventListener('resize', () => {
  invalidateCalibrationIfViewportChanged();
});

window.addEventListener('beforeunload', () => {
  if (loopHandle !== null) {
    clearInterval(loopHandle);
  }
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
});
