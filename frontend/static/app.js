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
const faceDistanceInput = document.getElementById('faceDistanceInput');
const applyDistanceBtn = document.getElementById('applyDistanceBtn');
const estimateDistanceBtn = document.getElementById('estimateDistanceBtn');
const cameraFovInput = document.getElementById('cameraFovInput');
const eyeWidthInput = document.getElementById('eyeWidthInput');
const distanceStatusEl = document.getElementById('distanceStatus');
const setupStatusEl = document.getElementById('setupStatus');
const trackingPolicyInputs = Array.from(document.querySelectorAll('input[name="trackingPolicy"]'));
const relaxedThresholdXInput = document.getElementById('relaxedThresholdX');
const relaxedThresholdYInput = document.getElementById('relaxedThresholdY');

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
let activeTrackingPolicy = 'freehand';
let activeRelaxedThresholdX = 0;
let activeRelaxedThresholdY = 0;
let nextSequentialWordIndex = 0;
let calibrationStep = 0;
let calibrationComplete = false;
let calibrationCapturing = false;
let calibratedViewport = null;
let physicalWorkspace = null;
let faceDistanceCm = null;
const calibrationRows = [
    calibrationEdgeY,
    0.25,
    0.5,
    0.75,
    1 - calibrationEdgeY,
];
const calibrationColumns = [
    {key: 'left', x: calibrationEdgeX},
    {key: 'center', x: 0.5},
    {key: 'right', x: 1 - calibrationEdgeX},
];
const calibrationPoints = [
    {label: 'center', x: 0.5, y: 0.5},
    ...calibrationRows.flatMap((y, rowIndex) => (
        calibrationColumns.map(column => ({
            label: `row ${rowIndex + 1} ${column.key}`,
            x: column.x,
            y,
        }))
    )).filter(point => !(point.x === 0.5 && point.y === 0.5)),
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

function setupComplete() {
    return Boolean(physicalWorkspace && Number.isFinite(faceDistanceCm));
}

function updateSetupStatus() {
    const needs = [];
    if (!physicalWorkspace) {
        needs.push('screen scale');
    }
    if (!Number.isFinite(faceDistanceCm)) {
        needs.push('face distance');
    }

    setupStatusEl.textContent = needs.length
        ? `Complete ${needs.join(' and ')} before calibration.`
        : 'Pre-calibration setup complete.';
}

function updateCalibrationControls() {
    const canCalibrate = Boolean(stream && setupComplete() && !calibrationComplete && !calibrationCapturing);
    calibrateBtn.disabled = !canCalibrate;
    calibrationTarget.disabled = !canCalibrate;
    estimateDistanceBtn.disabled = !stream;
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

    const response = await fetch(endpoint, {method: 'POST', body: form});
    if (!response.ok) {
        throw new Error(await response.text());
    }
    return await response.json();
}

async function applyFaceDistance(distanceCm) {
    const form = new FormData();
    form.append('face_distance_cm', String(distanceCm));
    const response = await fetch('/api/set-face-distance', {method: 'POST', body: form});
    if (!response.ok) {
        throw new Error(await response.text());
    }
    const result = await response.json();
    faceDistanceCm = Number(result.face_distance_cm);
    distanceStatusEl.textContent = `Face distance applied: ${faceDistanceCm.toFixed(1)} cm.`;
    updateSetupStatus();
    updateCalibrationStatus();
}

async function estimateFaceDistance() {
    if (!stream || !video.videoWidth) {
        throw new Error('Start the camera before estimating distance.');
    }

    const form = new FormData();
    for (let i = 0; i < 8; i += 1) {
        const blob = await captureFrameBlob();
        if (blob) {
            form.append('frames', blob, `distance-${i}.jpg`);
        }
        if (i < 7) {
            await sleep(40);
        }
    }

    form.append('horizontal_fov_deg', String(Number(cameraFovInput.value) || 78));
    form.append('real_eye_width_cm', String(Number(eyeWidthInput.value) || 9.5));

    const response = await fetch('/api/estimate-distance', {method: 'POST', body: form});
    if (!response.ok) {
        throw new Error(await response.text());
    }
    const result = await response.json();
    faceDistanceCm = Number(result.face_distance_cm);
    faceDistanceInput.value = faceDistanceCm.toFixed(0);
    distanceStatusEl.textContent = `Estimated face distance: ${faceDistanceCm.toFixed(1)} cm from ${result.sample_count} frame${result.sample_count === 1 ? '' : 's'}.`;
    updateSetupStatus();
    updateCalibrationStatus();
}

async function sendCalibrationBatch(point) {
    if (!setupComplete()) {
        throw new Error('Complete screen scale and face distance before calibration.');
    }

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

    const response = await fetch('/api/calibrate-point-batch', {method: 'POST', body: form});
    if (!response.ok) {
        throw new Error(await response.text());
    }
    return await response.json();
}

function renderResult(result, {trackWords = true} = {}) {
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
    updateCalibrationControls();
    positionCalibrationTarget();
}

function updateCalibrationStatus() {
    if (calibrationComplete) {
        statusEl.textContent = `${calibrationPoints.length}-point text-area calibration complete`;
        calibrateBtn.textContent = 'Calibrated';
        calibrateBtn.disabled = true;
        calibrationTarget.disabled = true;
        hideCalibrationTarget();
        return;
    }

    if (!setupComplete()) {
        calibrateBtn.textContent = 'Calibrate';
        statusEl.textContent = 'Complete pre-calibration setup first.';
        hideCalibrationTarget();
        updateCalibrationControls();
        return;
    }

    if (!stream) {
        calibrateBtn.textContent = 'Calibrate';
        statusEl.textContent = 'Start the camera before calibration.';
        hideCalibrationTarget();
        updateCalibrationControls();
        return;
    }

    const point = currentCalibrationPoint();
    calibrateBtn.textContent = `Calibrate ${calibrationStep + 1} / ${calibrationPoints.length}`;
    statusEl.textContent = `Look at the ${point.label} target, then click it`;
    updateCalibrationControls();
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
    updateCalibrationStatus();
    statusEl.textContent = 'Window size changed. Please recalibrate.';
    await fetch('/api/reset', {method: 'POST'});
    return true;
}

function updateCardReference() {
    if (!cardWidthSlider || !cardReference) return;
    const stage = cardReference.parentElement;
    const stageWidth = stage?.clientWidth || Number(cardWidthSlider.max);
    const stageHeight = stage?.clientHeight || Number(cardWidthSlider.max);
    const maxVisibleWidth = Math.min(stageWidth, stageHeight * (bankCardWidthCm / bankCardHeightCm));
    const widthPx = Math.min(Number(cardWidthSlider.value), maxVisibleWidth);
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
    updateSetupStatus();
    updateCalibrationStatus();

    if (calibrationComplete) {
        fetch('/api/reset', {method: 'POST'});
        resetCalibrationUi();
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

function selectedTrackingPolicy() {
    return trackingPolicyInputs.find(input => input.checked)?.value || 'freehand';
}

function clampThreshold(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(300, value));
}

function selectedRelaxedThresholds() {
    return {
        x: clampThreshold(Number(relaxedThresholdXInput.value)),
        y: clampThreshold(Number(relaxedThresholdYInput.value)),
    };
}

function updateTrackingPolicyControls() {
    const isRelaxed = selectedTrackingPolicy() === 'relaxed';
    relaxedThresholdXInput.disabled = !isRelaxed || sessionActive;
    relaxedThresholdYInput.disabled = !isRelaxed || sessionActive;
    trackingPolicyInputs.forEach(input => {
        input.disabled = sessionActive;
    });
}

function gazeWithinExpandedWord(x, y, wordElement, thresholdX, thresholdY) {
    const rect = wordElement.getBoundingClientRect();
    return (
        x >= rect.left - thresholdX &&
        x <= rect.right + thresholdX &&
        y >= rect.top - thresholdY &&
        y <= rect.bottom + thresholdY
    );
}

function currentSequentialWordElement() {
    return wordElements[nextSequentialWordIndex] || null;
}

function markWordRead(index) {
    const wordElement = wordElements[index];
    if (!wordElement || readWordIndexes.has(index)) return false;

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

    if (activeTrackingPolicy !== 'freehand') {
        nextSequentialWordIndex = index + 1;
    }

    updateSessionStatus();
    return true;
}

function trackWordHit(x, y) {
    const wordElement = findWordAtPoint(x, y);

    if (!sessionActive || activeTrackingPolicy === 'freehand') {
        if (!wordElement) {
            clearActiveHit();
            return;
        }

        setActiveHit(wordElement);

        if (!sessionActive) return;

        const index = Number(wordElement.dataset.index);
        if (!Number.isInteger(index)) return;
        markWordRead(index);
        return;
    }

    const nextWordElement = currentSequentialWordElement();
    if (!nextWordElement) {
        clearActiveHit();
        return;
    }

    const nextIndex = Number(nextWordElement.dataset.index);
    if (!Number.isInteger(nextIndex)) return;

    if (activeTrackingPolicy === 'strict') {
        if (wordElement === nextWordElement) {
            setActiveHit(nextWordElement);
            markWordRead(nextIndex);
        } else if (wordElement) {
            setActiveHit(wordElement);
        } else {
            clearActiveHit();
        }
        return;
    }

    if (gazeWithinExpandedWord(x, y, nextWordElement, activeRelaxedThresholdX, activeRelaxedThresholdY)) {
        setActiveHit(nextWordElement);
        markWordRead(nextIndex);
        return;
    }

    if (wordElement) {
        setActiveHit(wordElement);
    } else {
        clearActiveHit();
    }
}

function renderText(text) {
    const tokens = text.match(/\S+/g) || [];
    words = tokens;
    wordElements = [];
    readWordIndexes.clear();
    sessionRows = [];
    sessionActive = false;
    sessionStartMs = null;
    nextSequentialWordIndex = 0;
    finishSessionBtn.disabled = true;
    startSessionBtn.disabled = words.length === 0;
    updateTrackingPolicyControls();

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

function resetReadingState({keepText = true} = {}) {
    sessionActive = false;
    sessionStartMs = null;
    sessionRows = [];
    readWordIndexes.clear();
    nextSequentialWordIndex = 0;
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
    updateTrackingPolicyControls();
}

function updateSessionStatus() {
    if (sessionActive) {
        const policyText = activeTrackingPolicy === 'relaxed'
            ? `relaxed, ±${activeRelaxedThresholdX}px x / ±${activeRelaxedThresholdY}px y`
            : activeTrackingPolicy;
        sessionStatusEl.textContent = `Reading session active (${policyText}). ${sessionRows.length} / ${words.length} words logged.`;
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
    const blob = new Blob([csv], {type: 'text/csv;charset=utf-8'});
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
        stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        video.srcObject = stream;
        await video.play();
        resetCalibrationUi();
        startBtn.disabled = true;
        updateSetupStatus();
        updateCalibrationStatus();
        loopHandle = setInterval(trackingLoop, frameIntervalMs);
    } catch (error) {
        statusEl.textContent = `Camera error: ${error.message}`;
    }
});

async function captureCalibrationPoint() {
    if (calibrationCapturing) return;
    if (!setupComplete()) {
        updateCalibrationStatus();
        return;
    }

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

        renderResult(result, {trackWords: false});

        if (!result || result.valid === false || !result.debug?.calibration_point_added) {
            statusEl.textContent = `Calibration failed: ${result?.reason || 'unknown'}`;
            updateCalibrationControls();
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
        updateCalibrationControls();
    } finally {
        calibrationCapturing = false;
        if (!calibrationComplete) {
            updateCalibrationControls();
        }
    }
}

calibrateBtn.addEventListener('click', captureCalibrationPoint);
calibrationTarget.addEventListener('click', captureCalibrationPoint);

resetBtn.addEventListener('click', async () => {
    await fetch('/api/reset', {method: 'POST'});
    gazeDot.style.display = 'none';
    resetCalibrationUi();
    updateCalibrationStatus();
    resetReadingState({keepText: true});
});

cardScaleToggle.addEventListener('click', () => {
    const isHidden = cardScalePanel.hidden;
    cardScalePanel.hidden = !isHidden;
    cardScaleToggle.textContent = isHidden ? 'Hide screen scale' : 'Screen scale';
    updateCardReference();
});

cardWidthSlider.addEventListener('input', updateCardReference);
applyCardScaleBtn.addEventListener('click', applyCardScale);

applyDistanceBtn.addEventListener('click', async () => {
    try {
        await applyFaceDistance(Number(faceDistanceInput.value));
    } catch (error) {
        distanceStatusEl.textContent = `Distance error: ${error.message}`;
    }
});

estimateDistanceBtn.addEventListener('click', async () => {
    try {
        estimateDistanceBtn.disabled = true;
        distanceStatusEl.textContent = 'Estimating face distance...';
        await estimateFaceDistance();
    } catch (error) {
        distanceStatusEl.textContent = `Estimate error: ${error.message}`;
    } finally {
        updateCalibrationControls();
    }
});

startSessionBtn.addEventListener('click', () => {
    if (words.length === 0) {
        sessionStatusEl.textContent = 'Upload a text file before starting a reading session.';
        return;
    }

    activeTrackingPolicy = selectedTrackingPolicy();
    const relaxedThresholds = selectedRelaxedThresholds();
    activeRelaxedThresholdX = relaxedThresholds.x;
    activeRelaxedThresholdY = relaxedThresholds.y;
    resetReadingState({keepText: true});
    nextSequentialWordIndex = 0;
    sessionActive = true;
    sessionStartMs = performance.now();
    startSessionBtn.disabled = true;
    finishSessionBtn.disabled = false;
    updateTrackingPolicyControls();
    updateSessionStatus();
});

finishSessionBtn.addEventListener('click', () => {
    if (!sessionActive) return;

    sessionActive = false;
    finishSessionBtn.disabled = true;
    startSessionBtn.disabled = words.length === 0;
    updateTrackingPolicyControls();
    updateSessionStatus();
    downloadCsv();
});

trackingPolicyInputs.forEach(input => {
    input.addEventListener('change', updateTrackingPolicyControls);
});

resetCalibrationUi();
updateCardReference();
updateSetupStatus();
updateCalibrationStatus();
updateTrackingPolicyControls();

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
