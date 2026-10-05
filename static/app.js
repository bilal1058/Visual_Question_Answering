const $ = id => document.getElementById(id);

const state = {
  currentImage: null,
  taskId: null,
  result: null,
  cameraActive: false,
  voiceListening: false,
  chatHistory: [],
  _polling: false,
  _pollTimer: null,
  _pollCount: 0,
};

let cameraStream = null;
let livePreviewStream = null;
let currentCropIndex = null;
let speechRecognition = null;
let mediaRecorder = null;
let voiceAudioStream = null;
let voiceChunks = [];
let voiceSupported = false;
let speechSupported = false;
let recorderSupported = false;
let voiceSubmitTimer = null;
let voiceStopTimer = null;
let voiceHadTranscript = false;
let voiceLastError = "";
let voiceStartedAt = 0;
let preferServerTranscription = false;
let voiceStartingServerFallback = false;

const fileInput = $("fileInput");
const cameraBox = $("cameraBox");
const cameraPreview = $("cameraPreview");
const uploadToggle = $("uploadToggle");
const cameraToggle = $("cameraToggle");
const sourceInfo = $("sourceInfo");
const stage = $("stage");
const mainImage = $("mainImage");
const emptyState = $("emptyState");
const imageContainer = $("imageContainer");
const analyzeOverlay = $("analyzeOverlay");
const classesList = $("classesList");
const questionInput = $("questionInput");
const voiceBtn = $("voiceBtn");
const sendBtn = $("sendBtn");
const chatLog = $("chatLog");
const analyzeBtn = $("analyzeBtn");
const confSlider = $("confSlider");
const confVal = $("confVal");
const cropSection = $("cropSection");
const cropGrid = $("cropGrid");
const sceneHint = $("sceneHint");
const sceneSummary = $("sceneSummary");
const objectsMetric = $("objectsMetric");
const avgConfMetric = $("avgConfMetric");
const thresholdMetric = $("thresholdMetric");
const pipelineFill = $("pipelineFill");
const pipeline = $("pipeline");
const psSteps = {
  load: $("ps-load"),
  detect: $("ps-detect"),
  reason: $("ps-reason"),
  total: $("ps-total"),
};
const psMs = {
  load: $("ps-load-ms"),
  detect: $("ps-detect-ms"),
  reason: $("ps-reason-ms"),
  total: $("ps-total-ms"),
};
const stageOrder = ["load", "detect", "reason", "total"];
const stageTimers = [];
const shatterSpacingMs = 860;
let nextPipelineRevealAt = 0;

function updateConf(v) {
  confVal.textContent = v + "%";
  if (thresholdMetric) thresholdMetric.textContent = v + "%";
  const pct = Number(v);
  const min = Number(confSlider.min) || 0;
  const max = Number(confSlider.max) || 100;
  const fill = ((pct - min) / (max - min)) * 100;
  confSlider.style.setProperty("--fill", `${Math.max(0, Math.min(100, fill))}%`);
}

function pickFile() {
  setActiveSource("upload");
  fileInput.click();
}

function setActiveSource(source) {
  const isCamera = source === "camera";
  uploadToggle?.classList.toggle("hud-btn-primary", !isCamera);
  uploadToggle?.classList.toggle("is-active", !isCamera);
  uploadToggle?.setAttribute("aria-pressed", String(!isCamera));
  cameraToggle?.classList.toggle("hud-btn-primary", isCamera);
  cameraToggle?.classList.toggle("is-active", isCamera);
  cameraToggle?.setAttribute("aria-pressed", String(isCamera));
}

function onFilePicked(event) {
  const file = event.target.files[0];
  if (!file) return;

  stopCamera();
  const reader = new FileReader();
  reader.onload = () => {
    state.currentImage = reader.result;
    showImage(reader.result);
    sourceInfo.innerHTML =
      `<span class="chip-badge">IMG</span><div><div class="chip-name">${file.name}</div><div class="chip-size">${formatSize(file.size)}</div></div>`;
    sourceInfo.classList.remove("hidden");
    setActiveSource("upload");
    enableControls(true);
    resetDetection();
  };
  reader.readAsDataURL(file);
}

function toggleCamera() {
  if (state.cameraActive) {
    stopCamera();
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Camera is not available in this browser.");
    return;
  }

  navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => {
      state.cameraActive = true;
      cameraStream = stream;
      cameraPreview.srcObject = stream;
      cameraBox.classList.remove("hidden");
      setActiveSource("camera");
    })
    .catch(() => {
      stopCamera();
      alert("Camera access denied.");
    });
}

function captureCamera() {
  if (!state.cameraActive || !cameraPreview.srcObject) return;

  const canvas = document.createElement("canvas");
  canvas.width = cameraPreview.videoWidth || 640;
  canvas.height = cameraPreview.videoHeight || 480;
  canvas.getContext("2d").drawImage(cameraPreview, 0, 0);
  state.currentImage = canvas.toDataURL("image/jpeg", 0.92);
  showImage(state.currentImage);
  sourceInfo.innerHTML = `<span class="chip-badge">CAM</span><div><div class="chip-name">Camera capture</div></div>`;
  sourceInfo.classList.remove("hidden");
  setActiveSource("camera");
  enableControls(true);
  resetDetection();
  stopCamera();
}

function stopCamera() {
  cameraStream?.getTracks().forEach(track => track.stop());
  cameraStream = null;
  state.cameraActive = false;
  cameraPreview.pause();
  cameraPreview.srcObject = null;
  cameraBox.classList.add("hidden");
}

function showImage(src) {
  emptyState.classList.add("hidden");
  imageContainer.classList.remove("hidden");
  imageContainer.classList.remove("zoomed", "boxes-hidden");
  $("zoomBtn")?.classList.remove("is-active");
  $("boxesBtn")?.classList.remove("is-active");
  imageContainer.innerHTML = '<img id="mainImage" class="main-image">';
  const img = $("mainImage");
  if (img) img.src = src;
}

function enableControls(enabled) {
  questionInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  if (voiceBtn) {
    voiceBtn.disabled = !enabled || !voiceSupported;
    voiceBtn.title = voiceSupported ? "Ask by voice" : "Voice input is not supported in this browser";
  }
  analyzeBtn.disabled = !enabled;
  sceneHint.textContent = enabled ? "Ready. Click Analyze Scene." : "Upload an image to begin.";
}

function resetDetection() {
  stopVoiceInput();
  stopPolling();
  state.result = null;
  state.chatHistory = [];
  currentCropIndex = null;
  renderChat();
  cropSection.classList.add("hidden");
  imageContainer.innerHTML = '<img id="mainImage" class="main-image">';
  if (state.currentImage) {
    const img = $("mainImage");
    if (img) img.src = state.currentImage;
  }
  classesList.innerHTML = '<span class="placeholder">No detections yet</span>';
  updateSceneStats({}, []);
  if (sceneSummary) {
    sceneSummary.textContent = "No image analyzed yet. Upload or capture a frame, then run scene analysis.";
  }
  resetPipeline();
}

async function analyze() {
  if (!state.currentImage || state._polling) return;

  state._polling = true;
  state._pollCount = 0;
  analyzeOverlay.classList.remove("hidden");
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="mini-spinner"></span> Analyzing';
  resetPipeline();
  freezePipeline();
  setStepActive("load", true);

  const loadStart = performance.now();
  const response = await fetch(state.currentImage);
  const blob = await response.blob();
  setStageTiming("load", performance.now() - loadStart);
  const formData = new FormData();
  formData.append("file", blob, "image.jpg");
  formData.append("confidence", (confSlider.value / 100).toString());

  try {
    setStepActive("load", false);
    setStepDone("load", true);
    setStepActive("detect", true);

    const detectStart = performance.now();
    const detectResponse = await fetch("/detect", { method: "POST", body: formData });
    const data = await detectResponse.json();
    state.taskId = data.task_id;
    // Detection submitted — mark detect done as soon as backend returns task_id
    setStageTiming("detect", performance.now() - detectStart);
    setStepDone("detect", true);
    setStepActive("detect", false);
    setStepActive("reason", true);
    pollResult();
  } catch (_) {
    failAnalysis();
  }
}

function pollResult() {
  if (!state._polling) return;
  state._pollCount += 1;
  if (state._pollCount > 120) {
    failAnalysis();
    sceneHint.textContent = "Analysis timed out. Try again.";
    return;
  }

  fetch(`/detect/${state.taskId}`)
    .then(response => response.json())
    .then(result => {
      if (!state._polling) return;
      if (result.status === "completed") {
        completeAnalysis(result);
      } else if (result.status === "processing") {
        state._pollTimer = setTimeout(pollResult, 500);
      } else {
        failAnalysis();
      }
    })
    .catch(() => {
      state._pollTimer = setTimeout(pollResult, 500);
    });
}

function completeAnalysis(result) {
  state._polling = false;
  state.result = result;
  analyzeOverlay.classList.add("hidden");
  analyzeBtn.disabled = false;
  analyzeBtn.innerHTML = "Re-analyze";

  // Detect is already done (marked when POST /detect returned)
  // Mark reason done now that Groq response is back
  setStageTiming("reason", result.timing?.llm_ms || 0);
  setStepDone("reason", true);
  setStepActive("reason", false);
  setStepActive("total", true);
  setStageTiming("total", result.timing?.total_ms || 0);

  showDetectedImage(result.annotated, result.detections || []);
  showClasses(result.counts || {});
  showCrops(result.detections || []);
  updateSceneStats(result.counts || {}, result.detections || []);
  sceneHint.textContent = `Detected ${Object.values(result.counts || {}).reduce((sum, value) => sum + value, 0)} objects. Ask a question on the right.`;

  state.chatHistory = [];
  if (result.answer) {
    state.chatHistory.push({ role: "assistant", content: result.answer });
  }
  renderChat();

  // Total shatters after reason's shatter animation completes
  setTimeout(() => {
    setStepDone("total", true);
    setStepActive("total", false);
  }, 1200);
}

function failAnalysis() {
  state._polling = false;
  stopPolling();
  analyzeOverlay.classList.add("hidden");
  analyzeBtn.disabled = false;
  analyzeBtn.innerHTML = "Analyze";
  resetPipeline();
}

function stopPolling() {
  state._polling = false;
  if (state._pollTimer) {
    clearTimeout(state._pollTimer);
    state._pollTimer = null;
  }
}

function showDetectedImage(annotatedB64, detections) {
  imageContainer.innerHTML = `<img id="mainImage" src="data:image/jpeg;base64,${annotatedB64}" class="main-image">`;
  const img = $("mainImage");
  if (!detections.length || !img) return;

  img.onload = () => {
    const container = img.parentElement;
    const renderWidth = img.clientWidth;
    const renderHeight = img.clientHeight;
    const imageWidth = img.naturalWidth;
    const imageHeight = img.naturalHeight;
    const scale = renderWidth / imageWidth;
    const offsetX = (container.clientWidth - renderWidth) / 2;
    const offsetY = (container.clientHeight - renderHeight) / 2;
    const overlay = document.createElement("div");
    overlay.className = "bbox-container";

    detections.forEach(det => {
      const box = document.createElement("div");
      box.className = "bbox-box";
      box.style.left = `${offsetX + det.x1 * scale}px`;
      box.style.top = `${offsetY + det.y1 * scale}px`;
      box.style.width = `${(det.x2 - det.x1) * scale}px`;
      box.style.height = `${(det.y2 - det.y1) * scale}px`;
      box.innerHTML = `<div class="bbox-label">${cap(det.label)} ${Math.round(det.confidence * 100)}%</div>`;
      overlay.appendChild(box);
    });
    imageContainer.appendChild(overlay);
  };
}

function showClasses(counts) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  classesList.innerHTML = entries.length
    ? entries.map(([label, count]) => `<span class="tag">${cap(label)}<em>(${count})</em></span>`).join("")
    : '<span class="placeholder">No detections yet</span>';
}

function updateSceneStats(counts, detections) {
  const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
  const avg = detections.length
    ? Math.round((detections.reduce((sum, det) => sum + det.confidence, 0) / detections.length) * 100)
    : 0;

  if (objectsMetric) objectsMetric.textContent = total;
  if (avgConfMetric) avgConfMetric.textContent = detections.length ? `${avg}%` : "--";

  if (!sceneSummary) return;
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 4);
  if (!top.length) {
    sceneSummary.textContent = "Analysis complete, but no objects passed the current confidence threshold.";
    return;
  }
  const classText = top.map(([label, count]) => `${count} ${cap(label)}`).join(", ");
  sceneSummary.textContent = `${total} objects detected. Key classes: ${classText}. Average confidence is ${avg}%.`;
}

function showCrops(detections) {
  if (!detections.length) {
    cropSection.classList.add("hidden");
    return;
  }

  cropSection.classList.remove("hidden");
  currentCropIndex = null;
  cropGrid.innerHTML = detections.map((det, index) => {
    const percent = Math.round(det.confidence * 100) + "%";
    const extras = [];
    if (det.color && det.color !== "unknown") extras.push(det.color);
    if (det.size) extras.push(det.size);
    const details = extras.length ? " - " + extras.join(", ") : "";
    const cropSrc = det.crop_b64 ? `data:image/jpeg;base64,${det.crop_b64}` : buildCropPreview(state.result.annotated, det);
    return `<div class="crop-item" onclick="focusCrop(${index})" data-idx="${index}">
      <img src="${cropSrc}" alt="${cap(det.label)}">
      <div class="crop-info"><span>${cap(det.label)}</span><span>${percent}${details}</span></div>
    </div>`;
  }).join("");
}

function buildCropPreview(imageB64, det) {
  const img = new Image();
  img.src = `data:image/jpeg;base64,${imageB64}`;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const pad = 8;
  const x = Math.max(0, det.x1 - pad);
  const y = Math.max(0, det.y1 - pad);
  const w = Math.max(24, (det.x2 - det.x1) + pad * 2);
  const h = Math.max(24, (det.y2 - det.y1) + pad * 2);

  canvas.width = 160;
  canvas.height = 112;

  try {
    ctx.fillStyle = "#050607";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const scale = Math.min(canvas.width / w, canvas.height / h);
    const drawW = w * scale;
    const drawH = h * scale;
    const offsetX = (canvas.width - drawW) / 2;
    const offsetY = (canvas.height - drawH) / 2;
    ctx.drawImage(img, x, y, w, h, offsetX, offsetY, drawW, drawH);
    return canvas.toDataURL("image/jpeg", 0.9);
  } catch (_) {
    return `data:image/jpeg;base64,${imageB64}`;
  }
}

function zoomScene() {
  imageContainer.classList.add("zoomed");
  $("zoomBtn")?.classList.add("is-active");
}

function fitScene() {
  imageContainer.classList.remove("zoomed");
  $("zoomBtn")?.classList.remove("is-active");
}

function toggleBoxes() {
  imageContainer.classList.toggle("boxes-hidden");
  $("boxesBtn")?.classList.toggle("is-active", imageContainer.classList.contains("boxes-hidden"));
}

function focusCrop(index) {
  currentCropIndex = currentCropIndex === index ? null : index;
  document.querySelectorAll(".crop-item").forEach((element, idx) => {
    element.classList.toggle("active", idx === currentCropIndex);
  });
  if (currentCropIndex !== null && state.result?.detections?.[currentCropIndex]) {
    const det = state.result.detections[currentCropIndex];
    questionInput.value = `Tell me about the ${det.label} (${Math.round(det.confidence * 100)}% confidence)`;
    questionInput.focus();
  }
}

function resetPipeline() {
  stageTimers.splice(0).forEach(timer => clearTimeout(timer));
  nextPipelineRevealAt = 0;
  pipeline?.classList.remove("is-frozen", "is-unlocked");
  Object.values(psSteps).forEach(element => {
    element.classList.remove("active", "done", "frozen", "shattering", "complete");
  });
  Object.values(psMs).forEach(element => {
    if (element) element.textContent = "";
  });
  document.querySelectorAll(".pstep-sep").forEach(element => element.classList.remove("energized"));
  pipelineFill.style.width = "0";
}

function freezePipeline() {
  pipeline?.classList.add("is-frozen");
  pipeline?.classList.remove("is-unlocked");
  Object.values(psSteps).forEach(element => {
    element.classList.add("frozen");
    element.classList.remove("done", "shattering", "complete");
  });
  document.querySelectorAll(".pstep-sep").forEach(element => element.classList.remove("energized"));
  pipelineFill.style.width = "0";
}

function setStepActive(id, active) {
  psSteps[id]?.classList.toggle("active", active);
}

function setStepDone(id, done) {
  const step = psSteps[id];
  if (!step) return;
  if (!done) {
    step.classList.remove("done", "shattering", "complete");
    return;
  }

  step.classList.add("done");
  step.classList.remove("active");
  queuePipelineReveal(id);
}

function setStageTiming(id, ms) {
  if (psMs[id]) psMs[id].textContent = `${Math.max(0, Math.round(ms))} ms`;
}

function queuePipelineReveal(id) {
  const now = performance.now();
  const startAt = Math.max(now, nextPipelineRevealAt);
  nextPipelineRevealAt = startAt + shatterSpacingMs;

  const timer = setTimeout(() => revealPipelineStage(id), Math.max(0, startAt - now));
  stageTimers.push(timer);
}

function revealPipelineStage(id) {
  const step = psSteps[id];
  const index = stageOrder.indexOf(id);
  if (!step || index < 0) return;

  step.classList.add("shattering");
  pipelineFill.style.width = `${((index + 1) / stageOrder.length) * 100}%`;
  const connectors = Array.from(document.querySelectorAll(".pstep-sep"));
  if (connectors[index]) connectors[index].classList.add("energized");

  const timer = setTimeout(() => {
    step.classList.remove("frozen", "shattering");
    step.classList.add("complete");
    if (id === "total") {
      pipeline?.classList.remove("is-frozen");
      pipeline?.classList.add("is-unlocked");
    }
  }, 760);
  stageTimers.push(timer);
}

function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question || sendBtn.disabled) return;

  state.chatHistory.push({ role: "user", content: question });
  questionInput.value = "";
  renderChat();

  if (!state.result) {
    state.chatHistory.push({ role: "assistant", content: "Please analyze an image first." });
    renderChat();
    return;
  }

  if (currentCropIndex !== null && state.result.detections?.[currentCropIndex]) {
    askAboutCrop(question, state.result.detections[currentCropIndex]);
  } else {
    askVQA(question);
  }
}

async function askVQA(question) {
  try {
    const response = await fetch("/vqa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.taskId,
        question,
        history: state.chatHistory.slice(-6),
      }),
    });
    const result = await response.json();
    state.chatHistory.push({ role: "assistant", content: result.answer });
  } catch (_) {
    state.chatHistory.push({ role: "assistant", content: "VQA error. Retry." });
  }
  renderChat();
}

async function askAboutCrop(question, cropInfo) {
  try {
    const response = await fetch("/vqa/crop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.taskId,
        question,
        crop_index: currentCropIndex,
        history: state.chatHistory.slice(-6),
      }),
    });
    const result = await response.json();
    state.chatHistory.push({ role: "assistant", content: result.answer });
  } catch (_) {
    state.chatHistory.push({ role: "assistant", content: `Focus: ${cropInfo.label} - ${question}` });
  }
  renderChat();
}

function renderChat() {
  if (!state.chatHistory.length) {
    chatLog.innerHTML = '<div class="empty-chat">Analyze a scene to start asking questions.</div>';
    return;
  }

  const anchorId = "chat-end-" + Date.now();
  chatLog.innerHTML = state.chatHistory.map(message => {
    const content = esc(message.content);
    if (message.role === "user") {
      return `<div class="msg msg-user"><div class="msg-bubble">${content}</div><div class="msg-tag">YOU</div></div>`;
    }
    return `<div class="msg msg-ai"><div class="msg-bubble">${content}</div><div class="msg-tag">VQA</div></div>`;
  }).join("") + `<div id="${anchorId}"></div>`;

  setTimeout(() => {
    const anchor = $(anchorId);
    if (anchor) anchor.scrollIntoView({ behavior: "smooth" });
  }, 40);
}

function useTemplate(template) {
  questionInput.value = template;
  questionInput.focus();
  sendQuestion();
}

function initVoiceInput() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  speechSupported = Boolean(SpeechRecognition && window.isSecureContext);
  recorderSupported = Boolean(window.MediaRecorder && navigator.mediaDevices?.getUserMedia && window.isSecureContext);
  voiceSupported = speechSupported || recorderSupported;

  if (!voiceBtn) return;
  voiceBtn.disabled = questionInput.disabled || !voiceSupported;
  voiceBtn.title = speechSupported
    ? "Ask by voice"
    : recorderSupported
      ? "Ask by voice using server transcription"
    : "Voice input needs Chrome/Edge on localhost or HTTPS";

  if (!voiceSupported) {
    voiceBtn.classList.add("is-unsupported");
    return;
  }

  if (!speechSupported) {
    preferServerTranscription = true;
    return;
  }

  speechRecognition = new SpeechRecognition();
  speechRecognition.lang = navigator.language || "en-US";
  speechRecognition.continuous = false;
  speechRecognition.interimResults = true;
  speechRecognition.maxAlternatives = 1;

  speechRecognition.onstart = () => {
    state.voiceListening = true;
    voiceHadTranscript = false;
    voiceLastError = "";
    voiceStartedAt = performance.now();
    clearTimeout(voiceSubmitTimer);
    setVoiceButtonState("listening");
    questionInput.value = "";
    questionInput.focus();
    sceneHint.textContent = "Listening. Ask your question.";
  };

  speechRecognition.onresult = event => {
    let finalText = "";
    let interimText = "";
    for (let i = 0; i < event.results.length; i += 1) {
      const transcript = event.results[i][0]?.transcript?.trim();
      if (!transcript) continue;
      if (event.results[i].isFinal) {
        finalText += transcript + " ";
      } else {
        interimText += transcript + " ";
      }
    }

    const text = (finalText || interimText).trim();
    if (text) {
      voiceHadTranscript = true;
      questionInput.value = text;
      sceneHint.textContent = finalText.trim() ? "Voice captured. Sending question." : "Listening...";
    }

    const lastResult = event.results[event.results.length - 1];
    if (finalText.trim() && lastResult?.isFinal) {
      clearTimeout(voiceSubmitTimer);
      voiceSubmitTimer = setTimeout(() => {
        if (!sendBtn.disabled && questionInput.value.trim()) {
          sendQuestion();
        }
      }, 280);
    }
  };

  speechRecognition.onerror = event => {
    voiceLastError = event.error || "unknown";
    if (["network", "service-not-allowed"].includes(voiceLastError) && recorderSupported) {
      preferServerTranscription = true;
      voiceStartingServerFallback = true;
      sceneHint.textContent = "Browser speech service unavailable. Starting server voice mode.";
      setTimeout(() => startServerVoiceInput(), 180);
      return;
    }

    const messages = {
      "not-allowed": "Microphone permission changed. Click Allow/Done, then press Mic again.",
      "service-not-allowed": "Speech service blocked. Try Chrome/Edge on localhost.",
      "no-speech": "No speech detected. Click Mic and speak clearly.",
      "audio-capture": "No microphone found. Check your input device.",
      "network": "Voice recognition needs browser speech service access.",
      "aborted": "Voice input cancelled.",
    };
    const message = messages[voiceLastError] || `Voice input stopped (${voiceLastError}).`;
    sceneHint.textContent = message;
    setVoiceButtonState("error");
  };

  speechRecognition.onend = () => {
    const elapsed = performance.now() - voiceStartedAt;
    state.voiceListening = false;
    if (voiceStartingServerFallback) return;
    if (voiceLastError) {
      setVoiceButtonState("error");
      setTimeout(() => setVoiceButtonState("idle"), 1200);
      return;
    }

    setVoiceButtonState("idle");

    if (!voiceLastError && voiceHadTranscript && questionInput.value.trim() && !sendBtn.disabled) {
      clearTimeout(voiceSubmitTimer);
      voiceSubmitTimer = setTimeout(sendQuestion, 180);
      return;
    }

    if (!voiceLastError && !voiceHadTranscript && elapsed > 0) {
      sceneHint.textContent = elapsed < 800
        ? "Mic stopped immediately. Check browser permission and try again."
        : "No speech detected. Click Mic and ask again.";
    }
  };
}

async function toggleVoiceInput() {
  if (!voiceSupported || voiceBtn?.disabled) {
    sceneHint.textContent = "Voice input needs Chrome/Edge on localhost or HTTPS.";
    return;
  }

  if (mediaRecorder?.state === "recording") {
    stopServerVoiceInput();
    return;
  }

  if (state.voiceListening) {
    speechRecognition.stop();
    return;
  }

  if (preferServerTranscription || !speechRecognition) {
    await startServerVoiceInput();
    return;
  }

  try {
    speechRecognition.start();
  } catch (error) {
    const denied = error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError";
    sceneHint.textContent = denied
      ? "Microphone permission changed. Click Allow/Done, then press Mic again."
      : "Voice input is already starting or unavailable.";
    setVoiceButtonState("error");
    setTimeout(() => setVoiceButtonState("idle"), 1200);
  }
}

function setVoiceButtonState(mode) {
  if (!voiceBtn) return;
  const active = mode === "listening" || mode === "recording";
  voiceBtn.classList.toggle("is-listening", active);
  voiceBtn.classList.toggle("is-error", mode === "error");
  voiceBtn.classList.toggle("is-transcribing", mode === "transcribing");
  voiceBtn.textContent = active ? "Rec" : mode === "transcribing" ? "..." : "Mic";
  voiceBtn.setAttribute("aria-pressed", String(active));
}

function preferredAudioMimeType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return types.find(type => window.MediaRecorder?.isTypeSupported(type)) || "";
}

function voiceFileExtension(mimeType) {
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "m4a";
  return "webm";
}

async function startServerVoiceInput() {
  if (!recorderSupported) {
    sceneHint.textContent = "Server voice mode needs microphone recording support in this browser.";
    setVoiceButtonState("error");
    setTimeout(() => setVoiceButtonState("idle"), 1200);
    return;
  }

  try {
    voiceStartingServerFallback = false;
    voiceChunks = [];
    voiceAudioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const mimeType = preferredAudioMimeType();
    mediaRecorder = new MediaRecorder(voiceAudioStream, mimeType ? { mimeType } : undefined);

    mediaRecorder.ondataavailable = event => {
      if (event.data?.size) voiceChunks.push(event.data);
    };
    mediaRecorder.onstop = () => finishServerVoiceInput(mimeType || "audio/webm");
    mediaRecorder.start();

    clearTimeout(voiceStopTimer);
    voiceStopTimer = setTimeout(stopServerVoiceInput, 12000);
    setVoiceButtonState("recording");
    questionInput.value = "";
    questionInput.focus();
    sceneHint.textContent = "Recording voice. Ask your question, then click Rec to stop.";
  } catch (error) {
    voiceStartingServerFallback = false;
    const denied = error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError";
    sceneHint.textContent = denied
      ? "Microphone permission is blocked. Allow it from the address bar mic icon."
      : "Could not start microphone recording.";
    setVoiceButtonState("error");
    setTimeout(() => setVoiceButtonState("idle"), 1200);
  }
}

function stopServerVoiceInput() {
  clearTimeout(voiceStopTimer);
  if (mediaRecorder?.state === "recording") {
    sceneHint.textContent = "Transcribing voice...";
    setVoiceButtonState("transcribing");
    mediaRecorder.stop();
  }
}

async function finishServerVoiceInput(mimeType) {
  voiceAudioStream?.getTracks().forEach(track => track.stop());
  voiceAudioStream = null;

  if (!voiceChunks.length) {
    sceneHint.textContent = "No voice audio captured. Try again.";
    setVoiceButtonState("idle");
    return;
  }

  const blob = new Blob(voiceChunks, { type: mimeType });
  voiceChunks = [];
  const formData = new FormData();
  formData.append("audio", blob, `voice.${voiceFileExtension(mimeType)}`);

  try {
    const response = await fetch("/transcribe", { method: "POST", body: formData });
    const result = await response.json();
    if (!response.ok || !result.text) {
      throw new Error(result.error || "No speech recognized.");
    }
    questionInput.value = result.text;
    sceneHint.textContent = "Voice captured. Sending question.";
    setVoiceButtonState("idle");
    setTimeout(() => {
      if (!sendBtn.disabled && questionInput.value.trim()) sendQuestion();
    }, 180);
  } catch (error) {
    sceneHint.textContent = `Voice transcription failed: ${error.message}`;
    setVoiceButtonState("error");
    setTimeout(() => setVoiceButtonState("idle"), 1400);
  }
}

function stopVoiceInput() {
  clearTimeout(voiceSubmitTimer);
  clearTimeout(voiceStopTimer);
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
  }
  voiceAudioStream?.getTracks().forEach(track => track.stop());
  voiceAudioStream = null;
  if (!speechRecognition || !state.voiceListening) return;
  try {
    speechRecognition.abort();
  } catch (_) {
    // Browser speech recognition can throw while closing; the end handler resets UI.
  }
}

async function openStream() {
  $("streamModal").classList.remove("hidden");
  $("streamFrame").src = "";
  $("streamFrame").classList.add("hidden");
  $("browserStreamFrame").classList.add("hidden");
  $("browserStreamFrame").srcObject = null;
  $("streamInfo").textContent = "Checking camera...";

  try {
    stopLivePreviewStream();
    if (cameraStream) {
      stopCamera();
      await new Promise(resolve => setTimeout(resolve, 180));
    }

    const response = await fetch("/stream/status?camera_id=0");
    const status = await response.json();
    if (!status.ok) {
      await openBrowserStreamFallback(status.message);
      return;
    }
    const cameraId = Number.isInteger(status.camera_id) ? status.camera_id : 0;
    $("streamInfo").textContent = `Live stream connected on camera ${cameraId}.`;
    $("streamFrame").classList.remove("hidden");
    await new Promise(resolve => setTimeout(resolve, 180));
    $("streamFrame").src = `/stream?source=camera&camera_id=${cameraId}`;
  } catch (_) {
    await openBrowserStreamFallback("Camera not found or blocked.");
  }
}

function closeStream() {
  $("streamModal").classList.add("hidden");
  $("streamFrame").classList.add("hidden");
  $("streamFrame").src = "";
  $("browserStreamFrame").classList.add("hidden");
  $("browserStreamFrame").srcObject = null;
  stopLivePreviewStream();
  fetch("/stream/stop", { method: "POST" }).catch(() => {});
}

async function openBrowserStreamFallback(message) {
  if (!navigator.mediaDevices?.getUserMedia) {
    $("streamInfo").textContent = message || "Camera not found or blocked.";
    return;
  }

  try {
    livePreviewStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 854, height: 480 },
      audio: false,
    });
    const preview = $("browserStreamFrame");
    preview.srcObject = livePreviewStream;
    preview.classList.remove("hidden");
    $("streamInfo").textContent = "Browser live preview connected. AI stream is unavailable through OpenCV.";
  } catch (_) {
    $("streamInfo").textContent = message || "Camera not found or blocked.";
  }
}

function stopLivePreviewStream() {
  livePreviewStream?.getTracks().forEach(track => track.stop());
  livePreviewStream = null;
}

questionInput.addEventListener("keydown", event => {
  if (event.key === "Enter") sendQuestion();
});

initVoiceInput();
updateConf(confSlider.value);

const bmBtn = $("benchmarkBtn");
const benchmarkPanel = $("benchmark");
const bmProgress = $("benchmarkProgress");
const bmFill = $("benchmarkFill");
const bmMsg = $("benchmarkMsg");
const bmTable = $("benchmarkTable");
const bmNote = $("bmNote");
const bmBaseLabel = $("bmBaseLabel");
const bmTunedLabel = $("bmTunedLabel");
const benchmarkScope = $("benchmarkScope");

let bmRan = false;

async function runBenchmark() {
  if (bmBtn.disabled) return;

  bmBtn.disabled = true;
  bmBtn.innerHTML = '<span class="mini-spinner"></span> Running';
  benchmarkPanel?.classList.remove("has-results");
  benchmarkPanel?.classList.add("is-running");
  bmProgress.classList.remove("hidden");
  bmTable.classList.add("hidden");
  bmNote?.classList.add("hidden");
  bmFill.style.width = "0";
  bmMsg.textContent = "Preparing benchmark...";

  try {
    await fetch("/benchmark/start", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: "force=" + bmRan,
    });

    for (let i = 0; i < 1800; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const statusResponse = await fetch("/benchmark/status");
      const status = await statusResponse.json();
      const progress = status.progress || {};

      if (status.status === "completed" && status.results) {
        showBenchmarkResults(status.results);
        return;
      }

      const percent = progress.total
        ? Math.min(95, Math.round((progress.current / progress.total) * 100))
        : Math.min(5 + i * 0.05, 90);
      bmFill.style.width = percent + "%";
      bmMsg.textContent = progress.message || "Running benchmark...";
    }
    throw new Error("Timed out (30 min)");
  } catch (error) {
    bmMsg.textContent = "Error: " + error.message;
  } finally {
    bmBtn.disabled = false;
    bmBtn.innerHTML = "▶ Run";
  }
}

function showBenchmarkResults(data) {
  bmRan = true;
  benchmarkPanel?.classList.remove("is-running");
  benchmarkPanel?.classList.add("has-results");
  bmProgress.classList.add("hidden");
  bmTable.classList.remove("hidden");
  const m = v => (v ?? "-") + " ms";
  const fmt = v => v != null && v !== "-" ? typeof v === "number" ? v.toFixed(4) : v : "—";

  const b = data.comparison?.baseline || {};
  const f = data.comparison?.finetuned || {};
  const ba = b.accuracy || {};
  const fa = f.accuracy || {};
  const meta = data.meta || {};
  const baselineClasses = meta.baseline_classes || b.num_classes || ba.model_classes || (b.model === "yolov8n.pt" ? 80 : null);
  const tunedClasses = meta.finetuned_classes || f.num_classes || fa.model_classes || null;
  const scopeMismatch = meta.comparison_scope === "different-class-sets" || (baselineClasses && tunedClasses && baselineClasses !== tunedClasses);

  if (bmBaseLabel) bmBaseLabel.textContent = "Base";
  if (bmTunedLabel) bmTunedLabel.textContent = "Tuned";
  if (benchmarkScope) {
    benchmarkScope.textContent = scopeMismatch ? "Scope warn" : "Fair scope";
  }
  if (bmNote) {
    const fallbackNote = scopeMismatch
      ? "Not apples-to-apples: raw mAP uses different class sets."
      : "Same class count; verify validation data before ranking.";
    bmNote.textContent = meta.warning || fallbackNote;
    bmNote.classList.remove("hidden");
  }

  const set = (id, val) => { const el = $(id); if (el) el.textContent = val; };

  set("bm-b-images", b.num_images ?? "-");
  set("bm-f-images", f.num_images ?? "-");
  set("bm-b-fps", (b.avg_fps ?? "-"));
  set("bm-f-fps", (f.avg_fps ?? "-"));
  set("bm-b-lat", m(b.avg_latency_ms));
  set("bm-f-lat", m(f.avg_latency_ms));
  set("bm-b-p50", m(b.p50_latency_ms));
  set("bm-f-p50", m(f.p50_latency_ms));
  set("bm-b-p95", m(b.p95_latency_ms));
  set("bm-f-p95", m(f.p95_latency_ms));
  set("bm-b-p99", m(b.p99_latency_ms));
  set("bm-f-p99", m(f.p99_latency_ms));
  set("bm-b-time", (b.total_time_s ?? "-") + "s");
  set("bm-f-time", (f.total_time_s ?? "-") + "s");
  set("bm-b-dets", b.total_detections ?? "-");
  set("bm-f-dets", f.total_detections ?? "-");

  set("bm-b-mAP50", fmt(ba.mAP50));
  set("bm-f-mAP50", fmt(fa.mAP50));
  set("bm-b-mAP50_95", fmt(ba.mAP50_95));
  set("bm-f-mAP50_95", fmt(fa.mAP50_95));
  set("bm-b-precision", fmt(ba.precision));
  set("bm-f-precision", fmt(fa.precision));
  set("bm-b-recall", fmt(ba.recall));
  set("bm-f-recall", fmt(fa.recall));
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(2) + " MB";
}

function cap(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
