/**
 * 录音 + 上传 + 跳转报告页
 */

const API_BASE = ""; // 同源,走 nginx 代理

// ===== 题目加载 =====

let currentPart = 1;
let currentQuestion = null;

document.querySelectorAll(".part-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    currentPart = parseInt(btn.dataset.part);
    document.querySelectorAll(".part-tab").forEach(b => {
      b.classList.remove("bg-purple-600", "text-white");
      b.classList.add("bg-slate-200", "text-slate-700");
    });
    btn.classList.remove("bg-slate-200", "text-slate-700");
    btn.classList.add("bg-purple-600", "text-white");
    loadQuestions();
  });
});

// 默认加载 Part 1
loadQuestions();

async function loadQuestions() {
  try {
    const res = await fetch(`${API_BASE}/api/questions?part=${currentPart}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const list = document.getElementById("question-list");
    list.innerHTML = "";
    for (const q of data.questions || []) {
      const card = document.createElement("div");
      card.className = "border border-slate-200 rounded p-4 hover:border-purple-500 cursor-pointer transition";
      const mins = Math.round((q.expected_duration_sec || 60) / 60);
      card.innerHTML = `
        <div class="flex justify-between items-start">
          <div>
            <h3 class="font-medium text-slate-900">${q.topic}</h3>
            <p class="text-slate-600 text-sm mt-1">${q.prompt}</p>
            ${q.cues ? `<p class="text-xs text-slate-400 mt-2">提示: ${q.cues.join(" | ")}</p>` : ""}
          </div>
          <span class="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">${mins} 分钟</span>
        </div>
      `;
      card.onclick = () => selectQuestion(q);
      list.appendChild(card);
    }
  } catch (err) {
    document.getElementById("question-list").innerHTML =
      `<div class="text-red-600">加载题目失败: ${err.message}</div>`;
  }
}

function selectQuestion(q) {
  currentQuestion = q;
  document.getElementById("record-section").classList.remove("hidden");
  document.getElementById("current-question").textContent = q.prompt;
  document.getElementById("current-cues").textContent =
    q.cues ? "提示: " + q.cues.join(" / ") : "";
  document.getElementById("record-section").scrollIntoView({ behavior: "smooth" });
}

// ===== 录音 =====

let mediaRecorder = null;
let audioChunks = [];
let timerInterval = null;
let startTime = 0;

const btnRecord = document.getElementById("btn-record");
const btnStop = document.getElementById("btn-stop");
const timerEl = document.getElementById("timer");
const statusEl = document.getElementById("record-status");

btnRecord.addEventListener("click", startRecording);
btnStop.addEventListener("click", stopRecording);

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: 16000 },
        channelCount: { ideal: 1 },
        echoCancellation: { ideal: true },
        noiseSuppression: { ideal: true },
      }
    });
    // mimeType: 优先 webm;Safari 不支持则回落到默认(mp4)
    const mimeType = MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : (MediaRecorder.isTypeSupported("audio/mp4") ? "audio/mp4" : "");
    mediaRecorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.onstop = handleRecordingStop;
    mediaRecorder.onerror = e => {
      console.error("MediaRecorder error:", e);
      alert("录音过程中出错: " + (e.error?.message || "未知错误"));
      stopRecording();
    };

    mediaRecorder.start();
    startTime = Date.now();

    btnRecord.classList.add("hidden");
    btnStop.classList.remove("hidden");
    statusEl.textContent = "🎙️ 正在录音...请完整回答题目";

    timerInterval = setInterval(updateTimer, 100);
  } catch (err) {
    if (err.name === "NotAllowedError") {
      alert("麦克风权限被拒绝。请在浏览器设置中允许麦克风访问。");
    } else if (err.name === "NotFoundError") {
      alert("未检测到麦克风设备。请检查硬件连接。");
    } else {
      alert("无法访问麦克风: " + err.message);
    }
  }
}

function updateTimer() {
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  timerEl.textContent = `${m}:${s}`;
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  clearInterval(timerInterval);
  btnRecord.classList.remove("hidden");
  btnStop.classList.add("hidden");
  statusEl.textContent = "录音结束,正在上传评分...";
}

async function handleRecordingStop() {
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  await uploadAndScore(blob);
}

async function uploadAndScore(blob) {
  if (!currentQuestion) return;

  document.getElementById("loading-section").classList.remove("hidden");
  statusEl.textContent = "上传中...";

  const formData = new FormData();
  // 改名 .wav (后端接受多种格式)
  formData.append("audio", blob, "recording.webm");
  formData.append("question_id", currentQuestion.id);

  try {
    const res = await fetch(`${API_BASE}/api/score`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      // 尝试解析 JSON 错误,失败则用 HTTP status
      let errMsg = res.statusText;
      try {
        const err = await res.json();
        if (typeof err.detail === "string") {
          errMsg = err.detail;
        } else if (Array.isArray(err.detail)) {
          // FastAPI validation error: array of {loc, msg, type}
          errMsg = err.detail.map(e => e.msg).join("; ");
        }
      } catch (_) { /* 非 JSON 响应,继续用 statusText */ }
      throw new Error(`${res.status} ${errMsg}`);
    }

    const report = await res.json();
    // 缓存到 sessionStorage,跳转到报告页
    sessionStorage.setItem("ielts_report", JSON.stringify(report));
    sessionStorage.setItem("ielts_question", JSON.stringify(currentQuestion));
    window.location.href = "/report";
  } catch (err) {
    alert("评分失败: " + err.message);
    document.getElementById("loading-section").classList.add("hidden");
    statusEl.textContent = "";
  }
}

// 初始加载
loadQuestions();