/**
 * RAG from Scratch — Frontend Application Logic
 * Handles: status polling, file upload, ingestion, streaming chat
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

const state = {
  storeReady:  false,
  files:       [],       // staged files for upload
  streaming:   false,    // is an SSE stream in progress?
  msgCount:    0,
};

// ─────────────────────────────────────────────────────────────────────────────
// DOM References
// ─────────────────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const dom = {
  statusBadge:    $("status-badge"),
  statusText:     $("status-text"),
  cfgLlm:         $("cfg-llm"),
  cfgEmbed:       $("cfg-embed"),
  cfgStore:       $("cfg-store"),

  dropZone:       $("drop-zone"),
  fileInput:      $("file-input"),
  fileList:       $("file-list"),
  btnIngest:      $("btn-ingest"),
  btnSample:      $("btn-sample"),
  ingestProgress: $("ingest-progress"),
  ingestStatus:   $("ingest-status"),

  chatMessages:   $("chat-messages"),
  chatInput:      $("chat-input"),
  btnSend:        $("btn-send"),
  toastContainer: $("toast-container"),

  stepIngest:     $("step-ingest"),
  stepRetrieve:   $("step-retrieve"),
  stepGenerate:   $("step-generate"),
};

// ─────────────────────────────────────────────────────────────────────────────
// Status & Config
// ─────────────────────────────────────────────────────────────────────────────

async function fetchStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();

    state.storeReady = data.store_ready;

    // Update badge
    dom.statusBadge.className = `status-badge ${data.store_ready ? "status-ready" : "status-unready"}`;
    dom.statusText.textContent = data.store_ready ? "Vector Store Ready" : "Awaiting Ingestion";

    // Update config panel
    dom.cfgLlm.textContent   = `${data.llm_provider}/${data.llm_model}`;
    dom.cfgEmbed.textContent  = data.embed_model;
    dom.cfgStore.textContent  = data.store_type;

    // Enable send button if store is ready and input has content
    updateSendBtn();

  } catch {
    dom.statusText.textContent = "Server Offline";
    dom.statusBadge.className  = "status-badge status-unready";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// File Upload & Drop Zone
// ─────────────────────────────────────────────────────────────────────────────

dom.dropZone.addEventListener("click", () => dom.fileInput.click());
dom.dropZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") dom.fileInput.click(); });

dom.dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dom.dropZone.classList.add("over"); });
dom.dropZone.addEventListener("dragleave", () => dom.dropZone.classList.remove("over"));
dom.dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dom.dropZone.classList.remove("over");
  addFiles([...e.dataTransfer.files]);
});

dom.fileInput.addEventListener("change", () => {
  addFiles([...dom.fileInput.files]);
  dom.fileInput.value = "";
});

function addFiles(newFiles) {
  const allowed = [".pdf", ".txt", ".md", ".docx"];
  const valid   = newFiles.filter(f => allowed.includes("." + f.name.split(".").pop().toLowerCase()));
  
  if (valid.length < newFiles.length) {
    showToast("Some files skipped — only PDF, TXT, MD, DOCX supported.", "info");
  }

  state.files = [...state.files, ...valid];
  renderFileList();
}

function renderFileList() {
  if (state.files.length === 0) {
    dom.fileList.style.display = "none";
    dom.btnIngest.disabled = true;
    return;
  }
  dom.fileList.style.display = "flex";
  dom.btnIngest.disabled = false;
  dom.fileList.innerHTML = state.files.map((f, i) => `
    <div class="file-item">
      <span>${fileEmoji(f.name)} ${f.name}</span>
      <span style="color:var(--clr-text-3);font-size:0.7rem">${formatBytes(f.size)}</span>
      <button onclick="removeFile(${i})" style="background:none;border:none;cursor:pointer;color:var(--clr-text-3);font-size:1rem;line-height:1;padding:0;" title="Remove">×</button>
    </div>
  `).join("");
}

function removeFile(idx) {
  state.files.splice(idx, 1);
  renderFileList();
}

function fileEmoji(name) {
  const ext = name.split(".").pop().toLowerCase();
  return { pdf: "📄", txt: "📝", md: "📋", docx: "📃" }[ext] || "📎";
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + " KB";
  return (bytes/1024/1024).toFixed(1) + " MB";
}

// ─────────────────────────────────────────────────────────────────────────────
// Ingestion
// ─────────────────────────────────────────────────────────────────────────────

dom.btnIngest.addEventListener("click", () => ingestFiles());
dom.btnSample.addEventListener("click", () => ingestSample());

async function ingestFiles() {
  if (state.files.length === 0) return;

  const formData = new FormData();
  state.files.forEach(f => formData.append("files", f));

  await runIngestion(async () => {
    const res = await fetch("/api/ingest", { method: "POST", body: formData });
    return res.json();
  });
}

async function ingestSample() {
  await runIngestion(async () => {
    const res = await fetch("/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ use_sample: true }),
    });
    return res.json();
  });
}

async function runIngestion(fetchFn) {
  dom.btnIngest.disabled = true;
  dom.btnSample.disabled = true;
  dom.ingestProgress.style.display = "block";
  dom.ingestStatus.textContent = "Processing documents…";
  activateStep("ingest");

  try {
    const data = await fetchFn();

    if (data.error) {
      showToast(`Ingestion failed: ${data.error}`, "error");
      dom.ingestStatus.textContent = "Failed.";
    } else {
      dom.ingestStatus.textContent = `✅ ${data.chunks} chunks indexed from ${data.documents} document(s)`;
      state.files = [];
      renderFileList();
      showToast(`🎉 Indexed ${data.chunks} chunks! Ready to query.`, "success");
      await fetchStatus();
      updateSendBtn();
    }
  } catch (err) {
    showToast(`Network error: ${err.message}`, "error");
    dom.ingestStatus.textContent = "Error.";
  } finally {
    dom.btnIngest.disabled = false;
    dom.btnSample.disabled = false;
    deactivateStep("ingest");
    setTimeout(() => { dom.ingestProgress.style.display = "none"; }, 4000);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat — input handling
// ─────────────────────────────────────────────────────────────────────────────

dom.chatInput.addEventListener("input", () => {
  // Auto-resize
  dom.chatInput.style.height = "auto";
  dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 140) + "px";
  updateSendBtn();
});

dom.chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!dom.btnSend.disabled) sendMessage();
  }
});

dom.btnSend.addEventListener("click", sendMessage);

function updateSendBtn() {
  const hasInput = dom.chatInput.value.trim().length > 0;
  dom.btnSend.disabled = !(state.storeReady && hasInput && !state.streaming);
}

function setQuery(text) {
  dom.chatInput.value = text;
  dom.chatInput.style.height = "auto";
  dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 140) + "px";
  updateSendBtn();
  dom.chatInput.focus();
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat — sending & streaming
// ─────────────────────────────────────────────────────────────────────────────

async function sendMessage() {
  const query = dom.chatInput.value.trim();
  if (!query || state.streaming) return;

  // Clear welcome screen on first message
  if (state.msgCount === 0) {
    const welcome = dom.chatMessages.querySelector(".welcome-msg");
    if (welcome) welcome.remove();
  }
  state.msgCount++;

  // Add user bubble
  addUserBubble(query);
  dom.chatInput.value = "";
  dom.chatInput.style.height = "auto";
  updateSendBtn();

  // Add thinking bubble
  const thinkingId = "thinking-" + Date.now();
  addThinkingBubble(thinkingId);
  scrollToBottom();

  activateStep("retrieve");
  state.streaming = true;
  updateSendBtn();

  try {
    await streamAnswer(query, thinkingId);
  } finally {
    state.streaming = false;
    deactivateStep("retrieve");
    deactivateStep("generate");
    updateSendBtn();
  }
}

function streamAnswer(query, thinkingId) {
  return new Promise((resolve, reject) => {
    const url = `/api/query/stream?q=${encodeURIComponent(query)}`;
    const evtSource = new EventSource(url);

    let aiMsgEl     = null;
    let contentEl   = null;
    let cursorEl    = null;
    let fullText    = "";
    let sourcesData = null;
    let removed     = false;

    function removeThinking() {
      if (!removed) {
        removed = true;
        const el = document.getElementById(thinkingId);
        if (el) el.closest(".msg").remove();
      }
    }

    evtSource.onmessage = (e) => {
      const payload = JSON.parse(e.data);

      if (payload.type === "sources") {
        sourcesData = payload.sources;
        activateStep("generate");
        removeThinking();

        // Create AI bubble
        const { msgEl, bodyEl } = createAiBubble();
        contentEl = document.createElement("div");
        contentEl.className = "msg-content";
        bodyEl.appendChild(contentEl);
        cursorEl = document.createElement("span");
        cursorEl.className = "stream-cursor";
        contentEl.appendChild(cursorEl);
        aiMsgEl = bodyEl;
        dom.chatMessages.appendChild(msgEl);
        scrollToBottom();

      } else if (payload.type === "token") {
        if (!contentEl) return;
        fullText += payload.content;
        // Insert before cursor
        const textNode = document.createTextNode(payload.content);
        contentEl.insertBefore(textNode, cursorEl);
        scrollToBottom();

      } else if (payload.type === "done") {
        evtSource.close();
        if (cursorEl) cursorEl.remove();
        // Format markdown-ish content
        if (contentEl) contentEl.innerHTML = formatContent(fullText);
        // Append sources
        if (sourcesData && aiMsgEl) {
          aiMsgEl.appendChild(buildSourcesBox(sourcesData));
        }
        scrollToBottom();
        resolve();

      } else if (payload.type === "error") {
        evtSource.close();
        removeThinking();
        showToast(`Error: ${payload.error}`, "error");
        reject(new Error(payload.error));
      }
    };

    evtSource.onerror = () => {
      evtSource.close();
      removeThinking();
      showToast("Connection to server lost. Is the server running?", "error");
      reject(new Error("SSE connection error"));
    };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Bubble Builders
// ─────────────────────────────────────────────────────────────────────────────

function addUserBubble(text) {
  const msg = document.createElement("div");
  msg.className = "msg msg-user";
  msg.innerHTML = `
    <div class="msg-avatar">🧑</div>
    <div class="msg-body">
      <div class="msg-role">You</div>
      <div class="msg-content">${escapeHtml(text)}</div>
    </div>
  `;
  dom.chatMessages.appendChild(msg);
}

function addThinkingBubble(id) {
  const msg = document.createElement("div");
  msg.className = "msg msg-ai";
  msg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body" id="${id}">
      <div class="msg-role">RAG System</div>
      <div class="msg-content">
        <span style="color:var(--clr-text-3);font-size:0.82rem">Retrieving context</span>
        <span class="thinking-dots" style="margin-left:6px">
          <span></span><span></span><span></span>
        </span>
      </div>
    </div>
  `;
  dom.chatMessages.appendChild(msg);
}

function createAiBubble() {
  const msgEl  = document.createElement("div");
  const bodyEl = document.createElement("div");
  msgEl.className  = "msg msg-ai";
  bodyEl.className = "msg-body";
  msgEl.innerHTML  = `<div class="msg-avatar">🤖</div>`;
  bodyEl.innerHTML = `<div class="msg-role">RAG System</div>`;
  msgEl.appendChild(bodyEl);
  return { msgEl, bodyEl };
}

function buildSourcesBox(sources) {
  const box = document.createElement("div");
  box.className = "sources-box";
  const list = sources.map((s, i) => {
    const page = s.page != null ? ` · p.${s.page}` : "";
    const fname = s.source ? s.source.split(/[/\\]/).pop() : "Unknown";
    return `
      <div class="source-item">
        <div class="source-meta">📚 [${i+1}] ${escapeHtml(fname)}${page}</div>
        <div class="source-snippet">"${escapeHtml(s.content)}"</div>
      </div>
    `;
  }).join("");

  box.innerHTML = `
    <div class="sources-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'flex':'none'">
      📎 ${sources.length} source chunk${sources.length !== 1 ? "s" : ""} retrieved <span style="margin-left:auto">▾</span>
    </div>
    <div class="sources-list">${list}</div>
  `;
  return box;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Step Highlighting
// ─────────────────────────────────────────────────────────────────────────────

function activateStep(name) {
  const el = { ingest: dom.stepIngest, retrieve: dom.stepRetrieve, generate: dom.stepGenerate }[name];
  if (el) el.classList.add("active");
}
function deactivateStep(name) {
  const el = { ingest: dom.stepIngest, retrieve: dom.stepRetrieve, generate: dom.stepGenerate }[name];
  if (el) el.classList.remove("active");
}

// ─────────────────────────────────────────────────────────────────────────────
// Toast Notifications
// ─────────────────────────────────────────────────────────────────────────────

function showToast(message, type = "info", duration = 4000) {
  const icons = { success: "✅", error: "❌", info: "💡" };
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type]}</span><span>${message}</span>`;
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("leaving");
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function scrollToBottom() {
  requestAnimationFrame(() => {
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  });
}

function escapeHtml(text) {
  return (text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatContent(text) {
  // Very light markdown-to-HTML
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/^- (.+)/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
    .replace(/\n\n/g, "<br/><br/>")
    .replace(/\n/g, "<br/>");
}

// ─────────────────────────────────────────────────────────────────────────────
// Expose for inline onclick
// ─────────────────────────────────────────────────────────────────────────────

window.setQuery    = setQuery;
window.removeFile  = removeFile;

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

fetchStatus();
setInterval(fetchStatus, 15000);   // poll every 15s
dom.chatInput.focus();
