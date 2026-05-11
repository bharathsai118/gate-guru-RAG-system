const VISITOR_KEY = "gate_guru_visitor_id";

const state = {
  visitorId: getVisitorId(),
  mode: "ask",
  subjects: [],
  lastQuestion: "",
  lastAnswer: "",
  lastSources: [],
};

function visitorHeaders(extra = {}) {
  return {
    "X-Visitor-Id": state.visitorId,
    ...extra,
  };
}

const elements = {
  subjectSelect: document.getElementById("subjectSelect"),
  uploadSubject: document.getElementById("uploadSubject"),
  modeButtons: document.getElementById("modeButtons"),
  includeDa: document.getElementById("includeDa"),
  pdfInput: document.getElementById("pdfInput"),
  fileName: document.getElementById("fileName"),
  uploadButton: document.getElementById("uploadButton"),
  uploadStatus: document.getElementById("uploadStatus"),
  questionInput: document.getElementById("questionInput"),
  askButton: document.getElementById("askButton"),
  answerOutput: document.getElementById("answerOutput"),
  sourcesOutput: document.getElementById("sourcesOutput"),
  loadingIndicator: document.getElementById("loadingIndicator"),
  statusPanel: document.getElementById("statusPanel"),
  indexButton: document.getElementById("indexButton"),
  visitorBadge: document.getElementById("visitorBadge"),
  feedbackPanel: document.getElementById("feedbackPanel"),
};

function getVisitorId() {
  const existing = localStorage.getItem(VISITOR_KEY);
  if (existing) return existing;
  const generated =
    window.crypto && crypto.randomUUID
      ? crypto.randomUUID()
      : `visitor_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  localStorage.setItem(VISITOR_KEY, generated);
  return generated;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showStatus(message, type = "info") {
  elements.statusPanel.className = `status-panel ${type}`;
  elements.statusPanel.textContent = message;
  elements.statusPanel.classList.remove("hidden");
}

function clearStatus() {
  elements.statusPanel.classList.add("hidden");
  elements.statusPanel.textContent = "";
}

function setLoading(isLoading) {
  elements.askButton.disabled = isLoading;
  elements.loadingIndicator.classList.toggle("hidden", !isLoading);
}

function setUploadLoading(isLoading) {
  elements.uploadButton.disabled = isLoading;
  elements.uploadButton.querySelector("span").textContent = isLoading ? "Indexing..." : "Upload and Index";
}

function populateSubjects(payload) {
  const groups = payload.subject_groups || [];
  state.subjects = groups;

  elements.subjectSelect.innerHTML = groups
    .map((group) => `<option value="${escapeHtml(group.value)}">${escapeHtml(group.label)}</option>`)
    .join("");

  const uploadGroups = groups.filter((group) => group.value !== "All");
  elements.uploadSubject.innerHTML = uploadGroups
    .map((group) => {
      const selected = group.value === "General" ? "selected" : "";
      return `<option value="${escapeHtml(group.value)}" ${selected}>${escapeHtml(group.label)}</option>`;
    })
    .join("");
}

async function loadSubjects() {
  try {
    const response = await fetch("/api/subjects");
    const payload = await response.json();
    populateSubjects(payload);
  } catch (error) {
    showStatus(`Could not load subjects: ${error.message}`, "error");
  }
}

function renderAnswer(text) {
  elements.answerOutput.textContent = text || "No answer returned.";
}

function appendAnswer(text) {
  elements.answerOutput.textContent += text;
}

function setFeedbackVisible(isVisible) {
  elements.feedbackPanel.classList.toggle("hidden", !isVisible);
}

function renderWarnings(warnings) {
  if (!warnings || !warnings.length) return;
  showStatus(warnings[0], "warning");
}

function renderSources(sources) {
  if (!sources || !sources.length) {
    elements.sourcesOutput.innerHTML = '<div class="empty-source">No citations returned.</div>';
    return;
  }

  elements.sourcesOutput.innerHTML = sources
    .map((source) => {
      const page = source.page_number ? `Page ${escapeHtml(source.page_number)}` : "Page unavailable";
      const type = source.source_type === "user_upload" ? "Your upload" : "Preloaded";
      return `
        <div class="source-card">
          <div class="source-file">${escapeHtml(source.filename || "unknown.pdf")}</div>
          <div class="source-meta">${page}</div>
          <div class="source-meta">${escapeHtml(source.category || "Unknown")}</div>
          <span class="source-pill">${escapeHtml(type)}</span>
        </div>
      `;
    })
    .join("");
}

async function indexPreloaded() {
  clearStatus();
  elements.indexButton.disabled = true;
  elements.indexButton.querySelector("span").textContent = "Indexing...";
  try {
    const response = await fetch("/api/index-preloaded", {
      method: "POST",
      headers: visitorHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ force: false }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Indexing failed.");

    const missing = payload.missing_files && payload.missing_files.length
      ? ` Missing files: ${payload.missing_files.length}.`
      : "";
    showStatus(
      `Indexed ${payload.indexed_chunks || 0} chunks from ${(payload.indexed_files || []).length} file(s). Skipped ${(payload.skipped_files || []).length}.${missing}`,
      payload.errors && payload.errors.length ? "warning" : "success",
    );
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    elements.indexButton.disabled = false;
    elements.indexButton.querySelector("span").textContent = "Index Resources";
    if (window.lucide) lucide.createIcons();
  }
}

async function uploadPdf() {
  clearStatus();
  const file = elements.pdfInput.files[0];
  if (!file) {
    elements.uploadStatus.textContent = "Select a PDF first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("visitor_id", state.visitorId);
  formData.append("subject_group", elements.uploadSubject.value || "General");

  setUploadLoading(true);
  elements.uploadStatus.textContent = "";

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      headers: visitorHeaders(),
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Upload failed.");

    elements.uploadStatus.textContent = `${payload.filename} indexed with ${payload.chunk_count} chunks.`;
    showStatus((payload.warnings && payload.warnings[0]) || "Upload indexed for this browser visitor.", payload.warnings ? "warning" : "success");
  } catch (error) {
    elements.uploadStatus.textContent = error.message;
    showStatus(error.message, "error");
  } finally {
    setUploadLoading(false);
    if (window.lucide) lucide.createIcons();
  }
}

async function askQuestion() {
  clearStatus();
  const question = elements.questionInput.value.trim();
  if (!question) {
    showStatus("Enter a question before asking.", "error");
    return;
  }

  setLoading(true);
  renderAnswer("");
  renderSources([]);
  setFeedbackVisible(false);
  state.lastQuestion = question;
  state.lastAnswer = "";
  state.lastSources = [];

  try {
    const response = await fetch("/api/ask-stream", {
      method: "POST",
      headers: visitorHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        visitor_id: state.visitorId,
        question,
        mode: state.mode,
        subject_group: elements.subjectSelect.value || "All",
        include_da_resources: elements.includeDa.checked,
      }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || "Question failed.");
    }

    await consumeAnswerStream(response);
  } catch (error) {
    renderAnswer(error.message);
    renderSources([]);
    showStatus(error.message, "error");
  } finally {
    setLoading(false);
  }
}

async function consumeAnswerStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const rawEvent of events) {
      handleStreamEvent(rawEvent);
    }
  }
  if (buffer.trim()) handleStreamEvent(buffer);
}

function handleStreamEvent(rawEvent) {
  const lines = rawEvent.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLine = lines.find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return;

  const eventName = eventLine.replace("event:", "").trim();
  const payload = JSON.parse(dataLine.replace("data:", "").trim());

  if (eventName === "meta") {
    state.lastSources = payload.sources || [];
    renderSources(state.lastSources);
    renderWarnings(payload.warnings);
    return;
  }

  if (eventName === "token") {
    appendAnswer(payload.text || "");
    state.lastAnswer += payload.text || "";
    return;
  }

  if (eventName === "error") {
    throw new Error(payload.message || "Streaming failed.");
  }

  if (eventName === "done") {
    if (payload.answer && !state.lastAnswer) {
      renderAnswer(payload.answer);
      state.lastAnswer = payload.answer;
    }
    state.lastSources = payload.sources || state.lastSources || [];
    renderSources(state.lastSources);
    renderWarnings(payload.warnings);
    if (!payload.retrieved_count) showStatus("No matching context was found.", "warning");
    if (state.lastAnswer) setFeedbackVisible(true);
    if (window.lucide) lucide.createIcons();
  }
}

async function submitFeedback(rating) {
  if (!state.lastAnswer) return;
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: visitorHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        visitor_id: state.visitorId,
        rating,
        question: state.lastQuestion,
        answer: state.lastAnswer,
        sources: state.lastSources,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Feedback failed.");
    showStatus("Feedback saved. Thank you.", "success");
  } catch (error) {
    showStatus(error.message, "error");
  }
}

function wireEvents() {
  elements.modeButtons.addEventListener("click", (event) => {
    const button = event.target.closest("[data-mode]");
    if (!button) return;
    state.mode = button.dataset.mode;
    elements.modeButtons.querySelectorAll(".mode-button").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
  });

  elements.pdfInput.addEventListener("change", () => {
    const file = elements.pdfInput.files[0];
    elements.fileName.textContent = file ? file.name : "Choose a PDF";
  });

  elements.uploadButton.addEventListener("click", uploadPdf);
  elements.askButton.addEventListener("click", askQuestion);
  elements.indexButton.addEventListener("click", indexPreloaded);
  elements.feedbackPanel.addEventListener("click", (event) => {
    const button = event.target.closest("[data-rating]");
    if (!button) return;
    submitFeedback(button.dataset.rating);
  });

  elements.questionInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      askQuestion();
    }
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  elements.visitorBadge.textContent = `Visitor ${state.visitorId.slice(0, 8)}`;
  wireEvents();
  await loadSubjects();
  if (window.lucide) lucide.createIcons();
});
