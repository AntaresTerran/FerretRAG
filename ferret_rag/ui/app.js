const viewLinks = document.querySelectorAll("[data-view-link]");
const views = document.querySelectorAll("[data-view]");
const serverStatus = document.querySelector("#serverStatus");
const modelStatus = document.querySelector("#modelStatus");
const runtimeStatus = document.querySelector("#runtimeStatus");
const chunkStatus = document.querySelector("#chunkStatus");
const refreshHealth = document.querySelector("#refreshHealth");
const refreshIndexButton = document.querySelector("#refreshIndexButton");
const configDataDir = document.querySelector("#configDataDir");
const configChunkWords = document.querySelector("#configChunkWords");
const configChunkOverlap = document.querySelector("#configChunkOverlap");
const configTopK = document.querySelector("#configTopK");
const modelSelect = document.querySelector("#modelSelect");
const selectModelButton = document.querySelector("#selectModelButton");
const modelDetail = document.querySelector("#modelDetail");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const sources = document.querySelector("#sources");
const copyAnswerButton = document.querySelector("#copyAnswerButton");
const clearChatButton = document.querySelector("#clearChatButton");
const goHomeButton = document.querySelector("#goHomeButton");
const goParentButton = document.querySelector("#goParentButton");
const currentPath = document.querySelector("#currentPath");
const filesystemList = document.querySelector("#filesystemList");
const selectionSummary = document.querySelector("#selectionSummary");
const indexSelectedButton = document.querySelector("#indexSelectedButton");
const manualPath = document.querySelector("#manualPath");
const manualIndexButton = document.querySelector("#manualIndexButton");
const indexStatus = document.querySelector("#indexStatus");
const indexedLibrary = document.querySelector("#indexedLibrary");
const librarySummary = document.querySelector("#librarySummary");

const selectedPaths = new Set();
let filesystemState = null;
let lastAnswer = "";

function activeView() {
  const requested = window.location.hash.replace("#", "") || "chat";
  return ["chat", "index", "settings"].includes(requested) ? requested : "chat";
}

function renderView() {
  const current = activeView();
  for (const view of views) {
    view.classList.toggle("active", view.dataset.view === current);
  }
  for (const link of viewLinks) {
    link.classList.toggle("active", link.dataset.viewLink === current);
  }
}

async function refreshStatus() {
  try {
    const [healthResponse, modelsResponse, configResponse] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/models"),
      fetch("/api/config"),
    ]);
    const health = await healthResponse.json();
    const models = await modelsResponse.json();
    const config = await configResponse.json();
    serverStatus.textContent = health.status;
    modelStatus.textContent = health.model_exists ? "Found" : "Missing";
    runtimeStatus.textContent = health.model_compatible ? "Ready" : "Check";
    runtimeStatus.title = health.runtime_message;
    chunkStatus.textContent = health.chunks;
    renderModels(models.models || [], models.selected_model);
    renderConfig(config.index);
    await refreshIndexState();
  } catch (error) {
    serverStatus.textContent = "Offline";
    modelStatus.textContent = "Unknown";
    runtimeStatus.textContent = "Unknown";
  }
}

function renderConfig(indexConfig) {
  configDataDir.textContent = indexConfig.data_dir;
  configChunkWords.textContent = indexConfig.chunk_words;
  configChunkOverlap.textContent = indexConfig.chunk_overlap;
  configTopK.textContent = indexConfig.top_k;
}

async function refreshIndexState() {
  const response = await fetch("/api/index");
  const data = await response.json();
  renderIndexState(data);
}

async function loadFilesystem(path = null) {
  filesystemList.textContent = "Loading folders and files...";
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  try {
    const response = await fetch(`/api/filesystem${query}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Could not read folder.");
    }
    filesystemState = data;
    renderFilesystem(data);
  } catch (error) {
    filesystemList.textContent = error.message;
  }
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.appendChild(paragraph);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  if (role === "assistant") {
    lastAnswer = text;
  }
}

function renderModels(items, selectedModel) {
  modelSelect.innerHTML = "";
  if (!items.length) {
    modelDetail.textContent = "No GGUF files found in models.";
    return;
  }

  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = `${item.name} (${item.status})`;
    option.selected = item.path === selectedModel;
    option.disabled = !item.is_compatible;
    modelSelect.appendChild(option);
  }

  const selected = items.find((item) => item.path === selectedModel) || items[0];
  const sizeGb = (selected.size_bytes / 1024 / 1024 / 1024).toFixed(2);
  modelDetail.textContent = `${selected.architecture || "unknown"} - ${sizeGb} GB - ${selected.status}`;
}

function renderFilesystem(data) {
  currentPath.textContent = data.current_path || "Choose a starting location";
  goParentButton.disabled = !data.parent_path;
  filesystemList.innerHTML = "";

  const entries = [...(data.folders || []), ...(data.files || [])];
  if (!entries.length) {
    filesystemList.textContent = "No supported files or folders here.";
  }

  for (const entry of entries) {
    const row = document.createElement("div");
    row.className = "browser-row";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedPaths.has(entry.path);
    checkbox.addEventListener("change", () => toggleSelection(entry.path, checkbox.checked));

    const open = document.createElement("button");
    open.className = "browser-open";
    open.type = "button";
    open.textContent = entry.kind === "folder" ? "Open" : "File";
    open.disabled = entry.kind !== "folder";
    open.addEventListener("click", () => loadFilesystem(entry.path));

    const text = document.createElement("div");
    text.className = "browser-text";
    text.innerHTML = `<strong>${escapeHtml(entry.name)}</strong><span>${escapeHtml(entry.path)}</span>`;

    row.append(checkbox, open, text);
    filesystemList.appendChild(row);
  }

  if (data.unsupported_files_count) {
    const note = document.createElement("p");
    note.className = "muted-note";
    note.textContent = `${data.unsupported_files_count} unsupported files hidden.`;
    filesystemList.appendChild(note);
  }
  updateSelectionSummary();
}

function toggleSelection(path, isSelected) {
  if (isSelected) {
    selectedPaths.add(path);
  } else {
    selectedPaths.delete(path);
  }
  updateSelectionSummary();
}

function updateSelectionSummary() {
  const count = selectedPaths.size;
  selectionSummary.textContent = count ? `${count} item${count === 1 ? "" : "s"} selected.` : "No items selected.";
  indexSelectedButton.disabled = count === 0;
}

function renderIndexState(data) {
  indexedLibrary.innerHTML = "";
  const files = data.files || [];
  librarySummary.textContent = files.length
    ? `${files.length} files, ${data.chunks_total} chunks indexed.`
    : "No files indexed yet.";

  if (!files.length) {
    indexedLibrary.textContent = "No files indexed yet.";
    return;
  }

  const roots = data.roots || [];
  const grouped = new Map();
  for (const root of roots) {
    grouped.set(root.path, { root, files: [] });
  }
  const looseFiles = [];
  for (const file of files) {
    if (file.root_path && grouped.has(file.root_path)) {
      grouped.get(file.root_path).files.push(file);
    } else {
      looseFiles.push(file);
    }
  }

  for (const group of grouped.values()) {
    indexedLibrary.appendChild(rootGroup(group.root, group.files));
  }
  if (looseFiles.length) {
    indexedLibrary.appendChild(rootGroup({ path: "Individual files", active_file_count: looseFiles.length }, looseFiles));
  }
}

function rootGroup(root, files) {
  const group = document.createElement("section");
  group.className = "library-group";

  const header = document.createElement("div");
  header.className = "library-group-header";
  const title = document.createElement("div");
  title.innerHTML = `<strong>${escapeHtml(root.path)}</strong><span>${root.active_file_count || files.length} files</span>`;
  header.appendChild(title);

  if (root.path !== "Individual files") {
    const remove = removeButton(root.path, "Remove folder from index");
    header.appendChild(remove);
  }
  group.appendChild(header);

  for (const file of files) {
    group.appendChild(fileRow(file));
  }
  return group;
}

function fileRow(file) {
  const row = document.createElement("div");
  row.className = "file-row";
  const modified = file.modified_time ? new Date(file.modified_time * 1000).toLocaleString() : "Unknown";
  row.innerHTML = `
    <div class="file-main">
      <strong>${escapeHtml(file.file_name)}</strong>
      <span>${escapeHtml(file.file_path)}</span>
    </div>
    <div class="file-meta">${escapeHtml(file.file_type || "file")} - ${file.chunk_count} chunks - ${escapeHtml(modified)}</div>
  `;
  row.appendChild(removeButton(file.file_path, "Remove file from index"));
  return row;
}

function removeButton(path, title) {
  const remove = document.createElement("button");
  remove.className = "icon-button";
  remove.type = "button";
  remove.textContent = "X";
  remove.title = title;
  remove.addEventListener("click", () => removeFromIndex(path));
  return remove;
}

async function removeFromIndex(path) {
  indexStatus.textContent = "Removing from index...";
  const response = await fetch("/api/index", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const data = await response.json();
  if (!response.ok) {
    indexStatus.textContent = data.detail || "Remove failed.";
    return;
  }
  indexStatus.textContent = `${data.files_removed} files removed, ${data.chunks_removed} chunks removed.`;
  await refreshStatus();
}

async function indexPaths(paths) {
  if (!paths.length) {
    indexStatus.textContent = "Choose files or folders first.";
    return;
  }

  setIndexButtonsDisabled(true);
  indexStatus.textContent = `Indexing ${paths.length} item${paths.length === 1 ? "" : "s"}...`;
  const totals = { indexed: 0, skipped: 0, failed: 0 };
  try {
    for (const path of paths) {
      const response = await fetch("/api/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Indexing failed.");
      }
      totals.indexed += data.files_indexed;
      totals.skipped += data.files_skipped;
      totals.failed += data.files_failed;
    }
    indexStatus.textContent = `${totals.indexed} indexed, ${totals.skipped} skipped, ${totals.failed} failed.`;
    selectedPaths.clear();
    await refreshStatus();
    if (filesystemState) {
      renderFilesystem(filesystemState);
    }
  } catch (error) {
    indexStatus.textContent = error.message;
  } finally {
    setIndexButtonsDisabled(false);
  }
}

function setIndexButtonsDisabled(isDisabled) {
  indexSelectedButton.disabled = isDisabled || selectedPaths.size === 0;
  manualIndexButton.disabled = isDisabled;
}

function renderSources(items) {
  if (!items.length) {
    sources.textContent = "No matching snippets found.";
    return;
  }

  sources.innerHTML = "";
  for (const item of items) {
    const wrapper = document.createElement("article");
    wrapper.className = "source-item";

    const title = document.createElement("div");
    title.className = "source-path";
    const location = item.page_num ? `page ${item.page_num}` : item.label || item.file_type;
    title.textContent = `${item.file_name} - ${location} - chunk ${item.chunk_id}`;

    const path = document.createElement("div");
    path.className = "source-meta";
    path.textContent = item.file_path;

    const text = document.createElement("div");
    text.textContent = item.text.slice(0, 420);

    wrapper.append(title, path, text);
    sources.appendChild(wrapper);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return entities[char];
  });
}

window.addEventListener("hashchange", renderView);
refreshHealth.addEventListener("click", refreshStatus);
refreshIndexButton.addEventListener("click", refreshStatus);
goHomeButton.addEventListener("click", () => loadFilesystem());
goParentButton.addEventListener("click", () => {
  if (filesystemState?.parent_path) {
    loadFilesystem(filesystemState.parent_path);
  }
});
indexSelectedButton.addEventListener("click", () => indexPaths([...selectedPaths]));
manualIndexButton.addEventListener("click", () => indexPaths([manualPath.value.trim()].filter(Boolean)));

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  addMessage("user", message);
  addMessage("assistant", "Thinking...");
  const pendingMessage = messages.lastElementChild;
  messageInput.value = "";
  messageInput.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Chat failed.");
    }
    pendingMessage.querySelector("p").textContent = data.answer;
    lastAnswer = data.answer;
    renderSources(data.sources || []);
  } catch (error) {
    pendingMessage.querySelector("p").textContent = error.message;
    lastAnswer = error.message;
  } finally {
    messageInput.disabled = false;
    messageInput.focus();
  }
});

selectModelButton.addEventListener("click", async () => {
  const path = modelSelect.value;
  if (!path) {
    modelDetail.textContent = "Choose a model first.";
    return;
  }

  selectModelButton.disabled = true;
  modelDetail.textContent = "Switching model...";
  try {
    const response = await fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Model switch failed.");
    }
    modelDetail.textContent = data.runtime.message;
    await refreshStatus();
  } catch (error) {
    modelDetail.textContent = error.message;
  } finally {
    selectModelButton.disabled = false;
  }
});

copyAnswerButton.addEventListener("click", async () => {
  if (lastAnswer) {
    await navigator.clipboard.writeText(lastAnswer);
  }
});

clearChatButton.addEventListener("click", () => {
  messages.innerHTML = "";
  sources.textContent = "Relevant snippets will appear here.";
  lastAnswer = "";
});

renderView();
updateSelectionSummary();
loadFilesystem();
refreshStatus();
