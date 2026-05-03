const folderPath = document.querySelector("#folderPath");
const indexButton = document.querySelector("#indexButton");
const indexStatus = document.querySelector("#indexStatus");
const indexedRoots = document.querySelector("#indexedRoots");
const indexedFiles = document.querySelector("#indexedFiles");
const serverStatus = document.querySelector("#serverStatus");
const modelStatus = document.querySelector("#modelStatus");
const runtimeStatus = document.querySelector("#runtimeStatus");
const chunkStatus = document.querySelector("#chunkStatus");
const refreshHealth = document.querySelector("#refreshHealth");
const modelSelect = document.querySelector("#modelSelect");
const selectModelButton = document.querySelector("#selectModelButton");
const modelDetail = document.querySelector("#modelDetail");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const sources = document.querySelector("#sources");
const copyAnswerButton = document.querySelector("#copyAnswerButton");
const clearChatButton = document.querySelector("#clearChatButton");

let lastAnswer = "";

async function refreshStatus() {
  try {
    const [healthResponse, modelsResponse] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/models"),
    ]);
    const health = await healthResponse.json();
    const models = await modelsResponse.json();
    serverStatus.textContent = health.status;
    modelStatus.textContent = health.model_exists ? "Found" : "Missing";
    runtimeStatus.textContent = health.model_compatible ? "Ready" : "Check";
    runtimeStatus.title = health.runtime_message;
    chunkStatus.textContent = health.chunks;
    renderModels(models.models || [], models.selected_model);
    await refreshIndexState();
  } catch (error) {
    serverStatus.textContent = "Offline";
    modelStatus.textContent = "Unknown";
    runtimeStatus.textContent = "Unknown";
  }
}

async function refreshIndexState() {
  const response = await fetch("/api/index");
  const data = await response.json();
  renderIndexState(data);
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

function renderIndexState(data) {
  indexedRoots.innerHTML = "";
  indexedFiles.innerHTML = "";

  if (!data.roots?.length) {
    indexedRoots.textContent = "No folders indexed.";
  } else {
    for (const root of data.roots) {
      indexedRoots.appendChild(indexItem(root.path, `${root.active_file_count} files`, true));
    }
  }

  if (data.files?.length) {
    for (const file of data.files.slice(0, 12)) {
      indexedFiles.appendChild(
        indexItem(file.file_path, `${file.file_name} - ${file.chunk_count} chunks`, false),
      );
    }
    if (data.files.length > 12) {
      const more = document.createElement("div");
      more.className = "index-note";
      more.textContent = `${data.files.length - 12} more files indexed.`;
      indexedFiles.appendChild(more);
    }
  }
}

function indexItem(path, label, isRoot) {
  const wrapper = document.createElement("div");
  wrapper.className = "index-item";

  const text = document.createElement("div");
  text.className = "index-text";
  text.textContent = label;
  text.title = path;

  const remove = document.createElement("button");
  remove.className = "icon-button";
  remove.type = "button";
  remove.textContent = "X";
  remove.title = isRoot ? "Remove folder from index" : "Remove file from index";
  remove.addEventListener("click", () => removeFromIndex(path));

  wrapper.append(text, remove);
  return wrapper;
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

indexButton.addEventListener("click", async () => {
  const path = folderPath.value.trim();
  if (!path) {
    indexStatus.textContent = "Enter a folder path first.";
    return;
  }

  indexButton.disabled = true;
  indexStatus.textContent = "Indexing folder...";
  try {
    const response = await fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Indexing failed.");
    }
    indexStatus.textContent = `${data.files_indexed} indexed, ${data.files_skipped} skipped, ${data.files_failed} failed.`;
    if (data.failures?.length) {
      indexStatus.textContent += ` First failure: ${data.failures[0].error}`;
    }
    await refreshStatus();
  } catch (error) {
    indexStatus.textContent = error.message;
  } finally {
    indexButton.disabled = false;
  }
});

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
  if (!lastAnswer) {
    return;
  }
  await navigator.clipboard.writeText(lastAnswer);
});

clearChatButton.addEventListener("click", () => {
  messages.innerHTML = "";
  sources.textContent = "Relevant snippets will appear here.";
  lastAnswer = "";
});

refreshHealth.addEventListener("click", refreshStatus);
refreshStatus();
