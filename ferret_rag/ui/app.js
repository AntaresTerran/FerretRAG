const folderPath = document.querySelector("#folderPath");
const indexButton = document.querySelector("#indexButton");
const indexStatus = document.querySelector("#indexStatus");
const serverStatus = document.querySelector("#serverStatus");
const modelStatus = document.querySelector("#modelStatus");
const chunkStatus = document.querySelector("#chunkStatus");
const refreshHealth = document.querySelector("#refreshHealth");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const sources = document.querySelector("#sources");

async function refreshStatus() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    serverStatus.textContent = data.status;
    modelStatus.textContent = data.model_exists ? "Found" : "Missing";
    chunkStatus.textContent = data.chunks;
  } catch (error) {
    serverStatus.textContent = "Offline";
    modelStatus.textContent = "Unknown";
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
  messageInput.value = "";
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
    addMessage("assistant", data.answer);
    renderSources(data.sources || []);
  } catch (error) {
    addMessage("assistant", error.message);
  }
});

refreshHealth.addEventListener("click", refreshStatus);
refreshStatus();
