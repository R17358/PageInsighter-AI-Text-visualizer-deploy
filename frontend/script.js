// API Configuration - Change this to your deployed URL
// const API = "http://localhost:8000";
const API = "https://pageinsighter-yn21.onrender.com";

// Stats
let stats = {
  totalRequests: 0,
  imagesGenerated: 0,
  filesProcessed: 0,
  chatMessages: 0,
};

// Chat state
let chatHistory = [];
let chatSessionId = generateSessionId();
let chatFiles = [];

// Load stats from localStorage
if (localStorage.getItem("stats")) {
  stats = JSON.parse(localStorage.getItem("stats"));
  updateStatsDisplay();
}

// Load chat history from localStorage
if (localStorage.getItem(`chat_${chatSessionId}`)) {
  chatHistory = JSON.parse(localStorage.getItem(`chat_${chatSessionId}`));
  renderChatHistory();
}

function generateSessionId() {
  const stored = localStorage.getItem("currentSessionId");
  if (stored) return stored;
  
  const newId = "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
  localStorage.setItem("currentSessionId", newId);
  return newId;
}

function handleEnter(e) {
                // Enter without Ctrl → send
                if (e.key === "Enter" && !e.ctrlKey) {
                  e.preventDefault(); // new line roko
                  sendChatMessage();
                }

                // Ctrl + Enter → new line allow
              }

function updateStatsDisplay() {
  document.getElementById("totalRequests").textContent = stats.totalRequests;
  document.getElementById("imagesGenerated").textContent = stats.imagesGenerated;
  document.getElementById("filesProcessed").textContent = stats.filesProcessed;
  document.getElementById("chatMessagesState").textContent = stats.chatMessages;
  localStorage.setItem("stats", JSON.stringify(stats));
}

function incrementStat(type) {
  stats[type]++;
  updateStatsDisplay();
}

// Create particles
function createParticles() {
  const container = document.getElementById("particles");
  for (let i = 0; i < 30; i++) {
    const particle = document.createElement("div");
    particle.className = "particle";
    particle.style.left = Math.random() * 100 + "%";
    particle.style.top = Math.random() * 100 + "%";
    particle.style.animationDelay = Math.random() * 20 + "s";
    particle.style.animationDuration = 15 + Math.random() * 10 + "s";
    container.appendChild(particle);
  }
}
createParticles();

// Tab switching
function switchTab(tabName) {
  document
    .querySelectorAll(".tab")
    .forEach((t) => t.classList.remove("active"));
  document
    .querySelectorAll(".tab-content")
    .forEach((c) => c.classList.remove("active"));

  event.target.classList.add("active");
  document.getElementById(tabName).classList.add("active");
}

// Toast notifications
function showToast(type, title, message) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    warning: "⚠️",
    info: "ℹ️",
  };

  toast.innerHTML = `
    <div class="toast-icon">${icons[type]}</div>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "slideInRight 0.3s ease reverse";
    setTimeout(() => toast.remove(), 300);
  }, 5000);
}

// Helper functions
function showLoading(id) {
  document.getElementById(id).classList.add("show");
}

function hideLoading(id) {
  document.getElementById(id).classList.remove("show");
}

function showOutput(id, content) {
  const el = document.getElementById(id);
  el.innerHTML = content;
  el.style.display = "block";
  setTimeout(() => {
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 100);
}

function clearOutput(id) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  el.style.display = "none";
}

function downloadImage(base64Data, filename) {
  const link = document.createElement("a");
  link.href = "data:image/jpeg;base64," + base64Data;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("success", "Downloaded!", `Image saved as ${filename}`);
}

function downloadVisualImage() {
  const img = document.getElementById("visualImage");
  const base64 = img.src.split(",")[1];
  downloadImage(base64, "pageinsighter-visual.jpg");
}

// API call handler
async function handleApiCall(apiFunc, loadingId, errorTitle) {
  if (loadingId) showLoading(loadingId);
  incrementStat("totalRequests");

  try {
    await apiFunc();
    showToast("success", "Success!", "Operation completed successfully");
  } catch (error) {
    console.error("API Error:", error);

    let errorMessage = "Something went wrong. Please try again.";
    let toastType = "error";

    if (error.apiError) {
      errorMessage = error.message;
      toastType = error.type || "error";
    } else if (
      error.message.includes("Failed to fetch") ||
      error.message.includes("NetworkError")
    ) {
      errorMessage = "Network error. Please check your internet connection.";
      toastType = "error";
    } else {
      errorMessage = error.message || errorMessage;
    }

    showToast(toastType, errorTitle, errorMessage);
  } finally {
    if (loadingId) hideLoading(loadingId);
  }
}

// Parse API response
async function parseApiResponse(response) {
  const data = await response.json();

  if (!response.ok || !data.success) {
    const error = new Error(data.error || `HTTP ${response.status}`);
    error.apiError = true;

    if (response.status === 429) {
      error.type = "warning";
    } else if (response.status >= 400 && response.status < 500) {
      error.type = "warning";
    } else {
      error.type = "error";
    }

    throw error;
  }

  return data;
}

// ============================================================================
// CHAT FUNCTIONS
// ============================================================================

function renderChatHistory() {
  const container = document.getElementById("chatMessages");
  
  // Clear container but keep welcome message if no history
  if (chatHistory.length === 0) {
    return;
  }
  
  // Remove welcome message
  const welcomeMsg = container.querySelector(".welcome-message");
  if (welcomeMsg) {
    welcomeMsg.remove();
  }
  
  // Render all messages - don't increment stats for historical messages
  chatHistory.forEach((msg) => {
    addMessageToUI(msg.role, msg.content, msg.tool_calls, false, msg.timestamp, false);
  });
  
  scrollChatToBottom();
}

function addMessageToUI(role, content, toolCalls = null, animate = true, savedTimestamp = null, shouldIncrementStats = true) {
  const container = document.getElementById("conversation");
  
  // Remove welcome message if it exists (only on first message)
  const welcomeMsg = container.querySelector(".welcome-message");
  if (welcomeMsg) {
    welcomeMsg.remove();
  }
  
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${role}`;
  if (animate) {
    messageDiv.style.animation = "messageSlide 0.3s ease";
  }
  
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "👤" : "🤖";
  
  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  
  const textDiv = document.createElement("div");
  textDiv.className = "message-text";
  textDiv.innerHTML = formatMessageContent(content);
  
  const timestamp = document.createElement("div");
  timestamp.className = "message-timestamp";
  // Use saved timestamp if available, otherwise create new one
  if (savedTimestamp) {
    timestamp.textContent = savedTimestamp;
  } else {
    timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  
  contentDiv.appendChild(textDiv);
  
  // Add tool call results if present
  if (toolCalls && toolCalls.length > 0) {
    toolCalls.forEach((tool) => {
      const toolDiv = createToolResultUI(tool, shouldIncrementStats);
      contentDiv.appendChild(toolDiv);
    });
  }
  
  contentDiv.appendChild(timestamp);
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(contentDiv);
  
  container.appendChild(messageDiv);
  scrollChatToBottom();
}

function formatMessageContent(content) {
  // Convert markdown-style formatting to HTML
  let formatted = content
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
  
  return formatted;
}

function createToolResultUI(tool, shouldIncrementStats = true) {
  const toolDiv = document.createElement("div");
  toolDiv.className = "tool-result";
  
  const header = document.createElement("div");
  header.className = "tool-result-header";
  
  if (tool.name === "generate_image") {
    header.innerHTML = "🎨 Generated Image";
    
    if (tool.result.success && tool.result.image) {
      const imgContainer = document.createElement("div");
      imgContainer.className = "tool-result-image";
      
      const img = document.createElement("img");
      img.src = "data:image/jpeg;base64," + tool.result.image;
      img.alt = "Generated image";
      
      const downloadBtn = document.createElement("button");
      downloadBtn.className = "download-btn";
      downloadBtn.textContent = "⬇ Download";
      downloadBtn.onclick = () => downloadImage(tool.result.image, "chat-generated-image.jpg");
      
      imgContainer.appendChild(img);
      imgContainer.appendChild(downloadBtn);
      
      toolDiv.appendChild(header);
      toolDiv.appendChild(imgContainer);
      
      // Only increment stats when adding new messages, not when loading from history
      if (shouldIncrementStats) {
        incrementStat("imagesGenerated");
      }
    } else {
      header.innerHTML = "🎨 Image Generation Failed";
      const errorMsg = document.createElement("div");
      errorMsg.textContent = tool.result.error || "Unknown error";
      toolDiv.appendChild(header);
      toolDiv.appendChild(errorMsg);
    }
  } else if (tool.name === "web_search") {
    header.innerHTML = "🔍 Web Search Result";
    const result = document.createElement("div");
    result.innerHTML = `<p>${tool.result.message || JSON.stringify(tool.result)}</p>`;
    toolDiv.appendChild(header);
    toolDiv.appendChild(result);
  } else if (tool.name === "analyze_document") {
    header.innerHTML = "📄 Document Analysis";
    const result = document.createElement("div");
    result.innerHTML = `<p>${tool.result.analysis || "Analysis complete"}</p>`;
    toolDiv.appendChild(header);
    toolDiv.appendChild(result);
  }
  
  return toolDiv;
}

function scrollChatToBottom() {
  const container = document.getElementById("chatMessages");
  setTimeout(() => {
    container.scrollTop = container?.scrollHeight;
  }, 100);
}

function showTypingIndicator() {
  const container = document.getElementById("chatMessages");
  
  const indicator = document.createElement("div");
  indicator.className = "typing-indicator";
  indicator.id = "typingIndicator";
  
  indicator.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="typing-dots">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  
  container?.appendChild(indicator);
  scrollChatToBottom();
}

function hideTypingIndicator() {
  const indicator = document.getElementById("typingIndicator");
  if (indicator) {
    indicator.remove();
  }
}

async function sendChatMessage() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();
  
  if (!message && chatFiles.length === 0) {
    showToast("warning", "Input Required", "Please enter a message or attach files");
    return;
  }
  
  // Disable input while sending
  input.disabled = true;
  document.querySelector(".send-btn").disabled = true;
  
  // Generate timestamp for this message
  const currentTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  
  try {
    // Add user message to UI immediately
    const userMessage = message || "📎 [Files attached]";
    addMessageToUI("user", userMessage, null, true, currentTimestamp);
    
    // Add to history with timestamp
    const userHistoryEntry = {
      role: "user",
      content: userMessage,
      timestamp: currentTimestamp
    };
    chatHistory.push(userHistoryEntry);

    
    // Save to localStorage immediately after user message
    saveChatHistory();
    
    // Clear input
    input.value = "";
    input.style.height = "auto";

    
    // Show typing indicator
    showTypingIndicator();

    console.log("I ma")
    // Prepare form data
    const formData = new FormData();
    formData.append("message", message);
    formData.append("session_id", chatSessionId);
    formData.append("history", JSON.stringify(chatHistory.slice(-10))); // Last 10 messages
    
    // Add files if any
    chatFiles.forEach((file) => {
      formData.append("files", file);
    });

   
    
    // Send request
    const response = await fetch(`${API}/api/chat`, {
      method: "POST",
      body: formData,
    });
    
    const data = await parseApiResponse(response);
    
    // Hide typing indicator
    hideTypingIndicator();
    
    // Generate timestamp for assistant response
    const assistantTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // Add assistant response to UI immediately
    addMessageToUI("assistant", data.response, data.tool_calls, true, assistantTimestamp);
    
    // Add to history with timestamp
    const assistantHistoryEntry = {
      role: "assistant",
      content: data.response,
      tool_calls: data.tool_calls,
      timestamp: assistantTimestamp
    };
    chatHistory.push(assistantHistoryEntry);
    
    // Save to localStorage immediately after assistant response
    saveChatHistory();
    
    // Clear files
    chatFiles = [];
    updateChatFilePreview();
    
    // Update stats
    incrementStat("chatMessages");
    
    showToast("success", "Message Sent", "AI responded successfully");
    
  } catch (error) {
    hideTypingIndicator();
    console.error("Chat error:", error);
    
    let errorMessage = "Failed to send message. Please try again.";
    if (error.apiError) {
      errorMessage = error.message;
    }
    
    showToast("error", "Chat Error", errorMessage);
    
    // Remove the last user message from history if API call failed
    chatHistory.pop();
    saveChatHistory();
    
    // Reload chat to show correct state
    reloadChatUI();
    
  } finally {
    // Re-enable input
    input.disabled = false;
    document.querySelector(".send-btn").disabled = false;
    input.focus();
  }
}

// Helper function to save chat history
function saveChatHistory() {
  try {
    localStorage.setItem(`chat_${chatSessionId}`, JSON.stringify(chatHistory));
  } catch (e) {
    console.error("Failed to save chat history:", e);
  }
}

// Helper function to reload chat UI from history
function reloadChatUI() {
  const container = document.getElementById("chatMessages");
  container.innerHTML = '';
  
  if (chatHistory.length === 0) {
    container.innerHTML = `
      <div class="welcome-message">
        <div class="welcome-icon">🤖</div>
        <h3>Welcome to PageInsighter AI Chat!</h3>
        <p>I can help you with:</p>
        <ul>
          <li>🖼️ Generate images from text descriptions</li>
          <li>🔍 Search the web for current information</li>
          <li>📄 Analyze documents and images</li>
          <li>💡 Answer questions and solve problems</li>
          <li>🎨 Create visual content with multiple styles</li>
        </ul>
        <p class="welcome-tip">Try: "Generate an image of a sunset over mountains" or "Analyze this document"</p>
      </div>
    `;
  } else {
    renderChatHistory();
  }
}

function clearChat() {
  if (!confirm("Are you sure you want to clear the chat history?")) {
    return;
  }
  
  // Clear history
  chatHistory = [];
  
  // Remove from localStorage
  localStorage.removeItem(`chat_${chatSessionId}`);
  
  // Generate new session ID
  const oldSessionId = chatSessionId;
  chatSessionId = "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
  localStorage.setItem("currentSessionId", chatSessionId);
  
  // Reload UI
  reloadChatUI();
  
  showToast("info", "Chat Cleared", "Started a new conversation");
}

function exportChat() {
  if (chatHistory.length === 0) {
    showToast("warning", "No Chat History", "There's nothing to export");
    return;
  }
  
  const exportData = {
    sessionId: chatSessionId,
    exportDate: new Date().toISOString(),
    messages: chatHistory,
  };
  
  const blob = new Blob([JSON.stringify(exportData, null, 2)], {
    type: "application/json",
  });
  
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `pageinsighter-chat-${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  
  showToast("success", "Chat Exported", "Download started");
}

// Chat file handling
document.getElementById("chatFileInput").addEventListener("change", (e) => {
  const files = Array.from(e.target.files);
  chatFiles = [...chatFiles, ...files];
  updateChatFilePreview();
});

function updateChatFilePreview() {
  const preview = document.getElementById("chatFilePreview");
  
  if (chatFiles.length === 0) {
    preview.style.display = "none";
    preview.innerHTML = "";
    return;
  }
  
  preview.style.display = "block";
  preview.innerHTML = "";
  
  chatFiles.forEach((file, index) => {
    const fileItem = document.createElement("div");
    fileItem.className = "chat-file-item";
    fileItem.innerHTML = `
      <span class="attachment-icon">📎</span>
      <span>${file.name}</span>
      <span class="chat-file-remove" onclick="removeChatFile(${index})">×</span>
    `;
    preview.appendChild(fileItem);
  });
}

function removeChatFile(index) {
  chatFiles.splice(index, 1);
  updateChatFilePreview();
}

// Chat input auto-resize
document.getElementById("chatInput").addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 120) + "px";
});

// Chat input keyboard shortcuts
document.getElementById("chatInput").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    sendChatMessage();
  }
});

// ============================================================================
// OTHER API FUNCTIONS
// ============================================================================

async function askQuery() {
  const query = document.getElementById("queryText").value.trim();
  if (!query) {
    showToast("warning", "Input Required", "Please enter a question");
    return;
  }

  clearOutput("queryResult");

  await handleApiCall(
    async () => {
      const r = await fetch(`${API}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          language: document.getElementById("queryLang").value,
        }),
      });

      const d = await parseApiResponse(r);
      let content = `<div class="result-section"><strong>Answer:</strong><div class="result-content">${d.answer}</div></div>`;
      showOutput("queryResult", content);
    },
    "queryLoading",
    "Query Failed",
  );
}

async function generatePrompt() {
  const text = document.getElementById("promptText").value.trim();
  if (!text) {
    showToast("warning", "Input Required", "Please describe a scene");
    return;
  }

  clearOutput("promptResult");

  await handleApiCall(
    async () => {
      const r = await fetch(`${API}/api/generate-image-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });

      const d = await parseApiResponse(r);
      showOutput(
        "promptResult",
        `<div class="result-section"><strong>Generated Prompt:</strong><div class="result-content">${d.prompt}</div></div>`,
      );
    },
    "promptLoading",
    "Prompt Generation Failed",
  );
}

async function visualize() {
  const text = document.getElementById("visualText").value.trim();
  const style = document.getElementById("visualStyle").value;

  if (!text) {
    showToast(
      "warning",
      "Input Required",
      "Please enter text to visualize",
    );
    return;
  }

  document.getElementById("visualContainer").style.display = "none";

  await handleApiCall(
    async () => {
      const r = await fetch(`${API}/api/visualize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          style: style,
        }),
      });

      const d = await parseApiResponse(r);

      document.getElementById("visualImage").src =
        "data:image/jpeg;base64," + d.image;
      document.getElementById("visualContainer").style.display = "block";
      incrementStat("imagesGenerated");
    },
    "visualLoading",
    "Visualization Failed",
  );
}

async function processFile() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files[0]) {
    showToast(
      "warning",
      "File Required",
      "Please select a file to process",
    );
    return;
  }

  clearOutput("fileResult");
  document.getElementById("fileImages").innerHTML = "";

  await handleApiCall(
    async () => {
      const f = new FormData();
      f.append("file", fileInput.files[0]);
      f.append("language", document.getElementById("fileLang").value);
      f.append("explain", document.getElementById("explainCheck").checked);
      f.append("generate_images", document.getElementById("imageCheck").checked);
      f.append("style", document.getElementById("fileImageStyle").value);

      const r = await fetch(`${API}/api/process-file`, {
        method: "POST",
        body: f,
      });

      const d = await parseApiResponse(r);

      let content = "";
      
      if (d.extracted_text) {
        content += `<div class="result-section"><strong>Extracted Text:</strong><div class="result-content">${d.extracted_text}</div></div>`;
      }
      
      if (d.summary) {
        content += `<div class="result-section"><strong>Summary/Explanation:</strong><div class="result-content">${d.summary}</div></div>`;
      }
      
      if (d.translation) {
        content += `<div class="result-section"><strong>Translation:</strong><div class="result-content">${d.translation}</div></div>`;
      }

      if (content) {
        showOutput("fileResult", content);
      }

      const imagesContainer = document.getElementById("fileImages");

      if (d.images && d.images.length > 0) {
        d.images.forEach((img, index) => {
          const wrapper = document.createElement("div");
          wrapper.className = "image-wrapper";

          const imgElement = document.createElement("img");
          imgElement.src = "data:image/jpeg;base64," + img.image;
          imgElement.className = "generated";
          imgElement.alt = `Generated image ${index + 1}`;

          const downloadBtn = document.createElement("button");
          downloadBtn.className = "download-btn";
          downloadBtn.textContent = "⬇ Download";
          downloadBtn.onclick = () =>
            downloadImage(img.image, `pageinsighter-${index + 1}.jpg`);

          wrapper.appendChild(imgElement);
          wrapper.appendChild(downloadBtn);
          imagesContainer.appendChild(wrapper);

          incrementStat("imagesGenerated");
        });

        imagesContainer.style.display = "grid";
        setTimeout(() => {
          imagesContainer.scrollIntoView({
            behavior: "smooth",
            block: "nearest",
          });
        }, 100);
      }

      incrementStat("filesProcessed");
    },
    "fileLoading",
    "File Processing Failed",
  );
}

async function explainImage() {
  const fileInput = document.getElementById("explainInput");
  if (!fileInput.files[0]) {
    showToast(
      "warning",
      "File Required",
      "Please select an image to explain",
    );
    return;
  }

  clearOutput("explainResult");

  await handleApiCall(
    async () => {
      const f = new FormData();
      f.append("file", fileInput.files[0]);
      f.append("language", document.getElementById("explainLang").value);

      const r = await fetch(`${API}/api/explain-image`, {
        method: "POST",
        body: f,
      });

      const d = await parseApiResponse(r);
      let content = `<div class="result-section"><strong>Explanation:</strong><div class="result-content">${d.explanation}</div></div>`;
      showOutput("explainResult", content);
    },
    "explainLoading",
    "Image Explanation Failed",
  );
}

// ============================================================================
// FILE PREVIEW HANDLERS
// ============================================================================

document.getElementById("fileInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  let previewContainer = document.getElementById("filePreview");
  if (!previewContainer) {
    previewContainer = document.createElement("div");
    previewContainer.id = "filePreview";
    previewContainer.className = "file-preview";
    e.target.parentElement.appendChild(previewContainer);
  }

  previewContainer.innerHTML = "";

  if (file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = function (event) {
      previewContainer.innerHTML = `
        <div class="preview-wrapper">
          <img src="${event.target.result}" alt="Preview" class="preview-image">
          <p class="preview-filename">${file.name}</p>
        </div>
      `;
    };
    reader.readAsDataURL(file);
  } else {
    previewContainer.innerHTML = `
      <div class="preview-wrapper">
        <div class="file-icon">📄</div>
        <p class="preview-filename">${file.name}</p>
        <p class="preview-filesize">${(file.size / 1024).toFixed(2)} KB</p>
      </div>
    `;
  }
});

document.getElementById("explainInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  let previewContainer = document.getElementById("explainPreview");
  if (!previewContainer) {
    previewContainer = document.createElement("div");
    previewContainer.id = "explainPreview";
    previewContainer.className = "file-preview";
    e.target.parentElement.appendChild(previewContainer);
  }

  previewContainer.innerHTML = "";

  if (file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = function (event) {
      previewContainer.innerHTML = `
        <div class="preview-wrapper">
          <img src="${event.target.result}" alt="Preview" class="preview-image">
          <p class="preview-filename">${file.name}</p>
        </div>
      `;
    };
    reader.readAsDataURL(file);
  } else {
    previewContainer.innerHTML = `
      <div class="preview-wrapper">
        <div class="file-icon">🖼️</div>
        <p class="preview-filename">${file.name}</p>
        <p class="preview-error">Please select an image file</p>
      </div>
    `;
  }
});

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    const activeTab = document.querySelector(".tab-content.active");
    const textarea = activeTab.querySelector("textarea");

    if (textarea && textarea === document.activeElement && activeTab.id !== "chat") {
      e.preventDefault();
      const activeTabId = activeTab.id;

      switch (activeTabId) {
        case "query":
          askQuery();
          break;
        case "prompt":
          generatePrompt();
          break;
        case "visualize":
          visualize();
          break;
      }
    }
  }
});

// Auto-resize textareas
document.querySelectorAll("textarea").forEach((textarea) => {
  textarea.addEventListener("input", function () {
    if (this.id !== "chatInput") {
      this.style.height = "auto";
      this.style.height = this.scrollHeight + "px";
    }
  });
});

// Image click to enlarge
document.addEventListener("click", (e) => {
  if (e.target.classList.contains("generated") || (e.target.tagName === "IMG" && e.target.closest(".tool-result-image"))) {
    const modal = document.createElement("div");
    modal.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.95);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
      cursor: pointer;
      animation: fadeIn 0.3s ease;
    `;

    const img = document.createElement("img");
    img.src = e.target.src;
    img.style.cssText = `
      max-width: 90%;
      max-height: 90%;
      border-radius: 16px;
      box-shadow: 0 0 50px rgba(79,156,255,0.8);
    `;

    modal.appendChild(img);
    document.body.appendChild(modal);

    modal.onclick = () => modal.remove();
  }
});

// ============================================================================
// INITIALIZATION
// ============================================================================

console.log(
  "%c✦ PAGEINSIGHTER AI ✦",
  "font-size: 24px; color: #4f9cff; font-weight: bold; text-shadow: 0 0 10px #4f9cff;",
);
console.log(
  "%cAdvanced Vision Processing System",
  "font-size: 14px; color: #00fff5;",
);
console.log("%cAPI: " + API, "font-size: 12px; color: #e5e7eb;");

// Check API health on load
fetch(`${API}/`)
  .then((r) => r.json())
  .then((d) => {
    console.log(
      "%c✓ API Connected",
      "color: #10b981; font-weight: bold;",
      d,
    );
    showToast(
      "info",
      "System Ready",
      `PageInsighter AI v${d.version} is online and ready`,
    );
  })
  .catch((e) => {
    console.error(
      "%c✗ API Connection Failed",
      "color: #ef4444; font-weight: bold;",
      e,
    );
    showToast(
      "error",
      "Connection Error",
      "Unable to connect to API. Please check if the server is running.",
    );
  });