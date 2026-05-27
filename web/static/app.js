/* ── VibeBlade Chat — Application Logic ─────────────────────────── */

(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────
  const state = {
    conversations: [],
    activeConvId: null,
    isStreaming: false,
    abortController: null,
    settings: {
      model: "qwen3.6-27b-mtp",
      temperature: 0.7,
      max_tokens: 4096,
      top_p: 0.9,
      top_k: 40,
      system_prompt: "",
    },
  };

  // ── DOM refs ───────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dom = {
    sidebar: $("#sidebar"),
    sidebarToggle: $("#sidebar-toggle"),
    convList: $("#conversation-list"),
    searchInput: $("#search-input"),
    newChatBtn: $("#new-chat-btn"),
    chatTitle: $("#chat-title"),
    messages: $("#messages"),
    input: $("#message-input"),
    sendBtn: $("#send-btn"),
    stopBtn: $("#stop-btn"),
    tokenCounter: $("#token-counter"),
    settingsBtn: $("#settings-btn"),
    settingsPanel: $("#settings-panel"),
    settingsClose: $("#settings-close"),
    settingsOverlay: $("#settings-overlay"),
    modelBadge: $("#model-badge"),
    clearAllBtn: $("#clear-all-btn"),
  };

  // ── Marked config ──────────────────────────────────────────────
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function (code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(code, { language: lang }).value;
        } catch (_) {}
      }
      return hljs.highlightAuto(code).value;
    },
  });

  // Override renderer for code blocks (add copy button + language label)
  const defaultRenderer = new marked.Renderer();
  defaultRenderer.code = function (obj) {
    // marked v12 passes {text, lang, escaped}
    const code = typeof obj === "object" ? (obj.text || "") : obj;
    const lang = typeof obj === "object" ? (obj.lang || "") : "";
    const langLabel = lang || "code";
    let highlighted;
    if (lang && hljs.getLanguage(lang)) {
      try {
        highlighted = hljs.highlight(code, { language: lang }).value;
      } catch (_) {
        highlighted = escapeHtml(code);
      }
    } else {
      highlighted = hljs.highlightAuto(code).value;
    }
    return (
      '<pre><div class="code-header"><span>' +
      langLabel +
      '</span><button class="btn-copy-code" onclick="VibeBlade.copyCode(this)">Copy</button></div><code>' +
      highlighted +
      "</code></pre>"
    );
  };
  marked.setOptions({ renderer: defaultRenderer });

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ── API helpers ────────────────────────────────────────────────
  async function api(path, options = {}) {
    const resp = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || body.error || `HTTP ${resp.status}`);
    }
    return resp;
  }

  // ── Conversation management ────────────────────────────────────
  async function loadConversations() {
    try {
      state.conversations = await api("/api/conversations").then((r) => r.json());
      renderConversationList();
    } catch (e) {
      console.error("Failed to load conversations:", e);
    }
  }

  function renderConversationList(filter = "") {
    const filtered = filter
      ? state.conversations.filter((c) =>
          c.title.toLowerCase().includes(filter.toLowerCase())
        )
      : state.conversations;

    dom.convList.innerHTML =
      filtered.length === 0
        ? '<div style="padding:20px;text-align:center;color:var(--text-tertiary);font-size:13px">No conversations</div>'
        : filtered
            .map(
              (c) => `
        <div class="conv-item${c.id === state.activeConvId ? " active" : ""}" data-id="${c.id}" onclick="VibeBlade.selectConv('${c.id}')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16" style="flex-shrink:0;color:var(--text-tertiary)">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
          </svg>
          <span class="conv-item-text">${escapeHtml(c.title)}</span>
          <div class="conv-item-actions">
            <button class="conv-item-delete" onclick="event.stopPropagation();VibeBlade.deleteConv('${c.id}')" title="Delete">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6m5 0V4a1 1 0 011-1h2a1 1 0 011 1v2"/>
              </svg>
            </button>
          </div>
        </div>
      `
            )
            .join("");
  }

  async function createConversation() {
    try {
      const conv = await api("/api/conversations", {
        method: "POST",
        body: JSON.stringify({}),
      }).then((r) => r.json());
      state.conversations.unshift(conv);
      state.activeConvId = conv.id;
      renderConversationList();
      renderMessages([]);
      dom.chatTitle.textContent = conv.title;
      dom.input.focus();
    } catch (e) {
      console.error("Failed to create conversation:", e);
    }
  }

  async function selectConversation(convId) {
    if (state.isStreaming) return;
    state.activeConvId = convId;
    renderConversationList();

    try {
      const conv = await api(`/api/conversations/${convId}`).then((r) =>
        r.json()
      );
      dom.chatTitle.textContent = conv.title;
      renderMessages(conv.messages || []);

      // Load settings from conversation
      if (conv.temperature !== undefined)
        state.settings.temperature = conv.temperature;
      if (conv.max_tokens !== undefined)
        state.settings.max_tokens = conv.max_tokens;
      if (conv.top_p !== undefined) state.settings.top_p = conv.top_p;
      if (conv.top_k !== undefined) state.settings.top_k = conv.top_k;
      if (conv.system_prompt !== undefined)
        state.settings.system_prompt = conv.system_prompt || "";
      if (conv.model) state.settings.model = conv.model;
      updateSettingsUI();
      updateTokenCounter(conv.messages || []);
    } catch (e) {
      console.error("Failed to load conversation:", e);
    }

    // Close sidebar on mobile
    dom.sidebar.classList.remove("open");
  }

  async function deleteConversation(convId) {
    try {
      await api(`/api/conversations/${convId}`, { method: "DELETE" });
      state.conversations = state.conversations.filter((c) => c.id !== convId);
      if (state.activeConvId === convId) {
        state.activeConvId = null;
        renderMessages([]);
        dom.chatTitle.textContent = "New Chat";
      }
      renderConversationList();
    } catch (e) {
      console.error("Failed to delete conversation:", e);
    }
  }

  // ── Message rendering ──────────────────────────────────────────
  function renderMessages(messages) {
    if (messages.length === 0) {
      dom.messages.innerHTML = `
        <div class="empty-state">
          <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
          </svg>
          <h3>VibeBlade Chat</h3>
          <p>Qwen3.6-27B with NEXTN speculative decoding.<br>Start a conversation below.</p>
        </div>
      `;
      dom.tokenCounter.textContent = "";
      return;
    }

    dom.messages.innerHTML = messages
      .map((msg) => renderMessage(msg))
      .join("");
    updateTokenCounter(messages);
    scrollToBottom();
  }

  function renderMessage(msg) {
    const isUser = msg.role === "user";
    const contentHtml = isUser ? escapeHtml(msg.content) : marked.parse(msg.content);
    const tokenStr = msg.tokens ? `${msg.tokens} tok` : "";

    return `
      <div class="message ${msg.role}">
        <div class="message-inner" style="position:relative">
          <div class="message-role">
            <span class="message-role-name">${isUser ? "You" : "VibeBlade"}</span>
            ${tokenStr ? '<span class="message-role-tokens">' + tokenStr + "</span>" : ""}
          </div>
          <div class="message-content">${contentHtml}</div>
          ${!isUser ? '<button class="btn-copy-msg" onclick="VibeBlade.copyMessage(this)" title="Copy"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg></button>' : ""}
        </div>
      </div>
    `;
  }

  function appendStreamingMessage() {
    // Remove empty state if present
    const empty = dom.messages.querySelector(".empty-state");
    if (empty) empty.remove();

    const el = document.createElement("div");
    el.className = "message assistant";
    el.id = "streaming-msg";
    el.innerHTML = `
      <div class="message-inner">
        <div class="message-role">
          <span class="message-role-name">VibeBlade</span>
          <span class="message-role-tokens" id="stream-tokens"></span>
        </div>
        <div class="message-content streaming-cursor" id="stream-content"></div>
      </div>
    `;
    dom.messages.appendChild(el);
    scrollToBottom();
    return el;
  }

  function updateStreamingContent(content) {
    const el = document.getElementById("stream-content");
    if (el) {
      el.innerHTML = marked.parse(content);
      scrollToBottom();
    }
  }

  function finalizeStreamingMessage(content, tokens, convId) {
    const el = document.getElementById("streaming-msg");
    if (el) {
      el.id = "";
      const contentEl = el.querySelector(".message-content");
      if (contentEl) contentEl.classList.remove("streaming-cursor");
      const tokenEl = el.querySelector(".message-role-tokens");
      if (tokenEl && tokens) tokenEl.textContent = `${tokens} tok`;

      // Add copy button
      const inner = el.querySelector(".message-inner");
      inner.insertAdjacentHTML(
        "beforeend",
        '<button class="btn-copy-msg" onclick="VibeBlade.copyMessage(this)" title="Copy"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg></button>'
      );
    }

    // Update conversation list if title changed
    if (convId && state.activeConvId !== convId) {
      state.activeConvId = convId;
      loadConversations();
    }
    renderConversationList();
    updateTotalTokens();
  }

  function appendError(message) {
    const empty = dom.messages.querySelector(".empty-state");
    if (empty) empty.remove();

    const el = document.createElement("div");
    el.className = "message-error";
    el.textContent = message;
    dom.messages.appendChild(el);
    scrollToBottom();
  }

  // ── Streaming chat ─────────────────────────────────────────────
  async function sendMessage() {
    const text = dom.input.value.trim();
    if (!text || state.isStreaming) return;

    // Create conversation if none active
    if (!state.activeConvId) {
      const conv = await api("/api/conversations", {
        method: "POST",
        body: JSON.stringify({}),
      }).then((r) => r.json());
      state.conversations.unshift(conv);
      state.activeConvId = conv.id;
      dom.chatTitle.textContent = conv.title;
    }

    // Add user message to UI immediately
    const empty = dom.messages.querySelector(".empty-state");
    if (empty) empty.remove();

    const userMsgEl = document.createElement("div");
    userMsgEl.className = "message user";
    userMsgEl.innerHTML = `
      <div class="message-inner">
        <div class="message-role"><span class="message-role-name">You</span></div>
        <div class="message-content">${escapeHtml(text)}</div>
      </div>
    `;
    dom.messages.appendChild(userMsgEl);

    dom.input.value = "";
    dom.input.style.height = "auto";
    setStreamingState(true);
    scrollToBottom();

    // Stream response
    state.abortController = new AbortController();
    let fullContent = "";
    let tokenCount = 0;
    let streamEl = null;

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: state.activeConvId,
          message: text,
          model: state.settings.model,
          temperature: state.settings.temperature,
          max_tokens: state.settings.max_tokens,
          top_p: state.settings.top_p,
          top_k: state.settings.top_k,
          system_prompt: state.settings.system_prompt || null,
        }),
        signal: state.abortController.signal,
      });

      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.error || body.detail || `HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const dataStr = line.slice(6).trim();
          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);

            if (data.error) {
              appendError(data.error);
              fullContent = "";
              break;
            }

            if (data.done) {
              tokenCount = data.tokens || tokenCount;
              finalizeStreamingMessage(
                fullContent,
                tokenCount,
                data.conversation_id
              );
              break;
            }

            if (data.content) {
              if (!streamEl) streamEl = appendStreamingMessage();
              fullContent += data.content;
              tokenCount++;
              updateStreamingContent(fullContent);

              // Update token counter in header
              const streamTokens = document.getElementById("stream-tokens");
              if (streamTokens) streamTokens.textContent = `${tokenCount} tok`;
            }
          } catch (e) {
            // Ignore JSON parse errors for partial chunks
          }
        }
      }
    } catch (e) {
      if (e.name === "AbortError") {
        // User stopped generation
        if (fullContent) {
          finalizeStreamingMessage(fullContent, tokenCount, null);
        }
      } else {
        appendError(e.message);
      }
    } finally {
      setStreamingState(false);
      state.abortController = null;
    }
  }

  function stopStreaming() {
    if (state.abortController) {
      state.abortController.abort();
    }
  }

  function setStreamingState(streaming) {
    state.isStreaming = streaming;
    dom.sendBtn.classList.toggle("hidden", streaming);
    dom.stopBtn.classList.toggle("hidden", !streaming);
    dom.input.disabled = streaming;
  }

  // ── Token counter ──────────────────────────────────────────────
  function updateTokenCounter(messages) {
    const total = messages.reduce((sum, m) => sum + (m.tokens || 0), 0);
    dom.tokenCounter.textContent = total > 0 ? `${total} tokens` : "";
  }

  function updateTotalTokens() {
    const conv = state.conversations.find((c) => c.id === state.activeConvId);
    if (conv) updateTokenCounter(conv.messages || []);
  }

  // ── Settings ───────────────────────────────────────────────────
  function loadSettings() {
    const saved = localStorage.getItem("vibeblade_settings");
    if (saved) {
      try {
        Object.assign(state.settings, JSON.parse(saved));
      } catch (_) {}
    }
    updateSettingsUI();
  }

  function saveSettings() {
    localStorage.setItem("vibeblade_settings", JSON.stringify(state.settings));
  }

  function updateSettingsUI() {
    const s = state.settings;
    $("#setting-model").value = s.model;
    $("#setting-temp").value = s.temperature;
    $("#temp-value").textContent = s.temperature;
    $("#setting-maxtok").value = s.max_tokens;
    $("#maxtok-value").textContent = s.max_tokens;
    $("#setting-topp").value = s.top_p;
    $("#topp-value").textContent = s.top_p;
    $("#setting-topk").value = s.top_k;
    $("#topk-value").textContent = s.top_k;
    $("#setting-system").value = s.system_prompt || "";
    dom.modelBadge.textContent = "Qwen3.6-27B-FP8";
  }

  function openSettings() {
    dom.settingsPanel.classList.remove("hidden");
    dom.settingsOverlay.classList.remove("hidden");
  }

  function closeSettings() {
    dom.settingsPanel.classList.add("hidden");
    dom.settingsOverlay.classList.add("hidden");
    saveSettings();
  }

  // ── Copy helpers ───────────────────────────────────────────────
  function copyCode(btn) {
    const pre = btn.closest("pre");
    const code = pre.querySelector("code");
    navigator.clipboard.writeText(code.textContent).then(() => {
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = "Copy"), 1500);
    });
  }

  function copyMessage(btn) {
    const content = btn.closest(".message-inner").querySelector(".message-content");
    navigator.clipboard.writeText(content.textContent).then(() => {
      btn.style.opacity = "1";
      btn.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="20 6 9 17 4 12"/></svg>';
      setTimeout(() => {
        btn.style.opacity = "";
        btn.innerHTML =
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
      }, 1500);
    });
  }

  // ── Input handling ─────────────────────────────────────────────
  function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
  }

  function scrollToBottom() {
    dom.messages.scrollTop = dom.messages.scrollHeight;
  }

  // ── Event bindings ─────────────────────────────────────────────
  dom.input.addEventListener("input", () => autoResize(dom.input));

  dom.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  dom.sendBtn.addEventListener("click", sendMessage);
  dom.stopBtn.addEventListener("click", stopStreaming);

  dom.newChatBtn.addEventListener("click", () => {
    if (state.isStreaming) return;
    createConversation();
  });

  dom.settingsBtn.addEventListener("click", openSettings);
  dom.settingsClose.addEventListener("click", closeSettings);
  dom.settingsOverlay.addEventListener("click", closeSettings);

  dom.sidebarToggle.addEventListener("click", () => {
    dom.sidebar.classList.toggle("open");
  });

  dom.searchInput.addEventListener("input", (e) => {
    renderConversationList(e.target.value);
  });

  // Settings sliders
  ["temp", "maxtok", "topp", "topk"].forEach((key) => {
    const input = $(`#setting-${key}`);
    const display = $(`#${key}-value`);
    input.addEventListener("input", () => {
      const map = {
        temp: "temperature",
        maxtok: "max_tokens",
        topp: "top_p",
        topk: "top_k",
      };
      state.settings[map[key]] = parseFloat(input.value);
      display.textContent = input.value;
    });
  });

  $("#setting-model").addEventListener("change", (e) => {
    state.settings.model = e.target.value;
  });

  $("#setting-system").addEventListener("input", (e) => {
    state.settings.system_prompt = e.target.value;
  });

  dom.clearAllBtn.addEventListener("click", async () => {
    if (!confirm("Delete all conversations? This cannot be undone.")) return;
    try {
      for (const conv of state.conversations) {
        await fetch(`/api/conversations/${conv.id}`, { method: "DELETE" });
      }
      state.conversations = [];
      state.activeConvId = null;
      renderConversationList();
      renderMessages([]);
      dom.chatTitle.textContent = "New Chat";
    } catch (e) {
      console.error("Failed to clear conversations:", e);
    }
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Ctrl+N: new chat
    if ((e.ctrlKey || e.metaKey) && e.key === "n") {
      e.preventDefault();
      if (!state.isStreaming) createConversation();
    }
    // Escape: close settings
    if (e.key === "Escape") {
      closeSettings();
      dom.sidebar.classList.remove("open");
    }
  });

  // ── Public API (for inline onclick handlers) ───────────────────
  window.VibeBlade = {
    selectConv: selectConversation,
    deleteConv: deleteConversation,
    copyCode: copyCode,
    copyMessage: copyMessage,
  };

  // ── Init ───────────────────────────────────────────────────────
  async function init() {
    loadSettings();

    // Load models
    try {
      const models = await api("/api/models").then((r) => r.json());
      const select = $("#setting-model");
      select.innerHTML = models
        .map((m) => `<option value="${m}">${m}</option>`)
        .join("");
      select.value = state.settings.model || models[0] || "qwen3.6-27b-mtp";
    } catch (_) {}

    await loadConversations();

    // Show empty state
    renderMessages([]);
    dom.input.focus();
  }

  init();
})();
