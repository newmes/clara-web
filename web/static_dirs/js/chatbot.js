/* ═══════════════════════════════════════════════════════
   Shared Chatbot Panel JS — window.chatbot namespace
   Extracted from statistical_analysis.html for reuse
   across Stats, CRF Tables, and SAE Report pages.

   Usage:
     chatbot.init({
       runId:        'RUN_20260201_...',
       endpoint:     '/api/stats/RUN_ID/chat/',
       pageType:     'stats' | 'crf' | 'sae',
       suggestions:  ['Summarize ORR', 'Top AEs?', ...],
       getPageContext: () => ({ mode: 'natural', tab: 'safety' }),
       clickToRef:   (e) => 'KPI: 45.2%' | null,   // optional
     });
   ═══════════════════════════════════════════════════════ */

window.chatbot = (function() {
  'use strict';

  /* ── Private state ── */
  let chatOpen = false;
  let chatHistory = [];
  let chatRefs = [];
  let config = {};

  /* ── DOM helpers ── */
  function $(id) { return document.getElementById(id); }

  function getContainer() {
    return document.querySelector('.doc-container') ||
           document.querySelector('.report-container');
  }

  /* ── Time string ── */
  function chatTimeStr() {
    var now = new Date();
    return now.getHours().toString().padStart(2,'0') + ':' +
           now.getMinutes().toString().padStart(2,'0');
  }

  /* ── Format text (markdown-like) ── */
  function formatChat(text) {
    var s = text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    // @[...] data reference tags as inline chips
    s = s.replace(/@\[([^\]]+)\]/g,
      '<span class="chat-ref-chip" style="display:inline-flex;vertical-align:middle;margin:0 2px;">$1</span>');
    // bullet lists
    s = s.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
    s = s.replace(/(<li>.*<\/li>\n?)+/g, function(m){ return '<ul>'+m+'</ul>'; });
    // numbered lists
    s = s.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
    // bold, inline code
    s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    // line breaks (but not inside <ul>)
    s = s.replace(/\n/g, '<br>');
    s = s.replace(/<\/li><br>/g, '</li>');
    s = s.replace(/<br><li>/g, '<li>');
    return s;
  }

  /* ── Hide welcome splash ── */
  function hideChatWelcome() {
    var w = $('chatWelcome');
    if (w) w.style.display = 'none';
  }

  /* ── Data Reference Chips ── */
  function renderChatRefs() {
    var container = $('chatRefs');
    if (chatRefs.length === 0) {
      container.style.display = 'none';
      container.innerHTML = '';
      return;
    }
    container.style.display = 'flex';
    container.innerHTML = chatRefs.map(function(ref) {
      return '<span class="chat-ref-chip" data-ref="' + ref.replace(/"/g,'&quot;') + '">' +
             ref +
             '<button class="chat-ref-remove" onclick="chatbot.removeRef(this)">&times;</button></span>';
    }).join('');
  }

  function clearChatRefs() {
    chatRefs.length = 0;
    renderChatRefs();
    $('chatInput').placeholder = 'Ask a question...';
  }

  function addRef(label) {
    if (!chatOpen) toggle();
    if (chatRefs.includes(label)) return;
    chatRefs.push(label);
    renderChatRefs();
    var input = $('chatInput');
    input.placeholder = 'Type your question...';
    input.focus();
  }

  function removeRef(btn) {
    var chip = btn.parentElement;
    var label = chip.dataset.ref;
    var idx = chatRefs.indexOf(label);
    if (idx > -1) chatRefs.splice(idx, 1);
    renderChatRefs();
    if (chatRefs.length === 0) {
      $('chatInput').placeholder = 'Ask a question...';
    }
  }

  /* ── Toggle chat panel ── */
  function toggle() {
    chatOpen = !chatOpen;
    var panel = $('chatPanel');
    var container = getContainer();
    panel.classList.toggle('open', chatOpen);
    if (container) container.classList.toggle('chat-open', chatOpen);
    $('chatToggle').style.display = chatOpen ? 'none' : '';
    if (chatOpen) {
      if (container) container.style.marginRight = panel.offsetWidth + 'px';
      setTimeout(function() { $('chatInput').focus(); }, 300);
    } else {
      if (container) container.style.marginRight = '';
    }
  }

  /* ── Append message to chat ── */
  function appendMsg(role, text, queryMeta) {
    hideChatWelcome();
    var container = $('chatMessages');
    var div = document.createElement('div');
    div.className = 'chat-msg ' + role;
    var html = '';

    if (role === 'user') {
      html += '<div class="chat-msg-content">' + formatChat(text) + '</div>';
      html += '<span class="chat-msg-meta">' + chatTimeStr() + '</span>';
    } else if (role === 'assistant') {
      html += '<span class="chat-msg-label"><img src="/static/assets/ui/gemma.png" width="24" height="24" style="opacity:.7;" alt="MedGemma"></span>';
      html += '<div class="chat-bubble">';
      html += '<div class="chat-msg-content">' + formatChat(text) + '</div>';
      if (queryMeta) {
        html += '<div class="chat-query-toggle" onclick="this.nextElementSibling.classList.toggle(\'open\')">'
              + '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" style="margin-right:4px;vertical-align:middle;">'
              + '<path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>'
              + 'Context</div>';
        var sections = (queryMeta.matched_sections || []).join(', ') || '(default)';
        html += '<pre class="chat-query-block">'
              + 'SOURCE     ' + (queryMeta.source || '') + '\n'
              + 'RETRIEVED  ' + sections + '\n'
              + 'ACTIVE TAB ' + (queryMeta.tab || '(none)') + '\n'
              + 'HISTORY    ' + (queryMeta.history_turns || 0) + ' turns\n'
              + '\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
              + (queryMeta.context_data || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
              + '</pre>';
      }
      html += '<span class="chat-msg-meta">' + chatTimeStr() + '</span>';
      html += '</div>';
    } else {
      html = formatChat(text);
    }

    div.innerHTML = html;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  /* ── Send message ── */
  async function send() {
    var input = $('chatInput');
    var msg = input.value.trim();
    if (!msg && chatRefs.length === 0) return;

    // Build full message with data references prepended
    var fullMessage = msg;
    if (chatRefs.length > 0) {
      var refStr = chatRefs.map(function(r) { return '@[' + r + ']'; }).join(' ');
      fullMessage = msg ? refStr + ' ' + msg : refStr;
      clearChatRefs();
    }

    input.value = '';
    input.style.height = 'auto';
    hideChatWelcome();
    appendMsg('user', fullMessage);
    $('chatSuggestions').style.display = 'none';

    var msgContainer = $('chatMessages');
    var typing = document.createElement('div');
    typing.className = 'chat-typing visible';
    typing.id = 'chatTypingActive';
    typing.innerHTML = '<div class="chat-typing-dot"></div><div class="chat-typing-dot"></div><div class="chat-typing-dot"></div>';
    msgContainer.appendChild(typing);
    msgContainer.scrollTop = msgContainer.scrollHeight;
    $('chatSend').disabled = true;
    var sendTime = performance.now();

    // Gather page context from config callback
    var pageCtx = config.getPageContext ? config.getPageContext() : {};

    var body = {
      message: fullMessage,
      page_type: config.pageType || 'stats',
      history: chatHistory,
    };
    // Merge page-specific context
    Object.keys(pageCtx).forEach(function(k) { body[k] = pageCtx[k]; });

    try {
      var resp = await fetch(config.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = await resp.json();
      var latency = ((performance.now() - sendTime) / 1000).toFixed(1);
      typing.remove();
      $('chatSend').disabled = false;

      if (data.error) {
        appendMsg('system-msg error', data.error);
      } else {
        appendMsg('assistant', data.response, data.query || null);
        // Append latency to last assistant message
        var msgs = document.querySelectorAll('.chat-msg.assistant');
        var lastMsg = msgs[msgs.length - 1];
        if (lastMsg) {
          var meta = lastMsg.querySelector('.chat-msg-meta');
          if (meta) meta.textContent += ' \u00b7 ' + latency + 's';
        }
        chatHistory.push({ role: 'user', content: fullMessage });
        chatHistory.push({ role: 'model', content: data.response });
      }
    } catch (e) {
      typing.remove();
      $('chatSend').disabled = false;
      appendMsg('system-msg error', 'Network error: ' + e.message);
    }
    $('chatInput').focus();
  }

  /* ── Send suggestion chip ── */
  function sendSuggestion(el) {
    $('chatInput').value = el.textContent.trim();
    send();
  }

  /* ── Setup suggestions ── */
  function setupSuggestions() {
    var container = $('chatSuggestions');
    if (!container || !config.suggestions || !config.suggestions.length) return;
    container.innerHTML = config.suggestions.map(function(s) {
      return '<button class="chat-suggestion" onclick="chatbot.sendSuggestion(this)">' + s + '</button>';
    }).join('');
  }

  /* ── Setup textarea auto-resize ── */
  function setupAutoResize() {
    var input = $('chatInput');
    if (!input) return;
    input.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 100) + 'px';
    });
    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
  }

  /* ── Setup panel resize handle ── */
  function setupResize() {
    var panel = $('chatPanel');
    var handle = $('chatResizeHandle');
    if (!panel || !handle) return;
    var container = getContainer();
    var startX, startW;

    handle.addEventListener('mousedown', function(e) {
      e.preventDefault();
      startX = e.clientX;
      startW = panel.offsetWidth;
      panel.classList.add('resizing');
      if (container) container.classList.add('chat-resizing');
      document.addEventListener('mousemove', onDrag);
      document.addEventListener('mouseup', onUp);
    });

    function onDrag(e) {
      var w = Math.max(320, Math.min(700, startW + (startX - e.clientX)));
      panel.style.width = w + 'px';
      if (container) container.style.marginRight = w + 'px';
    }
    function onUp() {
      panel.classList.remove('resizing');
      if (container) container.classList.remove('chat-resizing');
      document.removeEventListener('mousemove', onDrag);
      document.removeEventListener('mouseup', onUp);
    }
  }

  /* ── Setup click-to-query delegation ── */
  function setupClickToQuery() {
    if (!config.clickToRef) return;
    var container = getContainer();
    if (!container) return;

    container.addEventListener('click', function(e) {
      if (!chatOpen) return;
      var refLabel = config.clickToRef(e);
      if (refLabel) {
        e.preventDefault();
        e.stopPropagation();
        addRef(refLabel);
      }
    });
  }

  /* ── Init ── */
  function init(cfg) {
    config = cfg || {};
    chatOpen = false;
    chatHistory = [];
    chatRefs = [];
    setupSuggestions();
    setupAutoResize();
    setupResize();
    setupClickToQuery();
  }

  /* ── Public API ── */
  return {
    init: init,
    toggle: toggle,
    send: send,
    sendSuggestion: sendSuggestion,
    addRef: addRef,
    removeRef: removeRef,
    appendMsg: appendMsg,
  };
})();
