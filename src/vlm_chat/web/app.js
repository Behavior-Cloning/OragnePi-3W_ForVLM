'use strict';
const $ = id => document.getElementById(id);
let db, chats = [], current, pendingImage = null, running = null, deleteTargetId = null;
let activeSettings = null, loadingModel = false, applying = false, settingsDirty = false;
const contextSizes = [1024, 2048, 4096, 8192];
const notice = text => { $('notice').textContent = text; $('notice').hidden = !text; };
const fresh = () => ({id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`, title: '新对话', updated: Date.now(), messages: []});

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('orange-vlm', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('chats', {keyPath: 'id'});
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
function storage(mode, action) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('chats', mode);
    const request = action(tx.objectStore('chats'));
    tx.oncomplete = () => resolve(request.result);
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}
async function save(chat) {
  chat.updated = Date.now();
  if (!chats.some(c => c.id === chat.id)) chats.push(chat);
  if (db) {
    try { await storage('readwrite', s => s.put(chat)); }
    catch { notice('浏览器存储已满或不可用：本次对话暂未保存，请先复制需要的内容。'); }
  }
  renderSidebar();
}
function renderSidebar() {
  $('conversations').replaceChildren();
  [...chats].sort((a, b) => b.updated - a.updated).forEach(chat => {
    const row = document.createElement('div'); row.className = 'conversation-row';
    const button = document.createElement('button');
    button.className = 'conversation' + (chat.id === current.id ? ' active' : '');
    button.textContent = chat.title;
    button.title = chat.title;
    button.disabled = !!running;
    button.onclick = () => { current = chat; setImage(null); notice(''); render(); document.body.classList.remove('sidebar-open'); };
    const remove = document.createElement('button');
    remove.className = 'delete-conversation'; remove.textContent = '删除';
    remove.setAttribute('aria-label', `删除对话：${chat.title}`);
    remove.disabled = !!running;
    remove.onclick = () => askDelete(chat);
    row.append(button, remove); $('conversations').append(row);
  });
  $('chat-count').textContent = chats.length;
  $('new-chat').disabled = !!running;
  $('delete-chat').disabled = !!running || !current.messages.length;
}
function askDelete(chat) {
  if (running) return;
  deleteTargetId = chat.id;
  $('delete-target').textContent = chat.title;
  $('delete-dialog').showModal();
}
function messageNode(message) {
  const row = document.createElement('article');
  row.className = `message ${message.role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar'; avatar.textContent = message.role === 'user' ? '你' : '◒';
  const body = document.createElement('div'); body.className = 'message-body';
  const label = document.createElement('div'); label.className = 'message-label';
  label.textContent = message.role === 'user' ? '你' : `橙语${message.model ? ' · ' + message.model : ''}`; body.append(label);
  if (message.image) {
    const img = document.createElement('img'); img.className = 'message-image';
    img.src = message.image; img.alt = '对话中上传的图片'; body.append(img);
  }
  const content = document.createElement('div'); content.className = 'message-text';
  content.textContent = message.text || (message.state === 'generating' ? '正在看图思考…' : '');
  if (message.state === 'generating' && !message.text) content.classList.add('thinking');
  const meta = document.createElement('div'); meta.className = 'message-meta';
  if (message.state === 'stopped') meta.textContent = '已停止生成';
  if (message.state === 'error') meta.textContent = '生成失败 · 可重新发送问题';
  if (message.state === 'complete' && message.seconds) meta.textContent = `${message.seconds.toFixed(1)} 秒`;
  if (message.ttft != null) meta.textContent += ` · 首字 ${message.ttft.toFixed(2)} 秒`;
  if (message.tps) meta.textContent += ` · ${message.tps.toFixed(1)} token/s`;
  if (message.finishReason === 'length') meta.textContent += ' · 已达到输出长度或上下文限制，可追问“继续”或新建对话';
  if (message.excluded) meta.textContent += ' · 已移出推理上下文（历史仍保留）';
  body.append(content, meta); row.append(avatar, body);
  return {row, content, meta};
}
function render() {
  renderSidebar(); $('welcome').hidden = current.messages.length > 0;
  $('messages').replaceChildren(...current.messages.map((m,i) => messageNode({...m, excluded:i < (current.contextStart || 0)}).row));
  $('context-hint').textContent = current.contextStart ? `较早的 ${current.contextStart} 条消息已移出推理上下文，其中的图片不再参与回答。历史仍保留。` : '';
  updateControls(); scrollDown();
}
function scrollDown() { $('scroll-area').scrollTop = $('scroll-area').scrollHeight; }
function updateControls() {
  $('send').hidden = !!running; $('stop').hidden = !running;
  $('send').disabled = !!running || loadingModel || applying || settingsDirty || (!$('prompt').value.trim() && !pendingImage);
  for (const id of ['model-select', 'context-range', 'apply-settings']) $(id).disabled = !!running || loadingModel || applying;
  $('upload').disabled = !!running || activeSettings?.vision === false; $('remove-image').disabled = !!running;
  $('prompt').disabled = !!running;
}
function setImage(value) {
  pendingImage = value; $('attachment').hidden = !value;
  $('attachment-preview').src = value ? value.data : '';
  $('attachment-name').textContent = value ? value.name : '';
  $('file').value = ''; updateControls();
}
async function attach(file) {
  if (activeSettings?.vision === false) { notice('当前为纯文字模型，请选择视觉模型再上传图片。'); return; }
  if (running || !file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { notice('请上传 JPEG、PNG 或 WebP 图片。'); return; }
  if (file.size > 20 * 1024 * 1024) { notice('原图请小于 20 MB。'); return; }
  try {
    const img = await createImageBitmap(file);
    const ratio = Math.min(1, 768 / Math.max(img.width, img.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(img.width * ratio)); canvas.height = Math.max(1, Math.round(img.height * ratio));
    const ctx = canvas.getContext('2d'); ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height); img.close();
    setImage({data: canvas.toDataURL('image/jpeg', .85), name: file.name || '粘贴的图片'}); notice('');
  } catch { notice('图片读取失败，请尝试其他图片。'); }
}
function apiMessages(messages) {
  return messages.filter(m => m.role === 'user' || m.text).map(m => ({role: m.role,
    content: m.image ? [{type: 'image_url', image_url: {url: m.image}}, {type: 'text', text: m.text}] : m.text}));
}
async function consumeStream(response, onText) {
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', done = false, finishReason = null, timings = null;
  function event(raw) {
    const data = raw.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
    if (!data) return;
    if (data === '[DONE]') { done = true; return; }
    const item = JSON.parse(data);
    if (item.error) throw new Error(typeof item.error === 'string' ? item.error : item.error.message || '模型错误');
    if (item.timings) timings = item.timings;
    const delta = item.choices?.[0]?.delta?.content;
    if (item.choices?.[0]?.finish_reason) finishReason = item.choices[0].finish_reason;
    if (delta) onText(delta);
  }
  try {
    while (!done) {
      const chunk = await reader.read();
      buffer += decoder.decode(chunk.value || new Uint8Array(), {stream: !chunk.done});
      buffer = buffer.replace(/\r\n/g, '\n');
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) { event(buffer.slice(0, boundary)); buffer = buffer.slice(boundary + 2); }
      if (chunk.done) { if (buffer.trim()) event(buffer); break; }
    }
    if (!done) throw new Error('推理连接中断，当前只保存了部分回答。');
    return {finishReason, timings};
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
async function send(event) {
  event.preventDefault();
  if (running || loadingModel || applying || settingsDirty) return;
  let text = $('prompt').value.trim();
  if (!text && !pendingImage) return;
  text ||= 'Describe this image.';
  const chat = current;
  const user = {role: 'user', text, image: pendingImage?.data || null};
  chat.messages.push(user);
  if (chat.messages.length === 1) chat.title = text.slice(0, 32);
  const sentIndices = chat.messages.map((m,i) => i).filter(i => i >= (chat.contextStart || 0) && (chat.messages[i].role === 'user' || chat.messages[i].text));
  const messages = apiMessages(sentIndices.map(i => chat.messages[i]));
  const assistant = {role: 'assistant', text: '', state: 'generating', model: $('model-name').textContent};
  chat.messages.push(assistant);
  const controller = new AbortController(); running = controller;
  $('prompt').value = ''; setImage(null); notice(''); render();
  await save(chat);
  const node = $('messages').lastElementChild, content = node.querySelector('.message-text');
  const start = performance.now();
  try {
    const response = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({messages, model_id:activeSettings?.model_id, context:activeSettings?.context}), signal: controller.signal});
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `请求失败 (${response.status})`);
    }
    const dropped = Number(response.headers.get('X-Context-Dropped') || 0);
    if (dropped > 0 && sentIndices[dropped] !== undefined) {
      chat.contextStart = sentIndices[dropped];
      await save(chat);
      notice(`窗口已满：已移出 ${dropped} 条最早消息；历史保留，但其中的图片不再参与推理。`);
    }
    const result = await consumeStream(response, token => {
      if (assistant.ttft == null) assistant.ttft = (performance.now() - start) / 1000;
      const nearBottom = $('scroll-area').scrollHeight - $('scroll-area').scrollTop - $('scroll-area').clientHeight < 130;
      assistant.text += token; content.textContent = assistant.text; content.classList.remove('thinking');
      if (nearBottom) scrollDown();
    });
    assistant.finishReason = result.finishReason;
    assistant.tps = result.timings?.predicted_per_second;
    assistant.state = 'complete';
    if (!assistant.text) throw new Error('模型未返回内容，请尝试简短问题。');
  } catch (error) {
    assistant.state = error.name === 'AbortError' ? 'stopped' : 'error';
    if (assistant.state === 'error') notice(error.message);
  } finally {
    assistant.seconds = (performance.now() - start) / 1000;
    running = null; await save(chat); render(); $('prompt').focus();
  }
}
async function health() {
  try {
    const response = await fetch('/api/health', {signal: AbortSignal.timeout(5000)});
    const state = await response.json();
    if (state.metrics) {
      const m = state.metrics, num = (value, digits=1) => value == null ? '—' : value.toFixed(digits);
      $('runtime-metrics').textContent = `整机 CPU ${num(m.system_cpu_percent)}%（满载 100%） · 模型 CPU ${num(m.model_cpu_percent)}%（单核 100%）\n模型 RSS ${num(m.model_rss_mib,0)} MiB · 可用内存 ${num(m.ram_available_mib,0)} / ${num(m.ram_total_mib,0)} MiB · Swap ${num(m.swap_used_mib)} MiB\n温度 ${num(m.temperature_c)}°C · 磁盘可用 ${num(m.disk_available_gib)} / ${num(m.disk_total_gib)} GiB · 每约 3 秒更新`;
    }
    loadingModel = !!state.loading || !!state.maintenance;
    if (state.models) {
      $('model-controls').hidden = false;
      const signature = JSON.stringify(state.models);
      if ($('model-select').dataset.signature !== signature) {
        const draft = $('model-select').value;
        $('model-select').replaceChildren(...state.models.map(m => {
          const option = document.createElement('option'); option.value = m.id;
          option.textContent = m.name + (m.available ? '' : '（尚未就绪）'); option.disabled = !m.available; return option;
        }));
        $('model-select').dataset.signature = signature;
        if (settingsDirty) $('model-select').value = draft;
      }
      activeSettings = {model_id:state.model_id, context:state.context, vision:state.vision};
      if (!settingsDirty && !applying) {
        $('model-select').value = state.model_id;
        $('context-range').value = contextSizes.indexOf(state.context);
        $('context-value').textContent = state.context;
      }
      if (!applying && !settingsDirty) $('settings-status').textContent = state.maintenance ? '统一性能评测进行中，暂时暂停对话与配置切换。' : state.loading
        ? '正在校验并加载模型，请等待；暂时不能发送消息。'
        : state.error || `当前 ${state.context} tokens · 切换对所有访客生效 · 超窗移出旧问答`;
    }
    if (state.model) {
      $('model-name').textContent = state.model;
      document.title = `橙语 · ${state.model}`;
      const footer = document.querySelector('.footer-note');
      footer.firstChild.textContent = `${state.model} · 本地推理 `;
      document.querySelector('.model-note').textContent = state.vision === false ? '纯文字对话模型，可用于陪伴角色扮演。当前不支持图片；旧对话中的图片需要切回视觉模型处理。' : state.model.startsWith('InternVL')
        ? '支持中文多轮对话。首次看图可能需数分钟，同图追问通常更快；细节和推理仍可能出错。'
        : state.model.startsWith('Qwen') ? '3B 量化模型，支持中文与图片。图片和长上下文计算较慢，算术及细节仍可能出错。'
        : '这是一个轻量实验模型，英文提问通常更可靠。它可能看错细节，回答仅供参考。';
    }
    $('status').className = state.ready ? '' : 'offline';
    $('status').replaceChildren();
    const dot = document.createElement('i'); dot.className = 'dot';
    $('status').append(dot, state.maintenance ? '性能评测中' : state.loading ? '模型加载中…' : state.ready ? '模型已就绪' : '模型未就绪');
    updateControls();
  } catch { $('status').className = 'offline'; $('status').textContent = '连接断开'; }
}
async function applySettings() {
  if (running || applying || loadingModel) return;
  applying = true; updateControls();
  $('settings-status').textContent = '正在校验权重并重新加载模型，请等待，通常需要数秒到一分钟…';
  try {
    const response = await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model_id:$('model-select').value, context:contextSizes[Number($('context-range').value)]})});
    const state = await response.json();
    if (!response.ok) throw new Error(state.error || '加载失败');
    settingsDirty = false;
    notice('模型与窗口已应用。旧对话仍保留，可新建对话比较模型表现。');
  } catch (error) { notice(error.message); settingsDirty = false; }
  finally { applying = false; await health(); updateControls(); }
}
async function init() {
  current = fresh();
  try {
    db = await openDB(); chats = await storage('readonly', s => s.getAll());
    chats.forEach(chat => chat.messages.forEach(m => { if (m.state === 'generating') m.state = 'stopped'; }));
    if (chats.length) current = [...chats].sort((a, b) => b.updated - a.updated)[0];
  } catch { notice('浏览器不允许本地存储；本次会话仅保留到页面关闭。'); }
  $('composer').addEventListener('submit', send);
  const changeSettings = () => {
    settingsDirty = true;
    $('context-value').textContent = contextSizes[Number($('context-range').value)];
    $('settings-status').textContent = '设置尚未应用，点击“应用并加载”后继续对话。';
    updateControls();
  };
  $('model-select').onchange = changeSettings;
  $('context-range').oninput = changeSettings;
  $('apply-settings').onclick = applySettings;
  $('prompt').oninput = updateControls;
  $('prompt').onkeydown = event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); }
  };
  $('new-chat').onclick = () => { if (running) return; current = fresh(); setImage(null); notice(''); render(); document.body.classList.remove('sidebar-open'); $('prompt').focus(); };
  $('menu').onclick = () => document.body.classList.toggle('sidebar-open');
  $('upload').onclick = () => $('file').click();
  $('file').onchange = event => attach(event.target.files[0]);
  $('remove-image').onclick = () => setImage(null);
  $('prompt').onpaste = event => { const file = [...event.clipboardData.items].find(i => i.type.startsWith('image/'))?.getAsFile(); if (file) { event.preventDefault(); attach(file); } };
  $('stop').onclick = () => running?.abort();
  $('delete-chat').onclick = () => askDelete(current);
  $('cancel-delete').onclick = () => $('delete-dialog').close();
  $('confirm-delete').onclick = async () => {
    if (running || !deleteTargetId) return;
    const target = deleteTargetId;
    $('confirm-delete').disabled = true;
    try {
      if (db) await storage('readwrite', s => s.delete(target));
      chats = chats.filter(c => c.id !== target);
      if (current.id === target) { current = fresh(); setImage(null); }
      render(); notice('');
    } catch { notice('删除失败，请检查浏览器存储设置。'); }
    $('confirm-delete').disabled = false; deleteTargetId = null;
    $('delete-dialog').close();
  };
  document.querySelectorAll('[data-prompt]').forEach(b => b.onclick = () => { $('prompt').value = b.dataset.prompt; updateControls(); $('file').click(); });
  render(); health(); setInterval(health, 3000);
}
init();
