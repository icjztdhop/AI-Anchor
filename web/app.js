const $ = (id) => document.getElementById(id);

// =========================
// 随机动作
// =========================
const AUDIO_BLOCK_SIZE = 2;
const blockMap = new Map();

function getBlockIdBySeq(seq) {
  // seq从1开始：1-4 => 0, 5-8 => 1 ...
  return Math.floor((seq - 1) / AUDIO_BLOCK_SIZE);
}

function ensureBlock(blockId) {
  if (!blockMap.has(blockId)) {
    blockMap.set(blockId, { seqs: [], hasTrigger: false, pickSeq: null });
  }
  return blockMap.get(blockId);
}
// =========================
// UI
// =========================
const chatEl = $("chat");
const statusEl = $("status");
const qsizeEl = $("qsize");
const segListEl = $("segList");
const nowTextEl = $("nowText");

const btnConnect = $("btnConnect");
const btnSend = $("btnSend");
const btnStop = $("btnStop");
const btnPause = $("btnPause");
const btnResume = $("btnResume");
const btnSkip = $("btnSkip");

const player = $("player");

// =========================
// WS
// =========================
let ws = null;

// =========================
// Playback state (ordered by seq)
// =========================
let nextSeq = 1;
let playing = false;

const audioReady = new Map(); // seq -> absolute audio_url
const segMap = new Map(); // seq -> { text, emotion, shouldAct, status, statusLabel, audio_url }
const motionTriggeredSeq = new Set(); // seq already triggered motion

// =========================
// Bubble state (for delta append)
// =========================
let currentBotBubbleTextEl = null;

// =========================
// Emotion config (load from /config)
// =========================
let runtimeConfig = null;
let isEmotionEnabled = false;
let emotionMotionMap = null;

// 兜底动作映射（如果 /config 拉不到）
const FALLBACK_EMOTION_MOTION_MAP = {
  "正常": [{ group: "Tap", index: 0 }],
  "开心": [{ group: "Tap", index: 1 }, { group: "FlickUp", index: 0 }],
  "失落": [{ group: "Flick@Body", index: 0 }],
  "疑惑": [{ group: "Flick", index: 0 }],
  "惊讶": [{ group: "FlickDown", index: 0 }],
  "生气": [{ group: "Tap@Body", index: 0 }],
};
// 防止 DEFAULT_EMOTION_MOTION_MAP 未定义导致整页 JS 直接报错（按钮全失效）
const DEFAULT_EMOTION_MOTION_MAP = FALLBACK_EMOTION_MOTION_MAP;

async function loadRuntimeConfig(apiBase) {
  try {
    const r = await fetch(apiBase + "/config", { cache: "no-store" });
    if (!r.ok) throw new Error("config http " + r.status);
    runtimeConfig = await r.json();

    isEmotionEnabled = !!runtimeConfig.emotion_enabled;
    emotionMotionMap = runtimeConfig.emotion_motion_map || null;

    if (!emotionMotionMap || Object.keys(emotionMotionMap).length === 0) {
      emotionMotionMap = FALLBACK_EMOTION_MOTION_MAP;
    }

    console.log("[Config] emotion_enabled=", isEmotionEnabled, "emotion_motion_map keys=", Object.keys(emotionMotionMap || {}));
  } catch (e) {
    console.warn("[Config] load /config failed, use fallback:", e);
    isEmotionEnabled = true; // 前端依然允许动作
    emotionMotionMap = FALLBACK_EMOTION_MOTION_MAP;
  }
}

function pickMotionForEmotion(emotion) {
  const emo = (emotion || "").trim() || "正常";
  let opts = (emotionMotionMap && emotionMotionMap[emo]) || null;
  if (!opts || !opts.length) opts = DEFAULT_EMOTION_MOTION_MAP[emo] || null;
  if (!opts || !opts.length) opts = (emotionMotionMap && emotionMotionMap["正常"]) || DEFAULT_EMOTION_MOTION_MAP["正常"] || null;
  if (!opts || !opts.length) return null;
  return opts[Math.floor(Math.random() * opts.length)];
}

function triggerMotionWithRetry(emotion, tries = 10, delayMs = 150) {
  if (!isEmotionEnabled) return;

  const opt = pickMotionForEmotion(emotion);
  if (!opt) return;

  let n = 0;
  const timer = setInterval(() => {
    n += 1;
    const ok = window.Live2DController?.playMotion?.(opt.group, opt.index, 3);
    if (ok) {
      clearInterval(timer);
    } else if (n >= tries) {
      clearInterval(timer);
      console.warn("[Motion] retry exhausted:", opt);
    }
  }, delayMs);
}

function pickEmotionForInjection(seg) {
  // 1) 优先用本段自带 emotion
  const e = (seg?.emotion || "").trim();
  if (e) return e;

  // 2) 优先用“正常”
  if (emotionMotionMap && emotionMotionMap["正常"]) return "正常";

  // 3) 随机选一个可用 emotion key
  const keys = emotionMotionMap ? Object.keys(emotionMotionMap) : [];
  if (keys.length > 0) return keys[Math.floor(Math.random() * keys.length)];

  // 4) 最后兜底
  return "正常";
}
// =========================
// Helpers
// =========================
function setStatus(text, level = "muted") {
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.classList.remove("ok", "warn", "err", "muted");
  statusEl.classList.add(level);
}

function setButtons(connected) {
  if (btnSend) btnSend.disabled = !connected;
  if (btnStop) btnStop.disabled = !connected;
  if (btnPause) btnPause.disabled = !connected;
  if (btnResume) btnResume.disabled = !connected;
  if (btnSkip) btnSkip.disabled = !connected;
}

function setQSize() {
  if (!qsizeEl) return;
  let count = 0;
  for (const [seq] of audioReady) {
    if (seq >= nextSeq) count++;
  }
  qsizeEl.textContent = String(count);
}

function chatIsPre() {
  return chatEl && chatEl.tagName && chatEl.tagName.toLowerCase() === "pre";
}

// =========================
// Bubble chat (name shown ABOVE bubble)
// =========================
function appendBubble(kind, text, name = "") {
  if (!chatEl) return;

  // Fallback: <pre> mode
  if (chatIsPre()) {
    const prefix =
      kind === "user"
        ? (name ? `${name}: ` : "YOU: ")
        : kind === "bot"
        ? (name ? `${name}: ` : "BOT: ")
        : "SYS: ";
    chatEl.textContent += (chatEl.textContent ? "\n" : "") + prefix + (text || "");
    chatEl.scrollTop = chatEl.scrollHeight;
    return;
  }

  const row = document.createElement("div");
  row.className =
    "bubbleRow " +
    (kind === "user" ? "right" : kind === "bot" ? "left" : "center");

  const wrap = document.createElement("div");
  wrap.className = `bubbleWrap ${kind}`;

  if (kind !== "sys") {
    const nameEl = document.createElement("div");
    nameEl.className = `bubbleName ${kind}`;
    nameEl.textContent = name || (kind === "user" ? "用户" : "主播");
    wrap.appendChild(nameEl);
  }

  const bubble = document.createElement("div");
  bubble.className = `bubble ${kind}`;

  const content = document.createElement("div");
  content.className = "bubbleContent";
  content.textContent = text || "";
  bubble.appendChild(content);

  wrap.appendChild(bubble);
  row.appendChild(wrap);

  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function startBotBubble(name = "主播") {
  appendBubble("bot", "", name);
  // 找到刚刚插入的最后一个 bot bubbleContent
  const nodes = chatEl.querySelectorAll(".bubble.bot .bubbleContent");
  currentBotBubbleTextEl = nodes[nodes.length - 1] || null;
}

function appendBotDelta(delta) {
  if (!currentBotBubbleTextEl) startBotBubble("主播");
  currentBotBubbleTextEl.textContent += delta || "";
  chatEl.scrollTop = chatEl.scrollHeight;
}

// =========================
// Segment rendering (if your UI uses segListEl)
// =========================
function statusToCssClass(status) {
  if (status === "done") return "done";
  if (status === "playing") return "playing";
  if (status === "tts") return "synth";
  if (status === "ready") return "pending";
  if (status === "err") return "error";
  return "pending";
}

function statusToBadgeText(status) {
  if (status === "done") return "已播放";
  if (status === "playing") return "播报中";
  if (status === "tts") return "合成中";
  if (status === "ready") return "待播放";
  if (status === "err") return "TTS失败";
  return "待播放";
}

function renderSegments() {
  if (!segListEl) return;
  segListEl.innerHTML = "";

  const keys = Array.from(segMap.keys()).sort((a, b) => a - b);
  for (const seq of keys) {
    const seg = segMap.get(seq) || {};

    const cls = statusToCssClass(seg.status);
    const label = statusToBadgeText(seg.status) || seg.statusLabel || "";
    const text = seg.text || "";

    const li = document.createElement("li");
    li.className = `segItem ${cls}`;
    li.dataset.seq = String(seq);

    const badge = document.createElement("div");
    badge.className = `segBadge badge-${cls}`; // 兼容你 app.css 末尾的 badge-* 着色
    badge.textContent = label;

    const main = document.createElement("div");
    main.className = "segMain";
    main.style.minWidth = "0"; // 防止长文本撑爆布局

    const top = document.createElement("div");
    top.className = "segTop";
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    top.style.alignItems = "center";

    const seqEl = document.createElement("div");
    seqEl.className = "segSeq";
    seqEl.textContent = `#${seq}`;
    seqEl.style.fontWeight = "800";
    seqEl.style.fontSize = "13px";

    top.appendChild(seqEl);

    const textEl = document.createElement("div");
    textEl.className = "segText";
    textEl.textContent = text;
    textEl.style.marginTop = "4px";
    textEl.style.whiteSpace = "pre-wrap";
    textEl.style.wordBreak = "break-word";

    main.appendChild(top);
    main.appendChild(textEl);

    li.appendChild(badge);
    li.appendChild(main);

    segListEl.appendChild(li);
  }
}

// =========================
// Core: play next seq audio in order
// =========================
async function tryPlayNext() {
  if (playing) return;

  const url = audioReady.get(nextSeq);
  if (!url) {
    setQSize();
    return;
  }

  playing = true;

  const seg = segMap.get(nextSeq);
  if (nowTextEl) nowTextEl.textContent = seg?.text || "";

  // ✅ 触发动作：仅当 emo_trigger=true 且该 seq 未触发过
  try {
    const shouldActNow = (seg?.shouldAct === true) || (seg?.forceAct === true);
    if (shouldActNow && !motionTriggeredSeq.has(nextSeq)) {
      motionTriggeredSeq.add(nextSeq);

      const emo = (seg?.forceAct === true)
        ? (seg?.forceActEmotion || pickEmotionForInjection(seg))
        : (seg?.emotion || "正常");

      triggerMotionWithRetry(emo);
    }
  } catch (e) {
    console.warn("[Motion] trigger failed:", e);
  }

  try {
    player.src = url;
    player.load();
    await player.play();
    setStatus("SPEAKING", "ok");

    // speaking gate (optional)
    window.Live2DController?.setSpeaking?.(true);

    // update seg status
    if (seg) {
        seg.status = "playing";
        seg.statusLabel = "播报中";
        segMap.set(nextSeq, seg);
        renderSegments();
    }
  } catch (e) {
    console.warn("autoplay blocked:", e);
    setStatus("自动播放被拦截：点一下页面后再点播放", "warn");
    playing = false;
    return;
  }
  setQSize();
}

player.addEventListener("ended", () => {
  const seq = nextSeq;

  const seg = segMap.get(seq);
  if (seg) {
    seg.status = "done";
    seg.statusLabel = "已播放";
    segMap.set(seq, seg);

  }
  renderSegments();

  audioReady.delete(seq);
  nextSeq += 1;
  playing = false;

  window.Live2DController?.setSpeaking?.(false);

  tryPlayNext();
});

player.addEventListener("pause", () => window.Live2DController?.setSpeaking?.(false));
player.addEventListener("error", () => {
  const seq = nextSeq;
  const seg = segMap.get(seq);
  if (seg) {
    seg.status = "err";
    seg.statusLabel = "TTS失败";
    segMap.set(seq, seg);
  }
  renderSegments();

  audioReady.delete(seq);
  nextSeq += 1;
  playing = false;

  window.Live2DController?.setSpeaking?.(false);

  tryPlayNext();
});

// =========================
// Reset per send
// =========================
function resetAudioState() {
  try { player.pause(); } catch {}
  player.removeAttribute("src");
  player.load();

  playing = false;
  nextSeq = 1;

  audioReady.clear();
  segMap.clear();
  motionTriggeredSeq.clear();
  blockMap.clear(); // ✅ 每次发送前重置分组状态

  if (nowTextEl) nowTextEl.textContent = "";
  renderSegments();
  setQSize();
  setStatus("READY", "muted");
}

// =========================
// WS connect / send
// =========================
function wsUrl(apiBase) {
  const u = new URL(apiBase);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = "/ws";
  u.search = "";
  u.hash = "";
  return u.toString();
}

// ✅ 兼容不同 checkbox id，避免“前端一直发 tts=false”
function getTtsEnabled() {
  const ids = ["enableTts", "ttsEnable", "tts_enabled", "tts"];
  for (const id of ids) {
    const el = $(id);
    if (el && typeof el.checked === "boolean") return !!el.checked;
  }
  return true; // 默认开
}

function connect() {
  const apiBase = ($("apiBase")?.value || "http://127.0.0.1:8000")
    .trim()
    .replace(/\/+$/, "");
  const url = wsUrl(apiBase);

  ws = new WebSocket(url);

  ws.onopen = async () => {
    setStatus("CONNECTED", "ok");
    setButtons(true);
    appendBubble("sys", "WS 已连接");
    await loadRuntimeConfig(apiBase); // ✅ 等配置到位
  };

  ws.onclose = () => {
    setStatus("DISCONNECTED", "err");
    setButtons(false);
    ws = null;
    appendBubble("sys", "WS 已断开");
  };

  ws.onerror = () => {
    setStatus("WS ERROR", "err");
    appendBubble("sys", "WS ERROR");
  };

  ws.onmessage = (ev) => {
    let obj;
    try { obj = JSON.parse(ev.data); } catch { return; }

    const apiBase2 = ($("apiBase")?.value || "http://127.0.0.1:8000")
      .trim()
      .replace(/\/+$/, "");

    if (obj.type === "meta") return;

    if (obj.type === "delta") {
      appendBotDelta(obj.delta || "");
      return;
    }

    if (obj.type === "segment") {
      const seq = Number(obj.seq);
      if (!Number.isFinite(seq)) return;
      segMap.set(seq, {
        text: obj.text || "",
        emotion: obj.emotion || "正常",
        shouldAct: !!obj.emo_trigger,
        status: "tts",
        statusLabel: "合成中",
        audio_url: ""
      });
      renderSegments();
      return;
    }

    // ✅ NEW: emotion_update（后端单独推情绪更新）
    if (obj.type === "emotion_update") {
      const seq = Number(obj.seq);
      if (!Number.isFinite(seq)) return;

      const existing = segMap.get(seq) || {
        text: obj.text || "",
        emotion: "正常",
        shouldAct: false,
        status: audioReady.has(seq) ? "ready" : "tts",
        statusLabel: audioReady.has(seq) ? "待播放" : "合成中",
        audio_url: audioReady.get(seq) || ""
      };

      if (obj.text) existing.text = obj.text;
      existing.emotion = (obj.emotion || existing.emotion || "正常");
      existing.shouldAct = !!obj.emo_trigger;

      // 如果这段已经 ready 但还没播放，保持 ready
      if (audioReady.has(seq) && existing.status !== "playing" && existing.status !== "done") {
        existing.status = "ready";
        existing.statusLabel = "待播放";
      }

      segMap.set(seq, existing);
      renderSegments();

      // 如果 emotion_update 来得很晚，而此时正在播放该 seq，可立即补触发一次
      if (existing.shouldAct && seq === nextSeq && playing && !motionTriggeredSeq.has(seq)) {
        motionTriggeredSeq.add(seq);
        triggerMotionWithRetry(existing.emotion);
      }
      return;
    }

    if (obj.type === "audio") {
      const seq = Number(obj.seq);
      const au = obj.audio_url || "";
      if (!Number.isFinite(seq) || !au) return;

      const absUrl = au.startsWith("http") ? au : apiBase2 + au;
      audioReady.set(seq, absUrl);

      const seg = segMap.get(seq) || {
        text: obj.text || "",
        emotion: obj.emotion || "正常",
        shouldAct: !!obj.emo_trigger,
        status: "ready",
        statusLabel: "待播放",
        audio_url: absUrl
      };

      // audio 事件可能会带 emotion 覆盖
      if (obj.emotion) seg.emotion = obj.emotion;
      if (obj.emo_trigger !== undefined) seg.shouldAct = !!obj.emo_trigger;

      seg.audio_url = absUrl;
      seg.status = "ready";
      seg.statusLabel = "待播放";
      segMap.set(seq, seg);

      try {
      const blockId = getBlockIdBySeq(seq);
      const blk = ensureBlock(blockId);

      // 记录本组 seq
      if (!blk.seqs.includes(seq)) blk.seqs.push(seq);

      // 只要组内出现过 shouldAct=true，就认为本组已有动作
      if (seg.shouldAct === true) blk.hasTrigger = true;

      // 凑齐4条且本组没有动作：随机挑一条“未播放”的 seq 插入动作
      if (blk.seqs.length === AUDIO_BLOCK_SIZE && !blk.hasTrigger && blk.pickSeq == null) {
        const candidates = blk.seqs.filter(s => s >= nextSeq); // 避免挑到已播放的
        const pool = (candidates.length > 0) ? candidates : blk.seqs;
        const pick = pool[Math.floor(Math.random() * pool.length)];
        blk.pickSeq = pick;

        const pickedSeg = segMap.get(pick);
        if (pickedSeg && pickedSeg.status !== "done") {
        pickedSeg.forceAct = true; // ✅ 前端插入动作标记
        pickedSeg.forceActEmotion = "正常";
        segMap.set(pick, pickedSeg);
      }
    }
  } catch (e) {
    console.warn("[BlockMotion] failed:", e);
  }

      renderSegments();
      setQSize();
      tryPlayNext();
      return;
    }

    // 可选：后端可能会发 audio_skip（被标点/空文本过滤）
    if (obj.type === "audio_skip") {
      const seq = Number(obj.seq);
      if (!Number.isFinite(seq)) return;

      const seg = segMap.get(seq) || { text: obj.text || "" };
      seg.status = "done";
      seg.statusLabel = "跳过";
      if (obj.emotion) seg.emotion = obj.emotion;
      if (obj.emo_trigger !== undefined) seg.shouldAct = !!obj.emo_trigger;
      segMap.set(seq, seg);

      renderSegments();
      return;
    }

    if (obj.type === "audio_error") {
      const seq = Number(obj.seq);
      if (Number.isFinite(seq)) {
        const s = segMap.get(seq) || { text: obj.text || "" };
        s.status = "err";
        s.statusLabel = "TTS失败";
        segMap.set(seq, s);
        renderSegments();
      }
      setStatus("TTS ERROR", "warn");
      return;
    }

    if (obj.type === "server_error") {
      setStatus("SERVER ERROR", "err");
      appendBubble("sys", "SERVER ERROR: " + (obj.error || ""));
      return;
    }

    if (obj.type === "done") {
      setStatus("DONE（音频可能仍在播放）", "ok");
      return;
    }

    console.log("WS msg:", obj);
  };
}

function disconnect() {
  if (ws) ws.close();
  ws = null;
  setButtons(false);
}

async function send() {
  if (!ws || ws.readyState !== 1) return;

  resetAudioState();

  const user_id = ($("userId")?.value || "default_room").trim();
  const sender_name = ($("nickname")?.value || "路人甲").trim();
  const message = ($("danmaku")?.value || "").trim();
  const tts = getTtsEnabled();

  if (!message) return;

  appendBubble("user", message, sender_name);
  startBotBubble("主播");
  setStatus("THINKING / BUFFERING", "warn");

  const apiBase = ($("apiBase")?.value || "http://127.0.0.1:8000")
    .trim()
    .replace(/\/+$/, "");

  // ✅ 确保 emotionMotionMap 已加载
  if (!emotionMotionMap) {
    await loadRuntimeConfig(apiBase);
  }

  ws.send(JSON.stringify({ user_id, sender_name, message, tts }));

  if ($("danmaku")) $("danmaku").value = "";
}

// =========================
// Bind UI events
// =========================
if (btnConnect)
  btnConnect.addEventListener("click", () => {
    if (ws) return;
    connect();
  });
if (btnStop) btnStop.addEventListener("click", () => disconnect());

if (btnSend) btnSend.addEventListener("click", () => send());
if ($("danmaku")) {
  $("danmaku").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      send();
    }
  });
}

if (btnPause)
  btnPause.addEventListener("click", () => {
    try { player.pause(); } catch {}
    setStatus("PAUSED", "warn");
  });

if (btnResume)
  btnResume.addEventListener("click", async () => {
    try {
      await player.play();
      setStatus("SPEAKING", "ok");
      window.Live2DController?.setSpeaking?.(true);
    } catch {
      setStatus("RESUME BLOCKED（点 ▶️）", "warn");
    }
  });

if (btnSkip)
  btnSkip.addEventListener("click", () => {
    const seq = nextSeq;

    const seg = segMap.get(seq);
    if (seg) {
      seg.status = "done";
      seg.statusLabel = "跳过";
      segMap.set(seq, seg);
    }
    renderSegments();

    audioReady.delete(seq);
    nextSeq += 1;
    playing = false;
    tryPlayNext();
  });

// =========================
// Init
// =========================
setButtons(false);
setStatus("DISCONNECTED", "muted");
renderSegments();
setQSize();
if (nowTextEl) nowTextEl.textContent = "";
