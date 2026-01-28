// live2d.js (Suppress idle mouth while keeping idle animations)
// - Load Live2D model into #live2dCanvas
// - Drag + Wheel zoom
// - Mouth control: window.Live2DController.setMouth(v 0.1)
// - Speaking gate: window.Live2DController.setSpeaking(true/false)
// - Suppress idle mouth: window.Live2DController.setSuppressIdleMouth(true/false)
//   Default: ON -> when NOT speaking, force mouth closed (and optionally mouth form = 0)

(() => {
  const $ = (id) => document.getElementById(id);

  let app = null;
  let model = null;
  let removeWheel = null;

  // Mouth params
  let mouthOpenParamIdRaw = null;
  let mouthOpenParamIdKey = null;
  let mouthOpenParamIndex = -1;

  let mouthFormParamIdRaw = null; // optional
  let mouthFormParamIndex = -1;

  // Controls
  let mouthValue = 0;                 // 0.1
  let suppressIdleMouth = true;       // default ON
  let isSpeaking = false;             // controlled by app.js via setSpeaking()

  // ===== 新增：motions 元数据缓存 =====
  let motionsMeta = null; // { groupName: [{File: "."}] }

  async function fetchModel3Json(modelUrl) {
    const r = await fetch(modelUrl, { cache: "no-store" });
    if (!r.ok) throw new Error(`fetch model3 failed: ${r.status}`);
    return await r.json();
  }

  function setStatus(msg) {
    const el = $("live2dStatus");
    if (el) el.textContent = msg;
  }

  function assertDeps() {
    if (!window.PIXI) throw new Error("PIXI not found: vendor/pixi.min.js");
    if (!window.Live2DCubismCore) throw new Error("Live2DCubismCore not found: vendor/live2dcubismcore.min.js");
    if (!window.PIXI?.live2d?.Live2DModel) throw new Error("Live2DModel not found: vendor/pixi-live2d-display.cubism4.min.js");
  }

  function clamp01(v) {
    return Math.max(0, Math.min(1, v));
  }

  function toKey(id) {
    if (typeof id === "string") return id;
    if (id && typeof id === "object") {
      if (typeof id.getString === "function") return id.getString();
      if (typeof id.toString === "function") return id.toString();
    }
    return String(id);
  }

  function coreModelOf() {
    return model?.internalModel?.coreModel || null;
  }

  // ---------------------------
  // Param discovery
  // ---------------------------
  function getAllParams(m) {
    const cm = m?.internalModel?.coreModel;
    if (!cm) return [];

    // A) getParameterCount + getParameterId(i)
    try {
      if (typeof cm.getParameterCount === "function" && typeof cm.getParameterId === "function") {
        const n = cm.getParameterCount();
        const out = [];
        for (let i = 0; i < n; i++) {
          const raw = cm.getParameterId(i);
          out.push({ raw, key: toKey(raw), index: i });
        }
        if (out.length) return out;
      }
    } catch (_) {}

    // B) internal arrays
    try {
      const ids = cm?._model?.parameters?.ids;
      if (Array.isArray(ids) && ids.length) {
        return ids.map((raw, i) => ({ raw, key: toKey(raw), index: i }));
      }
    } catch (_) {}

    return [];
  }

  function pickMouthOpenParam(m) {
    const list = getAllParams(m);
    const hit =
      list.find(x => /parammouthopeny/i.test(x.key))
      || list.find(x => /mouthopen/i.test(x.key))
      || list.find(x => /mouth/i.test(x.key) && /open/i.test(x.key));
    return hit || null;
  }

  function pickMouthFormParam(m) {
    const list = getAllParams(m);
    const hit =
      list.find(x => /parammouthform/i.test(x.key))
      || list.find(x => /mouthform/i.test(x.key))
      || list.find(x => /mouth/i.test(x.key) && /form/i.test(x.key));
    return hit || null;
  }

  // ---------------------------
  // Param write helpers
  // ---------------------------
  function setParamValueByIndex(i, v) {
    const cm = coreModelOf();
    if (!cm) return;

    if (typeof cm.setParameterValueByIndex === "function") {
      try { cm.setParameterValueByIndex(i, v); return; } catch (_) {}
    }

    // last resort (many builds work)
    try {
      const values = cm?._model?.parameters?.values;
      if (values && i >= 0 && i < values.length) values[i] = v;
    } catch (_) {}
  }

  function setParamValueById(idRaw, v) {
    const cm = coreModelOf();
    if (!cm) return false;

    if (idRaw && typeof cm.setParameterValueById === "function") {
      try { cm.setParameterValueById(idRaw, v); return true; } catch (_) {}
    }
    return false;
  }

  function setMouthOpenInternal(v) {
    const x = clamp01(v);
    const cm = coreModelOf();
    if (!cm) return;

    // Prefer byId
    if (mouthOpenParamIdRaw && typeof cm.setParameterValueById === "function") {
      try { cm.setParameterValueById(mouthOpenParamIdRaw, x); return; } catch (_) {}
    }
    // Fallback byIndex
    if (mouthOpenParamIndex >= 0) setParamValueByIndex(mouthOpenParamIndex, x);
  }

  function setMouthFormInternal(v) {
    const cm = coreModelOf();
    if (!cm) return;

    const x = Math.max(-1, Math.min(1, v));

    // Prefer byId
    if (mouthFormParamIdRaw && typeof cm.setParameterValueById === "function") {
      try { cm.setParameterValueById(mouthFormParamIdRaw, x); return; } catch (_) {}
    }
    // Fallback byIndex
    if (mouthFormParamIndex >= 0) setParamValueByIndex(mouthFormParamIndex, x);
  }

  // ---------------------------
  // Idle mouth suppression
  // ---------------------------
  function suppressMouthIfIdle() {
    if (!model) return;
    if (!suppressIdleMouth) return;
    if (isSpeaking) return;

    setMouthOpenInternal(0);

    if (mouthFormParamIdRaw || mouthFormParamIndex >= 0) {
      setMouthFormInternal(0);
    }
  }

  function applyMouthPolicyNow() {
    if (!model) return;
    if (isSpeaking) {
      setMouthOpenInternal(mouthValue);
    } else {
      suppressMouthIfIdle();
    }
  }

  // ---------------------------
  // Hook _render to guarantee "last write"
  // ---------------------------
  let originalRender = null;
  let renderHooked = false;

  function hookRender() {
    if (!model || renderHooked) return;

    if (typeof model._render !== "function") {
      console.warn("[Live2D] model._render not found; using ticker fallback (may be less stable).");
      const pr = 100000;
      PIXI.Ticker.shared.add(() => applyMouthPolicyNow(), null, pr);
      renderHooked = true;
      return;
    }

    originalRender = model._render.bind(model);
    model._render = function (renderer) {
      const r = originalRender(renderer);
      applyMouthPolicyNow();
      return r;
    };

    renderHooked = true;
    console.log("[Live2D] model._render hooked for idle mouth suppression");
  }

  function unhookRender() {
    if (!model || !renderHooked) return;
    try {
      if (originalRender) model._render = originalRender;
    } catch (_) {}
    originalRender = null;
    renderHooked = false;
  }

  // ---------------------------
  // Drag + Zoom
  // ---------------------------
  function fitModelCenter(m, a) {
    const w = a.renderer.width;
    const h = a.renderer.height;

    const bounds = m.getLocalBounds();
    const scaleX = w / bounds.width;
    const scaleY = h / bounds.height;
    const scale = Math.min(scaleX, scaleY) * 0.9;

    m.scale.set(scale, scale);

    const b2 = m.getLocalBounds();
    m.x = (w - b2.width * scale) / 2 - b2.x * scale;
    m.y = (h - b2.height * scale) / 2 - b2.y * scale;
  }

  function enableDrag(m) {
    m.interactive = true;
    m.cursor = "grab";

    let dragging = false;
    let start = { x: 0, y: 0 };
    let origin = { x: 0, y: 0 };

    m.on("pointerdown", (e) => {
      dragging = true;
      m.cursor = "grabbing";
      const p = e.data.getLocalPosition(m.parent);
      start = { x: p.x, y: p.y };
      origin = { x: m.x, y: m.y };
    });

    const end = () => {
      dragging = false;
      m.cursor = "grab";
    };

    m.on("pointerup", end);
    m.on("pointerupoutside", end);

    m.on("pointermove", (e) => {
      if (!dragging) return;
      const p = e.data.getLocalPosition(m.parent);
      m.x = origin.x + (p.x - start.x);
      m.y = origin.y + (p.y - start.y);
    });
  }

  function enableWheelZoom(a, canvas, m, opts = {}) {
    const minScale = opts.minScale ?? 0.12;
    const maxScale = opts.maxScale ?? 6.0;
    const zoomSpeed = opts.zoomSpeed ?? 0.0016;

    const onWheel = (e) => {
      e.preventDefault();

      const factor = Math.exp(-e.deltaY * zoomSpeed);
      const oldScale = m.scale.x;

      let newScale = oldScale * factor;
      newScale = Math.max(minScale, Math.min(maxScale, newScale));

      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const before = a.renderer.plugins.interaction.mapPositionToPoint(
        new PIXI.Point(), mx, my
      );

      m.scale.set(newScale, newScale);

      const after = a.renderer.plugins.interaction.mapPositionToPoint(
        new PIXI.Point(), mx, my
      );

      m.x += after.x - before.x;
      m.y += after.y - before.y;
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }

  function resizeRendererToCanvas(canvas) {
    if (!app) return;
    const w = canvas.clientWidth || canvas.width || 480;
    const h = canvas.clientHeight || canvas.height || 720;
    app.renderer.resize(w, h);
  }

  // ---------------------------
  // Destroy / Load
  // ---------------------------
  function destroy() {
    try { unhookRender(); } catch (_) {}

    try {
      if (removeWheel) {
        removeWheel();
        removeWheel = null;
      }
      if (model) {
        model.destroy?.({ children: true, texture: true, baseTexture: true });
        model = null;
      }
      if (app) {
        app.destroy?.(true, { children: true, texture: true, baseTexture: true });
        app = null;
      }
    } catch (_) {}

    mouthOpenParamIdRaw = null;
    mouthOpenParamIdKey = null;
    mouthOpenParamIndex = -1;
    mouthFormParamIdRaw = null;
    mouthFormParamIndex = -1;

    mouthValue = 0;
    suppressIdleMouth = true;
    isSpeaking = false;
    motionsMeta = null;
  }

  async function load(modelUrl) {
    assertDeps();

    const canvas = $("live2dCanvas");
    if (!canvas) throw new Error("live2dCanvas not found");

    setStatus("加载中…");
    destroy();

    const w = canvas.clientWidth || 480;
    const h = canvas.clientHeight || canvas.height || 720;

    app = new PIXI.Application({
      view: canvas,
      backgroundAlpha: 0,
      antialias: true,
      autoDensity: true,
      resolution: window.devicePixelRatio || 1,
      width: w,
      height: h,
    });

    model = await PIXI.live2d.Live2DModel.from(modelUrl, {
      autoInteract: false,
      autoUpdate: true,
    });

    // ===== 新增：读取 model3.json 的 Motions =====
    try {
      const m3 = await fetchModel3Json(modelUrl);
      motionsMeta = m3?.FileReferences?.Motions || null;
      console.log("[Live2D] Motions groups:", motionsMeta ? Object.keys(motionsMeta) : []);
    } catch (e) {
      console.warn("[Live2D] read motionsMeta failed:", e);
      motionsMeta = null;
    }

    app.stage.addChild(model);
    fitModelCenter(model, app);

    enableDrag(model);
    removeWheel = enableWheelZoom(app, canvas, model);

    app.ticker.start();
    canvas.style.touchAction = "none";

    await new Promise((r) => requestAnimationFrame(r));

    const params = getAllParams(model);
    console.log("[Live2D] Param count:", params.length);
    console.log("[Live2D] Parameter IDs:", params.map(x => x.key));

    const openHit = pickMouthOpenParam(model);
    if (!openHit) {
      console.warn("[Live2D] MouthOpen param not found; mouth control disabled.");
      setStatus("已显示（未找到嘴参数）");
    } else {
      mouthOpenParamIdRaw = openHit.raw;
      mouthOpenParamIdKey = openHit.key;
      mouthOpenParamIndex = Number.isFinite(openHit.index) ? openHit.index : -1;
      console.log("[Live2D] Mouth Open Param:", mouthOpenParamIdKey, "index=", mouthOpenParamIndex);
      setStatus("已显示（支持口型）");
    }

    const formHit = pickMouthFormParam(model);
    if (formHit) {
      mouthFormParamIdRaw = formHit.raw;
      mouthFormParamIndex = Number.isFinite(formHit.index) ? formHit.index : -1;
      console.log("[Live2D] Mouth Form Param:", formHit.key, "index=", mouthFormParamIndex);
    }

    hookRender();
    applyMouthPolicyNow();
  }

  // ---------------------------
  // Public API
  // ---------------------------
  window.Live2DController = {
    listMotions() {
      const out = {};
      const m = motionsMeta || {};
      for (const k of Object.keys(m)) out[k] = Array.isArray(m[k]) ? m[k].length : 0;
      return out;
    },

    playMotion(group, index = 0, priority = 3) {
      if (!model) return false;
      if (!group) return false;

      // If we have motionsMeta and group doesn't exist, fail fast
      if (motionsMeta && !motionsMeta[group]) return false;

      const cnt = motionsMeta?.[group]?.length ?? 0;
      let i = Number(index) || 0;
      if (cnt > 0) {
        if (i < 0 || i >= cnt) i = Math.floor(Math.random() * cnt);
      }

      try {
        model.motion(group, i, priority);
        return true;
      } catch (e) {
        console.warn("[Live2D] playMotion failed:", e);
        return false;
      }
    },

    setMouth(v) {
      mouthValue = clamp01(v);
      if (isSpeaking) {
        setMouthOpenInternal(mouthValue);
      }
    },

    setSpeaking(on) {
      isSpeaking = !!on;
      applyMouthPolicyNow();
    },

    setSuppressIdleMouth(on) {
      suppressIdleMouth = !!on;
      applyMouthPolicyNow();
    },

    // ✅ isReady 不再依赖嘴参数（否则会导致“能动但 isReady=false”）
    isReady() {
      return !!model;
    },

    hasMouth() {
      return !!model && (!!mouthOpenParamIdRaw || mouthOpenParamIndex >= 0);
    },

    getMouthParamId() {
      return mouthOpenParamIdKey || null;
    }
  };

  // ---------------------------
  // Bind UI
  // ---------------------------
  function bindUI() {
    const btn = $("btnLoadLive2D");
    const input = $("live2dModelUrl");
    const canvas = $("live2dCanvas");

    if (!btn || !input || !canvas) return;

    btn.addEventListener("click", async () => {
      const url = (input.value || "").trim();
      if (!url) {
        setStatus("请输入模型 URL（/Live2D/...model3.json）");
        return;
      }
      try {
        await load(url);
      } catch (e) {
        console.error("[Live2D] load failed:", e);
        setStatus("加载失败：看 Console");
      }
    });

    window.addEventListener("resize", () => {
      if (!app) return;
      resizeRendererToCanvas(canvas);
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    bindUI();
  });
})();
