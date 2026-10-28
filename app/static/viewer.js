// Browser 3D replay of a simulation attempt (three.js). The server sends body poses per frame
// (traj.bin) plus geom descriptions (viz.json); robot meshes are shared and cached (/viz/meshes.*).
// MuJoCo is z-up, so the camera uses z as "up" and world coordinates are used as-is.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const REACH = { x: [0.3, 0.7], y: [-0.4, 0.4] }; // keep in sync with schemas.REACH_*
let meshCache = null;

async function loadMeshes() {
  if (!meshCache) {
    meshCache = Promise.all([fetch("/viz/meshes.json").then((r) => r.json()),
                             fetch("/viz/meshes.bin").then((r) => r.arrayBuffer())])
      .then(([manifest, buf]) => {
        const out = {};
        for (const [name, m] of Object.entries(manifest)) {
          const q = new Int16Array(buf, m.offset, m.count * 3);
          const n = new Int8Array(buf, m.noffset, m.count * 3);
          const pos = new Float32Array(m.count * 3), nrm = new Float32Array(m.count * 3);
          for (let i = 0; i < m.count * 3; i++) {
            const k = i % 3;
            pos[i] = m.min[k] + ((q[i] + 32768) / 65535) * m.span[k];
            nrm[i] = n[i] / 127;
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
          g.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
          out[name] = g;
        }
        return out;
      });
  }
  return meshCache;
}

const quat = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]); // MuJoCo (w,x,y,z) -> three

function primitive(g) {
  const [a, b, c] = g.size;
  if (g.type === "box") return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
  if (g.type === "sphere") return new THREE.SphereGeometry(a, 40, 24);
  const cyl = new THREE.CylinderGeometry(a, a, 2 * b, 48); // three's cylinder axis is y, MuJoCo's is z
  cyl.rotateX(Math.PI / 2);
  return cyl;
}

function label(text) {
  const d = document.createElement("div");
  d.className = "v-caption";
  d.textContent = text;
  return d;
}


// ---------- realistic skins ----------
// Physics uses simple shapes; in "realistic" style the browser dresses them by what the object is
// (matched on its name), with procedural canvas textures. No image files, no brand marks.
const texCache = new Map();
function canvasTex(key, w, h, draw, repeat = null) {
  if (texCache.has(key)) return texCache.get(key);
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); }
  texCache.set(key, t);
  return t;
}
const css = (rgb, k = 1) => `rgb(${rgb.map((v) => Math.round(Math.min(1, v * k) * 255)).join(",")})`;
let seed = 1;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

const wood = (rgb, painted) => canvasTex(`wood${rgb}${painted}`, 256, 256, (g, w, h) => {
  seed = 7;
  g.fillStyle = css(rgb); g.fillRect(0, 0, w, h);
  for (let i = 0; i < 46; i++) {
    const y = rnd() * h, amp = 2 + rnd() * 6;
    g.strokeStyle = `rgba(${painted ? "255,255,255" : "60,30,10"},${painted ? 0.05 + rnd() * 0.06 : 0.08 + rnd() * 0.12})`;
    g.lineWidth = 0.6 + rnd() * 2;
    g.beginPath();
    for (let x = 0; x <= w; x += 8) g.lineTo(x, y + Math.sin(x / (18 + i)) * amp);
    g.stroke();
  }
});
const rubikFace = (color) => canvasTex(`rubik${color}`, 192, 192, (g) => {
  g.fillStyle = "#111"; g.fillRect(0, 0, 192, 192);
  g.fillStyle = color;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    g.beginPath(); g.roundRect(8 + i * 61, 8 + j * 61, 54, 54, 8); g.fill();
  }
});
const canLabel = (rgb) => canvasTex(`can${rgb}`, 512, 256, (g, w, h) => {
  g.fillStyle = css(rgb, 0.9); g.fillRect(0, 0, w, h);
  g.fillStyle = "rgba(255,255,255,0.9)";
  g.beginPath(); g.moveTo(0, h * 0.62);
  for (let x = 0; x <= w; x += 8) g.lineTo(x, h * 0.58 + Math.sin(x / 40) * 14);
  g.lineTo(w, h * 0.7); for (let x = w; x >= 0; x -= 8) g.lineTo(x, h * 0.68 + Math.sin(x / 40) * 14);
  g.fill();
  g.fillStyle = "rgba(255,255,255,0.95)"; g.font = "bold 64px Archivo, Arial, sans-serif"; g.textAlign = "center";
  g.fillText("SODA", w * 0.3, h * 0.45); g.fillText("SODA", w * 0.8, h * 0.45);
  const metal = g.createLinearGradient(0, 0, 0, h);
  metal.addColorStop(0, "#c9ced3"); metal.addColorStop(0.06, "#8d949b"); metal.addColorStop(0.1, "rgba(0,0,0,0)");
  metal.addColorStop(0.9, "rgba(0,0,0,0)"); metal.addColorStop(0.94, "#8d949b"); metal.addColorStop(1, "#c9ced3");
  g.fillStyle = metal; g.fillRect(0, 0, w, h);
});
const canTop = () => canvasTex("cantop", 128, 128, (g) => {
  const r = g.createRadialGradient(64, 64, 10, 64, 64, 64);
  r.addColorStop(0, "#d9dde1"); r.addColorStop(0.75, "#a8afb6"); r.addColorStop(0.85, "#e9ecef"); r.addColorStop(1, "#80878e");
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  g.fillStyle = "#6f767d"; g.beginPath(); g.ellipse(64, 46, 16, 9, 0, 0, Math.PI * 2); g.fill();
});
const tennis = () => canvasTex("tennis", 512, 256, (g, w, h) => {
  seed = 3;
  g.fillStyle = "#d7e84a"; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 2500; i++) { g.fillStyle = `rgba(${rnd() < 0.5 ? "255,255,200" : "120,140,20"},0.18)`; g.fillRect(rnd() * w, rnd() * h, 2, 2); }
  g.strokeStyle = "rgba(255,255,255,0.95)"; g.lineWidth = 7;
  g.beginPath(); for (let x = 0; x <= w; x += 4) g.lineTo(x, h / 2 + Math.sin((x / w) * Math.PI * 4) * h * 0.28); g.stroke();
});
const speckled = (rgb, spot, n) => canvasTex(`speck${rgb}${spot}`, 256, 128, (g, w, h) => {
  seed = 11;
  g.fillStyle = css(rgb); g.fillRect(0, 0, w, h);
  for (let i = 0; i < n; i++) { g.fillStyle = spot; g.globalAlpha = 0.15 + rnd() * 0.25; g.beginPath(); g.arc(rnd() * w, rnd() * h, 0.6 + rnd() * 1.4, 0, 7); g.fill(); }
  g.globalAlpha = 1;
});
const wicker = (rgb) => canvasTex(`wicker${rgb}`, 128, 128, (g) => {
  g.fillStyle = css(rgb, 0.7); g.fillRect(0, 0, 128, 128);
  for (let y = 0; y < 128; y += 16) for (let x = 0; x < 128; x += 16) {
    const odd = ((x + y) / 16) % 2;
    g.fillStyle = css(rgb, odd ? 1.15 : 0.95);
    g.beginPath(); g.roundRect(x + (odd ? 1 : 3), y + (odd ? 3 : 1), odd ? 14 : 10, odd ? 10 : 14, 4); g.fill();
  }
}, [1.2, 1.2]);

const RUBIK = ["#c41e3a", "#ff8c1a", "#ffffff", "#ffd500", "#009e60", "#0051ba"];
function std(opts) { return new THREE.MeshStandardMaterial(opts); }

// Returns a material (or material array) for an object geom, plus optional extra child meshes.
function realistic(g) {
  const n = (g.object || "").toLowerCase(), rgb = g.rgba.slice(0, 3);
  const base = g.object.split("/")[0];
  if (/rubik/.test(n) && g.type === "box") return { mat: RUBIK.map((c) => std({ map: rubikFace(c), roughness: 0.35 })) };
  if (/(can|soda|cola|beer)/.test(n) && g.type === "cylinder")
    return { mat: [std({ map: canLabel(rgb), roughness: 0.3, metalness: 0.55 }), std({ map: canTop(), roughness: 0.25, metalness: 0.85 }), std({ map: canTop(), roughness: 0.25, metalness: 0.85 })] };
  if (/tennis/.test(n)) return { mat: std({ map: tennis(), roughness: 1 }) };
  if (/apple/.test(n) && g.type === "sphere") {
    const r = g.size[0];
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.05, r * 0.07, r * 0.45, 8), std({ color: 0x5a3b1e, roughness: 0.9 }));
    stem.rotation.x = Math.PI / 2; stem.position.z = r * 1.05;
    const leaf = new THREE.Mesh(new THREE.SphereGeometry(r * 0.22, 12, 8), std({ color: 0x3f8f3a, roughness: 0.6 }));
    leaf.scale.set(1, 0.45, 0.12); leaf.position.set(r * 0.18, 0, r * 1.12); leaf.rotation.y = -0.6;
    return { mat: std({ map: speckled(rgb, "#ffe9a0", 500), roughness: 0.4 }), scale: [1, 1, 0.9], extras: [stem, leaf] };
  }
  if (/(lime|orange|lemon|citrus|mandarin)/.test(n)) return { mat: std({ map: speckled(rgb, "#ffffff", 900), roughness: 0.7 }) };
  if (/basket/.test(n)) return { mat: std({ map: wicker(rgb), roughness: 0.9 }) };
  if (/(drawer|cabinet|organizer)/.test(n) && g.type === "box") {
    const small = Math.max(...g.size) <= 0.06 && Math.min(...g.size) <= 0.009; // handle bar and its posts
    return { mat: small ? std({ color: 0xb8bcc0, roughness: 0.25, metalness: 0.9 })
                        : std({ map: wood(rgb, Math.min(...rgb) > 0.7), roughness: 0.7 }) };
  }
  if (/(block|cube|brick)/.test(n) && g.type === "box") return { mat: std({ map: wood(rgb, true), roughness: 0.55 }), bevel: true };
  if (/(bin|tray|box|container)/.test(base)) return { mat: std({ color: new THREE.Color(...rgb).convertSRGBToLinear(), roughness: 0.55, metalness: 0.0 }) };
  return null;
}

export class SimViewer {
  constructor(container) {
    this.el = container;
    this.el.classList.add("viewer");
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.el.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1b2630);
    this.scene.fog = new THREE.Fog(0x1b2630, 3.5, 7);
    this.camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.02, 20);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1.62, -1.02, 0.98); // close to the server-rendered video's view
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.36, 0, 0.26);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 3.5;
    // ?orbit=1: slow automatic orbit (for videos); ?clean=1 hides the mouse hint.
    const qs = new URLSearchParams(location.search);
    if (qs.get("theater") === "1") {  // full-window shots: frame the workspace tighter
      this.camera.position.set(1.22, -0.82, 0.72);
      this.controls.target.set(0.44, 0, 0.17);
    }
    this.controls.autoRotate = qs.get("orbit") === "1";
    this.controls.autoRotateSpeed = 0.25;
    this.clean = qs.get("clean") === "1";

    this.#stage();
    this.#ui();
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.playing = true; this.t = 0; this.speed = 1; this.last = performance.now(); this.style = "realistic"; this.followers = []; this.master = null; this.ghost = null;
    new ResizeObserver(() => this.#resize()).observe(this.el);
    this.#resize();
    this.renderer.setAnimationLoop(() => this.#tick());
  }

  #stage() {
    const s = this.scene;
    s.add(new THREE.HemisphereLight(0xdfe9f2, 0x2a2622, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(1.2, -1.0, 2.2);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.5, far: 5 });
    key.shadow.bias = -0.0004; key.shadow.normalBias = 0.01; key.shadow.radius = 4;
    s.add(key);
    const rim = new THREE.DirectionalLight(0x9fc4ff, 0.6);
    rim.position.set(-1.5, 1.2, 1.0);
    s.add(rim);

    // Lab bench: the robot is bolted to it at the origin; objects rest on its top (z = 0).
    const bench = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.3, 0.04),
      new THREE.MeshStandardMaterial({ color: 0xd9d4cb, roughness: 0.85 }));
    bench.position.set(0.35, 0, -0.02);
    bench.receiveShadow = true;
    s.add(bench);
    const edge = new THREE.Mesh(new THREE.BoxGeometry(1.52, 1.32, 0.008),
      new THREE.MeshStandardMaterial({ color: 0x5b6670, roughness: 0.6 }));
    edge.position.set(0.35, 0, -0.044);
    s.add(edge);
    // Reachable area, drawn on the bench.
    const w = REACH.x[1] - REACH.x[0], h = REACH.y[1] - REACH.y[0];
    const zone = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
      new THREE.MeshStandardMaterial({ color: 0x2c5d8f, transparent: true, opacity: 0.07, roughness: 1 }));
    zone.position.set(REACH.x[0] + w / 2, 0, 0.0006);
    zone.receiveShadow = true;
    s.add(zone);
    const outline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w, h)),
      new THREE.LineDashedMaterial({ color: 0x2c5d8f, dashSize: 0.02, gapSize: 0.015, transparent: true, opacity: 0.55 }));
    outline.position.copy(zone.position); outline.position.z = 0.001;
    outline.computeLineDistances();
    s.add(outline);
    const floor = new THREE.Mesh(new THREE.CircleGeometry(6, 64), new THREE.MeshStandardMaterial({ color: 0x141c23, roughness: 1 }));
    floor.position.z = -0.75;
    s.add(floor);
  }

  #ui() {
    const bar = document.createElement("div");
    bar.className = "v-bar";
    bar.innerHTML = `<button type="button" class="v-play" aria-label="Pause">❚❚</button>
      <div class="v-track"><input type="range" min="0" max="1000" value="0" aria-label="Timeline"><div class="v-marks"></div></div>
      <span class="v-time mono">0.0 s</span>
      <button type="button" class="v-speed" aria-label="Playback speed">1×</button>
      <button type="button" class="v-style" aria-label="Switch between realistic and plain objects">Realistic</button>
      <button type="button" class="v-edit" hidden title="Drag objects to new spots, then re-run the same plan">Move objects</button>
      <button type="button" class="v-full" aria-label="Full screen" title="Full screen">⛶</button>`;
    this.el.appendChild(bar);
    this.caption = label("");
    this.el.appendChild(this.caption);
    this.hint = label("drag to orbit · scroll to zoom");
    this.hint.classList.add("v-hint");
    this.el.appendChild(this.hint);
    this.playBtn = bar.querySelector(".v-play");
    this.range = bar.querySelector("input");
    this.timeEl = bar.querySelector(".v-time");
    this.marks = bar.querySelector(".v-marks");
    this.playBtn.onclick = () => this.setPlaying(!this.playing);
    bar.querySelector(".v-speed").onclick = (e) => {
      this.speed = this.speed === 1 ? 2 : this.speed === 2 ? 0.5 : 1;
      e.target.textContent = `${this.speed}×`;
    };
    bar.querySelector(".v-full").onclick = () => {
      const target = this.el.closest(".screen") || this.el;
      if (document.fullscreenElement) document.exitFullscreen();
      else target.requestFullscreen?.();
    };
    this.editBtn = bar.querySelector(".v-edit");
    this.editBtn.onclick = () => (this.editing ? this.stopEdit() : this.startEdit());
    this.editBar = document.createElement("div");
    this.editBar.className = "v-editbar";
    this.editBar.hidden = true;
    this.editBar.innerHTML = `<span>Drag objects on the bench. The plan is not changed, only where things start.</span>
      <button type="button" class="v-reset">Reset</button><button type="button" class="v-rerun">Re-run ▶</button>`;
    this.el.appendChild(this.editBar);
    this.editBar.querySelector(".v-reset").onclick = () => { this.moves = {}; this.#pose(0); this.#syncRerun(); };
    this.editBar.querySelector(".v-rerun").onclick = () => this.#rerun();
    this.#dragSetup();
    this.styleBtn = bar.querySelector(".v-style");
    this.styleBtn.onclick = () => this.setStyle(this.style === "realistic" ? "plain" : "realistic");
    this.range.oninput = () => { this.t = (this.range.value / 1000) * this.duration; this.setPlaying(false); };
    this.renderer.domElement.addEventListener("pointerdown", () => { this.hint.style.opacity = 0; }, { once: true });
    if (this.clean) this.hint.style.display = "none";
  }

  setPlaying(p) {
    this.playing = p;
    this.playBtn.textContent = p ? "❚❚" : "▶";
    this.playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
  }

  async load(url) {
    if (url === this.url) return;
    this.url = url;
    const base = url.slice(0, url.lastIndexOf("/") + 1);
    const [meta, buf, meshes] = await Promise.all([
      fetch(url).then((r) => r.json()), fetch(base + "traj.bin").then((r) => r.arrayBuffer()), loadMeshes()]);
    if (url !== this.url) return; // a newer load started meanwhile
    this.meta = meta;
    this.meshes = meshes;
    this.traj = new Float32Array(buf);
    this.nb = meta.bodies.length;
    this.duration = meta.times.length ? meta.times[meta.times.length - 1] : 0;
    this.root.clear();
    this.bodies = meta.bodies.map((name) => { const g = new THREE.Group(); g.userData.body = name; this.root.add(g); return g; });
    if (this.editing) this.stopEdit();
    this.#build();

    this.events = (meta.events || []).map((e) => {
      const m = /^t=([\d.]+)s (.*)$/.exec(e);
      return m ? { t: parseFloat(m[1]), text: m[2] } : null;
    }).filter(Boolean);
    this.marks.innerHTML = this.events.map((e) => {
      const kind = /grasped/.test(e.text) ? "g" : /released/.test(e.text) ? "r" : /blocked|stopped|nothing|tipped|fell|wide/.test(e.text) ? "x" : "n";
      return `<i class="m-${kind}" style="left:${(100 * e.t) / Math.max(this.duration, 0.01)}%" title="${e.text.replace(/"/g, "&quot;")}"></i>`;
    }).join("");
    // Goal checks as labels over the objects, shown on the final state.
    this.goalLayer?.remove();
    this.goalLayer = document.createElement("div");
    this.goalLayer.className = "v-goals";
    this.el.appendChild(this.goalLayer);
    const bodyOf = {};
    meta.geoms.forEach((g) => { if (g.object) bodyOf[g.object] = g.body; });
    this.goalTags = (meta.goals || []).filter((r) => bodyOf[r.goal.obj] !== undefined).map((r) => {
      const tag = document.createElement("div");
      tag.className = "v-goal " + (r.ok ? "ok" : "bad");
      const g = r.goal, off = /([\d.]+) cm off/.exec(r.detail);
      tag.textContent = `${r.ok ? "✓" : "✗"} ${g.obj} ${g.type.replace(/_/g, " ")}${g.target ? " " + g.target : ""}${!r.ok && off ? ` · ${off[1]} cm off` : ""}`;
      this.goalLayer.appendChild(tag);
      return { tag, body: bodyOf[g.obj] };
    });
        // ?t=5 opens the replay paused at 5 s (handy for screenshots and links to a moment).
    const t0 = parseFloat(new URLSearchParams(location.search).get("t"));
    this.t = Number.isFinite(t0) ? t0 : 0;
    this.setPlaying(!Number.isFinite(t0));
    this.#pose(this.#frameAt(this.t));
  }

  // Geometry and materials for the current run, in the current style ("realistic" or "plain").
  #build() {
    const meta = this.meta;
    for (const b of this.bodies) b.clear();
    const mats = new Map();
    for (const g of meta.geoms) {
      const geo = g.type === "mesh" ? this.meshes[g.mesh] : primitive(g);
      if (!geo) continue;
      const skin = this.style === "realistic" && g.object ? realistic(g) : null;
      let mat = skin?.mat;
      if (!mat) {
        const key = g.rgba.join(",") + (g.object ? "o" : "");
        if (!mats.has(key)) {
          mats.set(key, new THREE.MeshStandardMaterial({
            color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]).convertSRGBToLinear(),
            roughness: g.object ? 0.45 : 0.38, metalness: g.object ? 0.0 : 0.08,
            transparent: g.rgba[3] < 1, opacity: g.rgba[3] }));
        }
        mat = mats.get(key);
      }
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...g.pos);
      mesh.quaternion.copy(quat(g.quat));
      if (skin?.scale) mesh.scale.set(...skin.scale);
      for (const x of skin?.extras || []) { x.castShadow = true; mesh.add(x); }
      mesh.castShadow = true; mesh.receiveShadow = true;
      this.bodies[g.body].add(mesh);
    }
  }

  // ---------- "Try to break it": drag objects, re-run the same plan (physics only) ----------
  enableTry(onTry) { this.onTry = onTry; this.editBtn.hidden = !onTry; }

  startEdit() {
    if (!this.meta) return;
    this.editing = true; this.moves = {};
    this.setPlaying(false); this.t = 0; this.#pose(0);
    this.editBar.hidden = false; this.editBtn.textContent = "Cancel";
    this.goalLayer.style.display = "none";
    this.#syncRerun();
  }

  stopEdit() {
    this.editing = false; this.editBar.hidden = true; this.editBtn.textContent = "Move objects";
    this.goalLayer.style.display = "";
    this.renderer.domElement.style.cursor = "";
  }

  #syncRerun() {
    const n = Object.keys(this.moves || {}).length;
    const b = this.editBar.querySelector(".v-rerun");
    b.disabled = !n || this.busy;
    b.textContent = this.busy ? "Simulating…" : n ? `Re-run with ${n} moved ▶` : "Re-run ▶";
  }

  async #rerun() {
    if (!this.onTry || this.busy) return;
    this.busy = true; this.#syncRerun();
    try { await this.onTry({ ...this.moves }); this.stopEdit(); }
    catch (e) { this.editBar.querySelector("span").textContent = String(e.message || e); }
    finally { this.busy = false; this.#syncRerun(); }
  }

  #dragSetup() {
    const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const hit = new THREE.Vector3();
    const dom = this.renderer.domElement;
    const at = (ev) => { const r = dom.getBoundingClientRect(); ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ndc, this.camera); };
    const bodyName = (obj) => { while (obj && !obj.userData.body) obj = obj.parent; return obj ? obj.userData.body : null; };
    dom.addEventListener("pointerdown", (ev) => {
      if (!this.editing) return;
      at(ev);
      const groups = this.bodies.filter((g) => g.userData.body && g.userData.body.startsWith("obj/") && !g.userData.body.endsWith("/drawer"));
      const h = ray.intersectObjects(groups, true)[0];
      if (!h) return;
      const name = bodyName(h.object).slice(4);
      const g = this.bodies.find((b) => b.userData.body === "obj/" + name);
      plane.constant = -g.position.z;
      ray.ray.intersectPlane(plane, hit);
      this.drag = { name, g, child: this.bodies.find((b) => b.userData.body === `obj/${name}/drawer`),
                    off: g.position.clone().sub(hit) };
      this.controls.enabled = false;
      dom.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    });
    dom.addEventListener("pointermove", (ev) => {
      if (!this.editing) return;
      at(ev);
      if (!this.drag) {
        const groups = this.bodies.filter((g) => g.userData.body && g.userData.body.startsWith("obj/"));
        dom.style.cursor = ray.intersectObjects(groups, true).length ? "grab" : "";
        return;
      }
      if (!ray.ray.intersectPlane(plane, hit)) return;
      const { g, child, off } = this.drag;
      const nx = hit.x + off.x, ny = hit.y + off.y;
      if (child) { child.position.x += nx - g.position.x; child.position.y += ny - g.position.y; }
      g.position.x = nx; g.position.y = ny;
      dom.style.cursor = "grabbing";
    });
    dom.addEventListener("pointerup", () => {
      if (!this.drag) return;
      const { name, g } = this.drag;
      this.moves[name] = [Math.round(g.position.x * 1000) / 1000, Math.round(g.position.y * 1000) / 1000];
      this.drag = null; this.controls.enabled = true; this.#syncRerun();
    });
  }

  // Compare view: this viewer mirrors `master`'s clock and camera (master loops over the longer run).
  follow(master) {
    this.master = master;
    if (!master.followers.includes(this)) master.followers.push(this);
    this.el.querySelector(".v-bar").style.display = "none";
  }

  unfollow() {
    if (this.master) this.master.followers = this.master.followers.filter((f) => f !== this);
    this.master = null;
    this.el.querySelector(".v-bar").style.display = "";
  }

  // Ghost: another attempt's objects drawn translucent in this scene, on the same clock.
  async setGhost(url) {
    if (this.ghostRoot) { this.scene.remove(this.ghostRoot); this.ghostRoot = null; }
    this.ghost = null;
    if (!url) return;
    const base = url.slice(0, url.lastIndexOf("/") + 1);
    const [meta, buf] = await Promise.all([fetch(url).then((r) => r.json()), fetch(base + "traj.bin").then((r) => r.arrayBuffer())]);
    const root = new THREE.Group();
    const bodies = meta.bodies.map(() => { const g = new THREE.Group(); root.add(g); return g; });
    const mat = new THREE.MeshStandardMaterial({ color: 0xff6b4a, transparent: true, opacity: 0.32, roughness: 0.6, depthWrite: false });
    for (const g of meta.geoms) {
      if (!g.object || g.type === "mesh") continue;
      const isBin = meta.geoms.filter((x) => x.body === g.body).length > 1;
      if (isBin && !/\/drawer$/.test(meta.bodies[g.body])) continue; // furniture doesn't move: skip
      const m = new THREE.Mesh(primitive(g), mat);
      m.position.set(...g.pos); m.quaternion.copy(quat(g.quat));
      bodies[g.body].add(m);
    }
    this.scene.add(root);
    this.ghostRoot = root;
    this.ghost = { meta, traj: new Float32Array(buf), bodies, nb: meta.bodies.length,
                   duration: meta.times[meta.times.length - 1] || 0 };
  }

  #poseGhost(t) {
    const gh = this.ghost, times = gh.meta.times;
    let lo = 0, hi = times.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (times[mid] <= t) lo = mid; else hi = mid - 1; }
    for (let i = 0; i < gh.nb; i++) {
      const o = (lo * gh.nb + i) * 7, tr = gh.traj;
      gh.bodies[i].position.set(tr[o], tr[o + 1], tr[o + 2]);
      gh.bodies[i].quaternion.set(tr[o + 4], tr[o + 5], tr[o + 6], tr[o + 3]);
    }
  }

  setStyle(style) {
    if (style === this.style) return;
    this.style = style;
    if (this.styleBtn) this.styleBtn.textContent = style === "realistic" ? "Realistic" : "Plain";
    if (this.meta) this.#build();
  }

  #frameAt(t) {
    const times = this.meta.times;
    let lo = 0, hi = times.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (times[mid] <= t) lo = mid; else hi = mid - 1; }
    return lo;
  }

  #pose(f) {
    const tr = this.traj, nb = this.nb;
    for (let i = 0; i < nb; i++) {
      const o = (f * nb + i) * 7;
      this.bodies[i].position.set(tr[o], tr[o + 1], tr[o + 2]);
      this.bodies[i].quaternion.set(tr[o + 4], tr[o + 5], tr[o + 6], tr[o + 3]);
    }
  }

  #tick() {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (this.master) {  // compare view: time and camera follow the master viewer
      this.t = this.master.t;
      this.camera.position.copy(this.master.camera.position);
      this.camera.quaternion.copy(this.master.camera.quaternion);
      this.controls.target.copy(this.master.controls.target);
    }
    if (this.meta && this.meta.times.length) {
      if (this.playing && !this.master) {
        this.t += dt * this.speed;
        const span = Math.max(this.duration, ...this.followers.map((f) => f.duration || 0));
        if (this.t > span + 1.6) this.t = 0; // hold on the final state, then loop
      }
      const t = Math.min(this.t, this.duration);
      if (!this.editing) this.#pose(this.#frameAt(t));
      if (this.ghost) this.#poseGhost(Math.min(this.t, this.ghost.duration));
      this.range.value = Math.round((1000 * t) / Math.max(this.duration, 0.01));
      this.timeEl.textContent = `${t.toFixed(1)} s`;
      const recent = this.events.filter((e) => e.t <= t && t - e.t < 1.6).pop();
      this.caption.textContent = recent ? recent.text.replace(/: center at .*$/, "") : "";
      this.caption.style.opacity = recent ? 1 : 0;
    }
    if (!this.master) this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.#placeGoals();
  }

  #placeGoals() {
    if (!this.goalTags) return;
    const show = this.meta && this.t >= this.duration - 0.02;
    this.goalLayer.style.opacity = show ? 1 : 0;
    if (!show) return;
    const w = this.el.clientWidth, h = this.el.clientHeight, v = new THREE.Vector3();
    const spots = this.goalTags.map(({ tag, body }) => {
      v.copy(this.bodies[body].position); v.z += 0.07;
      v.project(this.camera);
      return { tag, x: (v.x * 0.5 + 0.5) * w, y: (-v.y * 0.5 + 0.5) * h, tw: tag.offsetWidth + 6, th: tag.offsetHeight + 4 };
    }).sort((a, b) => b.y - a.y);
    const placed = []; // greedy: move a label up until it clears the ones already placed
    for (const s of spots) {
      for (let guard = 0; guard < 12; guard++) {
        const hit = placed.find((p) => Math.abs(p.x - s.x) < (p.tw + s.tw) / 2 && Math.abs(p.y - s.y) < (p.th + s.th) / 2);
        if (!hit) break;
        s.y = hit.y - (hit.th + s.th) / 2;
      }
      placed.push(s);
      s.tag.style.transform = `translate(${s.x}px, ${s.y}px) translate(-50%, -100%)`;
    }
  }

  #resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  dispose() {
    this.unfollow();
    for (const f of this.followers) f.master = null;
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();
  }
}

window.SimViewer = SimViewer;
window.dispatchEvent(new Event("simviewer-ready"));
