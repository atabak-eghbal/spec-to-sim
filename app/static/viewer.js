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

    this.#stage();
    this.#ui();
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.playing = true; this.t = 0; this.speed = 1; this.last = performance.now();
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
      <button type="button" class="v-speed" aria-label="Playback speed">1×</button>`;
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
    this.range.oninput = () => { this.t = (this.range.value / 1000) * this.duration; this.setPlaying(false); };
    this.renderer.domElement.addEventListener("pointerdown", () => { this.hint.style.opacity = 0; }, { once: true });
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
    this.traj = new Float32Array(buf);
    this.nb = meta.bodies.length;
    this.duration = meta.times.length ? meta.times[meta.times.length - 1] : 0;
    this.root.clear();
    this.bodies = meta.bodies.map(() => { const g = new THREE.Group(); this.root.add(g); return g; });
    const mats = new Map();
    for (const g of meta.geoms) {
      const key = g.rgba.join(",") + (g.object ? "o" : "");
      if (!mats.has(key)) {
        mats.set(key, new THREE.MeshStandardMaterial({
          color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]).convertSRGBToLinear(),
          roughness: g.object ? 0.45 : 0.38, metalness: g.object ? 0.0 : 0.08,
          transparent: g.rgba[3] < 1, opacity: g.rgba[3] }));
      }
      const geo = g.type === "mesh" ? meshes[g.mesh] : primitive(g);
      if (!geo) continue;
      const mesh = new THREE.Mesh(geo, mats.get(key));
      mesh.position.set(...g.pos);
      mesh.quaternion.copy(quat(g.quat));
      mesh.castShadow = true; mesh.receiveShadow = true;
      this.bodies[g.body].add(mesh);
    }
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
    if (this.meta && this.meta.times.length) {
      if (this.playing) {
        this.t += dt * this.speed;
        if (this.t > this.duration + 1.2) this.t = 0; // short hold on the final state, then loop
      }
      const t = Math.min(this.t, this.duration);
      this.#pose(this.#frameAt(t));
      this.range.value = Math.round((1000 * t) / Math.max(this.duration, 0.01));
      this.timeEl.textContent = `${t.toFixed(1)} s`;
      const recent = this.events.filter((e) => e.t <= t && t - e.t < 1.6).pop();
      this.caption.textContent = recent ? recent.text.replace(/: center at .*$/, "") : "";
      this.caption.style.opacity = recent ? 1 : 0;
    }
    this.controls.update();
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
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();
  }
}

window.SimViewer = SimViewer;
window.dispatchEvent(new Event("simviewer-ready"));
