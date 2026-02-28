import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { normalizeForViz } from '../network/activations.js';

// Layout constants
const LAYER_SPACING = 4.0;
const NODE_SPACING = 1.6;
const NEURON_RADIUS = 0.22;
const MAX_PARTICLES = 3000;
const PARTICLE_BASE_SPEED = 2.2; // world-units/second along a unit-length edge

// Colors (raw, pre-bloom — values >1 will bloom)
const COLOR_BG = 0x000000;
const COLOR_NEURON_DIM = new THREE.Color(0.02, 0.02, 0.06);
const COLOR_NEURON_ACTIVE = new THREE.Color(0.0, 2.2, 2.4); // overbright cyan → big glow
const COLOR_POS = new THREE.Color(0.1, 0.5, 1.8);  // bright blue
const COLOR_NEG = new THREE.Color(1.8, 0.1, 0.1);  // bright red
const COLOR_PARTICLE = new THREE.Color(2.0, 2.0, 0.8); // warm white

// Temp objects — reused every frame to avoid GC pressure
const _tmpObj = new THREE.Object3D();
const _tmpPos = new THREE.Vector3();
const _tmpColor = new THREE.Color();

export class NetworkVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.network = null;
    this.showParticles = true;

    // Built after buildFromNetwork():
    this.neuronPositions = [];   // [layer][node] → Vector3
    this.neuronOffset = [];      // [layer] → start index into instancedMesh
    this._connectionMap = [];    // [{layer, from, to}]
    this._particles = [];
    this._particlePool = [];

    this._setupRenderer();
    this._setupScene();
    this._setupCamera();
    this._setupPostprocessing();
    this._setupParticleMesh();

    this._ro = new ResizeObserver(() => this._onResize());
    this._ro.observe(canvas.parentElement ?? canvas);
    this._onResize();
  }

  // ─── Setup ──────────────────────────────────────────────────────────────────

  _setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(COLOR_BG);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  _setupScene() {
    this.scene = new THREE.Scene();
    // Very subtle fog for depth
    this.scene.fog = new THREE.FogExp2(COLOR_BG, 0.018);
  }

  _setupCamera() {
    const w = this.canvas.clientWidth || 800;
    const h = this.canvas.clientHeight || 600;
    this.camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 500);
    this.camera.position.set(0, 4, 18);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 120;
  }

  _setupPostprocessing() {
    const w = this.canvas.clientWidth || 800;
    const h = this.canvas.clientHeight || 600;

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(w, h),
      1.4,   // strength
      0.55,  // radius
      0.08   // threshold — low so dim neurons still glow faintly
    );
    this.composer.addPass(this.bloomPass);
  }

  _setupParticleMesh() {
    const geo = new THREE.SphereGeometry(0.07, 6, 4);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.particleMesh = new THREE.InstancedMesh(geo, mat, MAX_PARTICLES);
    this.particleMesh.count = 0;
    this.particleMesh.frustumCulled = false;
    this.scene.add(this.particleMesh);

    // Initialize all instances far offscreen
    _tmpObj.position.set(0, -9999, 0);
    _tmpObj.scale.setScalar(0.001);
    _tmpObj.updateMatrix();
    for (let i = 0; i < MAX_PARTICLES; i++) {
      this.particleMesh.setMatrixAt(i, _tmpObj.matrix);
      this.particleMesh.setColorAt(i, COLOR_PARTICLE);
    }
    this.particleMesh.instanceMatrix.needsUpdate = true;
    this.particleMesh.instanceColor.needsUpdate = true;
  }

  // ─── Build from network ──────────────────────────────────────────────────────

  buildFromNetwork(network) {
    this.network = network;
    this._particles = [];
    this._particlePool = [];

    if (this.neuronMesh) { this.scene.remove(this.neuronMesh); this.neuronMesh.dispose(); }
    if (this.connectionLines) { this.scene.remove(this.connectionLines); this.connectionLines.geometry.dispose(); }

    this._computePositions(network);
    this._buildNeurons(network);
    this._buildConnections(network);
    this._positionCamera(network);
  }

  _computePositions(network) {
    this.neuronPositions = [];
    this.neuronOffset = [];
    let offset = 0;

    for (let l = 0; l < network.numLayers; l++) {
      const n = network.layerConfigs[l].size;
      const cols = Math.ceil(Math.sqrt(n));
      const rows = Math.ceil(n / cols);
      const z = (l - (network.numLayers - 1) / 2) * LAYER_SPACING;

      const positions = [];
      for (let i = 0; i < n; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = (col - (cols - 1) / 2) * NODE_SPACING;
        const y = ((rows - 1) / 2 - row) * NODE_SPACING;
        positions.push(new THREE.Vector3(x, y, z));
      }

      this.neuronPositions.push(positions);
      this.neuronOffset.push(offset);
      offset += n;
    }
  }

  _buildNeurons(network) {
    const total = network.getTotalNeurons();
    const geo = new THREE.SphereGeometry(NEURON_RADIUS, 20, 14);
    // MeshBasicMaterial + per-instance color → perfect for bloom
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });

    this.neuronMesh = new THREE.InstancedMesh(geo, mat, total);
    this.neuronMesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    this.neuronMesh.frustumCulled = false;

    let idx = 0;
    for (let l = 0; l < network.numLayers; l++) {
      for (let i = 0; i < network.layerConfigs[l].size; i++) {
        _tmpObj.position.copy(this.neuronPositions[l][i]);
        _tmpObj.scale.setScalar(1);
        _tmpObj.updateMatrix();
        this.neuronMesh.setMatrixAt(idx, _tmpObj.matrix);
        this.neuronMesh.setColorAt(idx, COLOR_NEURON_DIM);
        idx++;
      }
    }
    this.neuronMesh.instanceMatrix.needsUpdate = true;
    this.neuronMesh.instanceColor.needsUpdate = true;
    this.scene.add(this.neuronMesh);
  }

  _buildConnections(network) {
    const totalConn = network.getTotalConnections();
    // Two vec3 per line segment
    const positions = new Float32Array(totalConn * 6);
    const colors = new Float32Array(totalConn * 6);

    this._connectionMap = [];
    let c = 0;

    for (let l = 0; l < network.numLayers - 1; l++) {
      const inN = network.layerConfigs[l].size;
      const outN = network.layerConfigs[l + 1].size;
      for (let i = 0; i < inN; i++) {
        for (let j = 0; j < outN; j++) {
          const s = this.neuronPositions[l][i];
          const e = this.neuronPositions[l + 1][j];

          positions[c * 6 + 0] = s.x; positions[c * 6 + 1] = s.y; positions[c * 6 + 2] = s.z;
          positions[c * 6 + 3] = e.x; positions[c * 6 + 4] = e.y; positions[c * 6 + 5] = e.z;

          this._connectionMap.push({ layer: l, from: i, to: j });
          c++;
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75 });
    this.connectionLines = new THREE.LineSegments(geo, mat);
    this.connectionLines.frustumCulled = false;
    this.scene.add(this.connectionLines);
  }

  _positionCamera(network) {
    const span = (network.numLayers - 1) * LAYER_SPACING;
    this.camera.position.set(0, 5, span * 0.5 + 14);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  // ─── Per-frame sync ──────────────────────────────────────────────────────────

  syncActivations() {
    if (!this.network || !this.neuronMesh?.instanceColor) return;
    const network = this.network;
    const colArr = this.neuronMesh.instanceColor.array;

    let base = 0;
    for (let l = 0; l < network.numLayers; l++) {
      const acts = network.activations[l];
      const actName = network.layerConfigs[l].activation;
      for (let i = 0; i < acts.length; i++) {
        const norm = normalizeForViz(acts[i], actName);
        // Lerp from dim base to overbright cyan — bloom amplifies the glow
        const r = COLOR_NEURON_DIM.r + (COLOR_NEURON_ACTIVE.r - COLOR_NEURON_DIM.r) * norm;
        const g = COLOR_NEURON_DIM.g + (COLOR_NEURON_ACTIVE.g - COLOR_NEURON_DIM.g) * norm;
        const b = COLOR_NEURON_DIM.b + (COLOR_NEURON_ACTIVE.b - COLOR_NEURON_DIM.b) * norm;
        colArr[base * 3]     = r;
        colArr[base * 3 + 1] = g;
        colArr[base * 3 + 2] = b;
        base++;
      }
    }
    this.neuronMesh.instanceColor.needsUpdate = true;
  }

  syncWeights() {
    if (!this.network || !this.connectionLines) return;
    const network = this.network;
    const colArr = this.connectionLines.geometry.attributes.color.array;

    for (let c = 0; c < this._connectionMap.length; c++) {
      const { layer, from, to } = this._connectionMap[c];
      const w = network.getWeight(layer, from, to);
      const intensity = Math.tanh(Math.abs(w) * 1.8); // compress to 0-1

      let r, g, b;
      if (w >= 0) {
        r = COLOR_POS.r * intensity; g = COLOR_POS.g * intensity; b = COLOR_POS.b * intensity;
      } else {
        r = COLOR_NEG.r * intensity; g = COLOR_NEG.g * intensity; b = COLOR_NEG.b * intensity;
      }

      // Both endpoints same color
      const vi = c * 6;
      colArr[vi]     = r; colArr[vi + 1] = g; colArr[vi + 2] = b;
      colArr[vi + 3] = r; colArr[vi + 4] = g; colArr[vi + 5] = b;
    }
    this.connectionLines.geometry.attributes.color.needsUpdate = true;
  }

  // ─── Particle system ─────────────────────────────────────────────────────────

  triggerSignalFlow() {
    if (!this.network || !this.showParticles) return;
    const network = this.network;

    for (let c = 0; c < this._connectionMap.length; c++) {
      if (this._particles.length >= MAX_PARTICLES - 3) break;

      const { layer, from, to } = this._connectionMap[c];
      const w = network.getWeight(layer, from, to);
      const preAct = normalizeForViz(network.activations[layer][from], network.layerConfigs[layer].activation);
      const postAct = normalizeForViz(network.activations[layer + 1][to], network.layerConfigs[layer + 1].activation);

      const activity = preAct * postAct * Math.abs(w);
      if (activity < 0.04) continue;

      // Spawn 1–3 particles staggered in time
      const count = Math.ceil(activity * 2.5);
      for (let p = 0; p < count; p++) {
        const particle = this._particlePool.pop() ?? {};
        particle.start = this.neuronPositions[layer][from];
        particle.end   = this.neuronPositions[layer + 1][to];
        particle.t     = -(Math.random() * 0.4); // stagger start
        const dist = particle.start.distanceTo(particle.end);
        particle.speed  = PARTICLE_BASE_SPEED / Math.max(0.1, dist);
        particle.weight = w;
        this._particles.push(particle);
      }
    }
  }

  _updateParticles(dt) {
    let visCount = 0;
    const toRecycle = [];

    for (let i = 0; i < this._particles.length; i++) {
      const p = this._particles[i];
      p.t += dt * p.speed;

      if (p.t > 1.0) {
        toRecycle.push(i);
        continue;
      }

      const t = Math.max(0, p.t);
      _tmpPos.lerpVectors(p.start, p.end, t);

      // Scale: small at ends, full in middle
      const scale = Math.sin(t * Math.PI) * 0.8 + 0.2;
      const hidden = p.t < 0;

      _tmpObj.position.copy(_tmpPos);
      _tmpObj.scale.setScalar(hidden ? 0 : scale);
      _tmpObj.updateMatrix();
      this.particleMesh.setMatrixAt(visCount, _tmpObj.matrix);

      // Color: warm white for positive weights, pink for negative
      _tmpColor.copy(COLOR_PARTICLE);
      if (p.weight < 0) _tmpColor.setRGB(2.0, 0.8, 0.8);
      this.particleMesh.setColorAt(visCount, _tmpColor);

      visCount++;
    }

    // Recycle dead particles (reverse order to preserve indices)
    for (let i = toRecycle.length - 1; i >= 0; i--) {
      const p = this._particles.splice(toRecycle[i], 1)[0];
      this._particlePool.push(p);
    }

    this.particleMesh.count = visCount;
    if (visCount > 0) {
      this.particleMesh.instanceMatrix.needsUpdate = true;
      this.particleMesh.instanceColor.needsUpdate = true;
    }
  }

  // ─── Main loop hooks ─────────────────────────────────────────────────────────

  update(dt) {
    this._updateParticles(dt);
    this.controls.update();
  }

  render() {
    this.composer.render();
  }

  // ─── Resize ──────────────────────────────────────────────────────────────────

  _onResize() {
    const w = this.canvas.clientWidth  || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
