import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

const GRID_SIZE  = 32;
const SUB_STEPS  = 3;
const FIXED_DT   = 1 / (60 * SUB_STEPS);
const MIN_DIST_SQ = 0.09; // (0.3 units)²

// Temp vector — reused in _subStep to avoid GC pressure
const _d = new THREE.Vector3();

export class STLLayout {
  constructor() {
    // Callbacks
    this.onProgress = null; // (pct: 0–1, label: string) => void
    this.onReady    = null; // (geometry: THREE.BufferGeometry) => void
    this.onTick     = null; // (positions: THREE.Vector3[][]) => void

    // Tunable params (exposed to GUI)
    this.kRepel    = 0.6;
    this.kLayer    = 0.04;
    this.kBoundary = 0.4;
    this.damping   = 0.85;

    // Internal state
    this._voxels      = null;    // Uint8Array [GRID^3]
    this._insideCache = [];      // [{ix,iy,iz}] shuffled inside cells
    this._bounds      = null;    // {min, max, size} world-space
    this._neurons     = [];      // sim particle state
    this._layerTargets = [];     // Vector3[] per layer
    this._positions   = [];      // Vector3[layer][node] — output

    this._rafId   = null;
    this.isLoaded = false;
    this.energy   = 0;           // mean kinetic energy (convergence indicator)
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  async loadFile(file) {
    this.isLoaded = false;
    this.stop();

    const loader = new STLLoader();
    const buffer = await file.arrayBuffer();
    const geo    = loader.parse(buffer);

    // ── Normalize: center + scale so max dimension = 10 ──
    geo.computeBoundingBox();
    const cent = new THREE.Vector3();
    geo.boundingBox.getCenter(cent);
    geo.translate(-cent.x, -cent.y, -cent.z);

    geo.computeBoundingBox();
    const sizeV  = new THREE.Vector3();
    geo.boundingBox.getSize(sizeV);
    const maxDim = Math.max(sizeV.x, sizeV.y, sizeV.z);
    const s      = 10 / maxDim;
    geo.scale(s, s, s);

    geo.computeBoundingBox();
    this._bounds = {
      min:  geo.boundingBox.min.clone(),
      max:  geo.boundingBox.max.clone(),
      size: new THREE.Vector3(),
    };
    geo.boundingBox.getSize(this._bounds.size);

    await this._voxelize(geo);

    this.isLoaded = true;
    if (this.onReady) this.onReady(geo);
  }

  /** Place neurons inside the voxel volume. Call after loadFile resolves. */
  initNeurons(network) {
    if (!this.isLoaded || this._insideCache.length === 0) return;

    const { min, size } = this._bounds;
    const numLayers     = network.numLayers;

    // ── Principal axis = longest bounding-box dimension ──
    let axis = 'z';
    if (size.x >= size.y && size.x >= size.z) axis = 'x';
    else if (size.y >= size.z)                 axis = 'y';

    this._layerTargets = [];
    this._positions    = [];
    this._neurons      = [];

    // Layer target positions: evenly spaced along principal axis (±37% from center)
    for (let l = 0; l < numLayers; l++) {
      const t      = numLayers === 1 ? 0.5 : l / (numLayers - 1);
      const axisV  = min[axis] + size[axis] * (t * 0.74 + 0.13);
      const center = this._bounds.min.clone().add(this._bounds.max).multiplyScalar(0.5);
      center[axis] = axisV;
      this._layerTargets.push(center);
    }

    // Place neurons near their layer targets
    for (let l = 0; l < numLayers; l++) {
      const n   = network.layerConfigs[l].size;
      const pos = [];
      for (let i = 0; i < n; i++) {
        const p = this._findInsidePointNear(this._layerTargets[l], 40);
        pos.push(p.clone());
        this._neurons.push({
          pos:   p.clone(),
          prev:  p.clone(),
          vel:   new THREE.Vector3(
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1,
          ),
          force: new THREE.Vector3(),
          layer: l,
          index: i,
        });
      }
      this._positions.push(pos);
    }
  }

  start() {
    if (this._rafId !== null) return;
    const loop = () => {
      this._rafId = requestAnimationFrame(loop);
      this._step();
    };
    this._rafId = requestAnimationFrame(loop);
  }

  stop() {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  getPositions() { return this._positions; }

  // ─── Voxelization (32³ scanline approach) ───────────────────────────────────

  async _voxelize(geo) {
    const G = GRID_SIZE;
    this._voxels      = new Uint8Array(G * G * G);
    this._insideCache = [];

    const { min, size } = this._bounds;
    const posAttr   = geo.getAttribute('position');
    const indexAttr = geo.index;
    const triCount  = indexAttr ? indexAttr.count / 3 : posAttr.count / 3;

    const v0 = new THREE.Vector3();
    const v1 = new THREE.Vector3();
    const v2 = new THREE.Vector3();

    const getTri = (tri) => {
      if (indexAttr) {
        v0.fromBufferAttribute(posAttr, indexAttr.getX(tri * 3));
        v1.fromBufferAttribute(posAttr, indexAttr.getX(tri * 3 + 1));
        v2.fromBufferAttribute(posAttr, indexAttr.getX(tri * 3 + 2));
      } else {
        v0.fromBufferAttribute(posAttr, tri * 3);
        v1.fromBufferAttribute(posAttr, tri * 3 + 1);
        v2.fromBufferAttribute(posAttr, tri * 3 + 2);
      }
    };

    const rayOX = min.x - 1; // ray origin left of mesh — ensures t > 0 for all hits

    for (let iy = 0; iy < G; iy++) {
      if (this.onProgress) this.onProgress(iy / G, `Voxelizing row ${iy}/${G}…`);

      // Yield every 4 rows to keep the UI responsive
      if (iy % 4 === 0) await new Promise(r => setTimeout(r, 0));

      for (let iz = 0; iz < G; iz++) {
        const wy = min.y + (iy + 0.5) / G * size.y;
        const wz = min.z + (iz + 0.5) / G * size.z;

        // Collect X-intersections along this scanline
        const hits = [];
        for (let tri = 0; tri < triCount; tri++) {
          getTri(tri);
          const wx = _rayTriX(rayOX, wy, wz, v0, v1, v2);
          if (wx !== null) hits.push(wx);
        }
        if (!hits.length) continue;
        hits.sort((a, b) => a - b);

        // Deduplicate hits that are very close (shared edges / vertices)
        const deduped = [hits[0]];
        for (let h = 1; h < hits.length; h++) {
          if (hits[h] - deduped[deduped.length - 1] > 1e-5) deduped.push(hits[h]);
        }

        // Mark voxels inside odd-pair spans
        for (let h = 0; h + 1 < deduped.length; h += 2) {
          const x0  = deduped[h];
          const x1  = deduped[h + 1];
          const ix0 = Math.max(0,     Math.floor((x0 - min.x) / size.x * G));
          const ix1 = Math.min(G - 1, Math.floor((x1 - min.x) / size.x * G));
          for (let ix = ix0; ix <= ix1; ix++) {
            this._voxels[ix * G * G + iy * G + iz] = 1;
          }
        }
      }
    }

    // Build shuffled inside-cell cache for fast "find inside point near target"
    for (let ix = 0; ix < G; ix++) {
      for (let iy2 = 0; iy2 < G; iy2++) {
        for (let iz2 = 0; iz2 < G; iz2++) {
          if (this._voxels[ix * G * G + iy2 * G + iz2]) {
            this._insideCache.push({ ix, iy: iy2, iz: iz2 });
          }
        }
      }
    }
    // Fisher-Yates shuffle
    for (let i = this._insideCache.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      const t = this._insideCache[i];
      this._insideCache[i] = this._insideCache[j];
      this._insideCache[j] = t;
    }

    if (this.onProgress) this.onProgress(1, 'Ready');
  }

  _isInside(pos) {
    const { min, size } = this._bounds;
    const G  = GRID_SIZE;
    const ix = Math.floor((pos.x - min.x) / size.x * G);
    const iy = Math.floor((pos.y - min.y) / size.y * G);
    const iz = Math.floor((pos.z - min.z) / size.z * G);
    if (ix < 0 || ix >= G || iy < 0 || iy >= G || iz < 0 || iz >= G) return false;
    return this._voxels[ix * G * G + iy * G + iz] === 1;
  }

  _cellToWorld(ix, iy, iz) {
    const { min, size } = this._bounds;
    const G = GRID_SIZE;
    return new THREE.Vector3(
      min.x + (ix + 0.5) / G * size.x,
      min.y + (iy + 0.5) / G * size.y,
      min.z + (iz + 0.5) / G * size.z,
    );
  }

  _findInsidePointNear(target, maxTries) {
    const { size } = this._bounds;
    const spread   = size.length() * 0.25;

    for (let i = 0; i < maxTries; i++) {
      const p = new THREE.Vector3(
        target.x + (Math.random() - 0.5) * spread,
        target.y + (Math.random() - 0.5) * spread,
        target.z + (Math.random() - 0.5) * spread,
      );
      if (this._isInside(p)) return p;
    }

    // Fallback: sample cache, pick closest to target
    let best = null, bestDist = Infinity;
    const sample = Math.min(60, this._insideCache.length);
    for (let i = 0; i < sample; i++) {
      const c = this._insideCache[(Math.random() * this._insideCache.length) | 0];
      const p = this._cellToWorld(c.ix, c.iy, c.iz);
      const d = p.distanceToSquared(target);
      if (d < bestDist) { bestDist = d; best = p; }
    }
    return best ?? this._cellToWorld(
      this._insideCache[0].ix,
      this._insideCache[0].iy,
      this._insideCache[0].iz,
    );
  }

  // ─── Force simulation ────────────────────────────────────────────────────────

  _step() {
    if (this._neurons.length === 0) return;

    for (let s = 0; s < SUB_STEPS; s++) this._subStep(FIXED_DT);

    // Sync _positions from neuron state
    let n = 0;
    for (let l = 0; l < this._positions.length; l++) {
      for (let i = 0; i < this._positions[l].length; i++) {
        this._positions[l][i].copy(this._neurons[n++].pos);
      }
    }

    if (this.onTick) this.onTick(this._positions);
  }

  _subStep(dt) {
    const neurons = this._neurons;
    const N       = neurons.length;
    if (N === 0) return;

    // Zero forces
    for (let i = 0; i < N; i++) neurons[i].force.set(0, 0, 0);

    // ── 1. All-pairs repulsion ──
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        _d.subVectors(neurons[i].pos, neurons[j].pos);
        const f = this.kRepel / Math.max(_d.lengthSq(), MIN_DIST_SQ);
        _d.normalize().multiplyScalar(f);
        neurons[i].force.add(_d);
        neurons[j].force.sub(_d);
      }
    }

    // ── 2. Layer guidance — weak pull toward layer centroid target ──
    for (let i = 0; i < N; i++) {
      const tgt = this._layerTargets[neurons[i].layer];
      if (!tgt) continue;
      _d.subVectors(tgt, neurons[i].pos).multiplyScalar(this.kLayer);
      neurons[i].force.add(_d);
    }

    // ── 3. Boundary soft force — push away from adjacent outside voxels ──
    const G              = GRID_SIZE;
    const { min, size }  = this._bounds;
    for (let i = 0; i < N; i++) {
      const pos = neurons[i].pos;
      const cx  = Math.floor((pos.x - min.x) / size.x * G);
      const cy  = Math.floor((pos.y - min.y) / size.y * G);
      const cz  = Math.floor((pos.z - min.z) / size.z * G);

      for (let dx = -2; dx <= 2; dx++) {
        for (let dy = -2; dy <= 2; dy++) {
          for (let dz = -2; dz <= 2; dz++) {
            if (dx === 0 && dy === 0 && dz === 0) continue;
            const nx = cx + dx, ny = cy + dy, nz = cz + dz;
            const outside =
              nx < 0 || nx >= G || ny < 0 || ny >= G || nz < 0 || nz >= G ||
              !this._voxels[nx * G * G + ny * G + nz];
            if (!outside) continue;

            _d.set(
              min.x + (nx + 0.5) / G * size.x,
              min.y + (ny + 0.5) / G * size.y,
              min.z + (nz + 0.5) / G * size.z,
            );
            _d.subVectors(pos, _d); // direction: outside-voxel → neuron (inward)
            const distSq = Math.max(_d.lengthSq(), 0.01);
            _d.normalize().multiplyScalar(this.kBoundary / distSq);
            neurons[i].force.add(_d);
          }
        }
      }
    }

    // ── 4. Integrate + hard clamp ──
    let ke = 0;
    for (let i = 0; i < N; i++) {
      const n = neurons[i];
      n.prev.copy(n.pos);
      n.vel.addScaledVector(n.force, dt).multiplyScalar(this.damping);
      n.pos.addScaledVector(n.vel, dt);
      ke += n.vel.lengthSq();

      // Hard clamp: if outside mesh, revert and bounce
      if (!this._isInside(n.pos)) {
        n.pos.copy(n.prev);
        n.vel.multiplyScalar(-0.3);
      }
    }
    this.energy = ke / N;
  }
}

// ─── Möller-Trumbore: ray along +X at (y=wy, z=wz), origin at (rayOX, wy, wz) ──
// Returns world-X coordinate of intersection (>= rayOX) or null.
function _rayTriX(rayOX, wy, wz, v0, v1, v2) {
  const EPSILON = 1e-8;

  const e1x = v1.x - v0.x, e1y = v1.y - v0.y, e1z = v1.z - v0.z;
  const e2x = v2.x - v0.x, e2y = v2.y - v0.y, e2z = v2.z - v0.z;

  // dir=(1,0,0) → h = dir × e2 = (0, -e2z, e2y)
  const hy = -e2z, hz = e2y;
  const a  = e1y * hy + e1z * hz; // e1 · h  (hx=0 so e1x*hx drops)
  if (Math.abs(a) < EPSILON) return null; // ray parallel to triangle

  const f  = 1 / a;
  const sx = rayOX - v0.x, sy = wy - v0.y, sz = wz - v0.z;

  const u = f * (sy * hy + sz * hz); // s · h / a
  if (u < -EPSILON || u > 1 + EPSILON) return null;

  const qx = sy * e1z - sz * e1y;
  const qy = sz * e1x - sx * e1z;
  const qz = sx * e1y - sy * e1x;

  const v = f * qx; // dir · q / a  (dir.x=1 → picks qx)
  if (v < -EPSILON || u + v > 1 + EPSILON) return null;

  const t = f * (e2x * qx + e2y * qy + e2z * qz);
  if (t < EPSILON) return null; // behind origin

  return rayOX + t; // world-X of intersection
}
