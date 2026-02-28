# net-viz

A configurable, living neural network rendered in 3D — built as a piece of functional art.

Feed it stimuli, watch activations ripple through layers, reward or punish its behavior, and observe the weights drift over time. The network is real: every synapse fires, every gradient flows.

---

## Inspiration

Neural networks are usually invisible. We interact with their outputs — a classification, a generated image, a predicted token — but the internal machinery stays hidden behind loss curves and accuracy metrics.

This project makes the machinery visible.

The aesthetic draws from:

- **Bioluminescent organisms** — neurons glow cyan when active, dim to near-black at rest, the whole structure breathing in the dark like deep-sea life
- **Connectome visualizations** — the brain rendered as a wiring diagram, every synapse accounted for
- **Cyberpunk data aesthetics** — overbright bloom, dark void backgrounds, signal particles racing along glowing wires
- **Interactive art installations** — a system you can poke and watch respond, where the "output" is the visualization itself rather than a prediction

The longer-term vision: feed the network audio, motion, or sensor data as stimuli, arrange its layers to fill the interior of a 3D-printed skull or coral shape (now possible — see STL Layout below), and let it run as a physical installation — a brain that lives in a body.

---

## Running

```bash
npm install
npm run dev
```

Open **http://localhost:5173** (or whichever port Vite picks).

No build step needed for development. To produce a static bundle:

```bash
npm run build
npm run preview
```

---

## Controls

The GUI panel (top-right) has four sections:

### Architecture

Configure the network topology live. Changes apply when you click **↺ Apply** or add/remove a layer.

| Control | Description |
|---|---|
| **Nodes** | Number of neurons in that layer |
| **Activation** | Per-layer activation function (`sigmoid`, `relu`, `tanh`, `linear`) |
| **✕ Remove** | Delete a hidden layer (input/output layers are protected) |
| **+ Add Hidden Layer** | Insert a new hidden layer before the output |

The input and output layer sizes determine how many stimulus values are read and how many outputs are produced.

### Stimulus

Controls what gets fed into the input layer each frame.

| Mode | Description |
|---|---|
| `sine` | Each input neuron is driven by a sine wave. Frequency and amplitude are configurable per-neuron. |
| `noise` | Each input follows an independent smooth random walk bounded in [0, 1]. |
| `manual` | Sliders let you set each input value directly. |

In `sine` mode you'll see activation waves propagate through the network at the rhythm of the input frequencies. In `noise` mode the network churns continuously with no periodic structure.

### Training

The network learns via **eligibility traces** — a biologically-inspired mechanism that tracks which synapses were recently co-active (Hebbian "fire together, wire together").

| Control | Description |
|---|---|
| **Learning Rate** | Step size for weight updates (0.001 – 0.1) |
| **★ Reward (+)** | Strengthen recently active connections — the network will tend to repeat whatever it was just doing |
| **✗ Punish (−)** | Weaken recently active connections — suppress current behavior |
| **↺ Reset Weights** | Randomize all weights back to Xavier initialization |

There is no target output. Reward and punish are signals you supply based on what you observe — making this a form of interactive reinforcement learning driven by aesthetic judgment.

### Visualization

| Control | Description |
|---|---|
| **Particles** | Toggle signal-flow particles on/off |
| **Bloom Strength** | Intensity of the glow post-process |
| **Bloom Radius** | Spread of the bloom effect |
| **Bloom Threshold** | Minimum brightness before bloom kicks in |

Camera is controlled with **orbit controls**: drag to rotate, scroll to zoom, right-drag to pan.

### STL Layout

Import any watertight STL mesh and use it as a volumetric container for the neurons. A particle-physics simulation pushes neurons apart, keeps them inside the mesh, and maintains loose layer ordering along the mesh's longest axis — so the network organically fills the shape of any 3D form.

**Workflow:**

1. Click **Load STL…** and pick any `.stl` file (binary or ASCII).
2. The mesh is normalized to a 10-unit bounding box and voxelized into a 32³ grid. Watch the **Status** field for progress.
3. Once ready, neurons scatter inside the volume and the simulation starts automatically.
4. Adjust the physics sliders live — changes take effect immediately.
5. Click **Reset to Grid** to return to the standard flat-layer layout at any time.

Reconfiguring the network topology (Architecture panel → Apply) while an STL is loaded will re-scatter the neurons inside the same mesh, preserving the shape.

| Control | Description |
|---|---|
| **Load STL…** | Open a `.stl` file (binary or ASCII) |
| **Status** | Voxelization progress and current state |
| **▶ Start / ■ Stop** | Toggle the force simulation |
| **Repulsion** | Strength of neuron-to-neuron push-apart force |
| **Layer Guidance** | Pull strength toward each layer's target region along the principal axis |
| **Boundary** | Strength of the inward push from the mesh surface |
| **Damping** | Velocity damping — lower values = more jitter, higher = quicker settling |
| **Reset to Grid** | Stop simulation, remove mesh overlay, restore flat-layer layout |

The HUD adds an **energy** readout when the simulation is running — a mean kinetic energy value that falls toward zero as neurons settle.

---

## How It Works

### Neural Network (`src/network/`)

A standard fully-connected feedforward network implemented from scratch in plain JavaScript — no ML libraries.

**Forward pass** (`NeuralNetwork.forward(inputs)`):

Each layer computes `output[j] = activation(Σ weight[i,j] * input[i] + bias[j])`. Activations are stored for every layer after each pass so the visualizer can read them.

**Eligibility traces** (`NeuralNetwork.eligibility`):

After each forward pass, for every weight connecting neuron `i` in layer `l` to neuron `j` in layer `l+1`:

```
eligibility[i,j] = 0.92 * eligibility[i,j] + pre_activation[i] * post_activation[j]
```

This is a decaying Hebbian trace: connections between neurons that fired together recently have high eligibility. When a reward or punishment arrives, these are the connections that get adjusted.

**Reinforcement** (`NeuralNetwork.reinforce(reward, lr)`):

```
weight[i,j] += lr * reward * eligibility[i,j]
```

Weights are soft-clamped to `[−4, 4]` to prevent divergence.

### STL Layout (`src/viz/STLLayout.js`)

Loads an STL file, normalizes it, voxelizes the interior, and runs a force-directed simulation to position neurons inside the volume.

**Voxelization** — for each of the 32² scanlines (one per `(y, z)` grid pair), a ray is cast along the +X axis using the Möller–Trumbore triangle intersection algorithm. Intersection X-coordinates are sorted; odd-paired spans are marked inside. The result is a 32³ `Uint8Array` grid. Processing yields to the event loop every 4 rows so the UI stays responsive.

**Force simulation** — runs at 3 sub-steps per frame with a fixed `dt`:

1. **Repulsion** — all-pairs `K / dist²` force keeps neurons from collapsing together
2. **Layer guidance** — each neuron is gently pulled toward its layer's target point along the principal axis, loosely preserving the input→output depth ordering
3. **Boundary** — for each voxel in the neuron's 5×5×5 neighborhood that lies outside the mesh, a proportional inward push is applied
4. **Integration** — semi-implicit Euler with velocity damping; any neuron that escapes the mesh is snapped back to its previous position with a velocity bounce

The simulation fires `onTick(positions)` each frame, which the visualizer uses to update GPU buffers in place.

### Visualization (`src/viz/NetworkVisualizer.js`)

Built with [Three.js](https://threejs.org). Everything runs in a single WebGL render pass followed by an `UnrealBloomPass` post-process that makes bright pixels glow.

**Neurons** — `THREE.InstancedMesh` of spheres with per-instance color (via `setColorAt`). Activation is mapped to color: inactive neurons are nearly black `(0.02, 0.02, 0.06)`, fully active neurons are overbright cyan `(0.0, 2.2, 2.4)`. Values above 1.0 in linear color space trigger bloom, creating the halo effect.

**Connections** — A single `THREE.LineSegments` geometry containing all weight edges. Colors update each frame: positive weights trend blue, negative red, intensity proportional to `tanh(|weight| * 1.8)`. Even thin 1px lines glow when overbright — bloom does the visual work.

**Particles** — A second `THREE.InstancedMesh` (pool of 3000 spheres). After each forward pass, particles are spawned on connections where both the pre-synaptic and post-synaptic neurons are active and the weight is non-trivial. Each particle travels from source to destination over ~0.5 seconds, scaling up mid-path and fading at the endpoints. Positive-weight particles are warm white; negative-weight particles are pink.

**Layout** — Default: layers are spaced along the Z axis, neurons arranged in a square grid per layer. When an STL mesh is loaded, neuron positions are replaced by the force simulation output; GPU buffers (instance matrices and line segment endpoints) are updated in place each tick via `setPositions()` / `_syncPositionBuffers()`. The STL mesh itself is rendered as a semi-transparent shell + wireframe overlay using `showSTLMesh()` / `hideSTLMesh()`.

### Stimulus (`src/stimulus/StimulusController.js`)

Generates input vectors each frame. Sine mode assigns each input neuron its own oscillator with independently configurable frequency and phase (phases are evenly spread by default so inputs don't all peak simultaneously). Noise mode uses a random walk with `±0.12` steps per frame, clamped to `[0, 1]`.

### Animation Loop (`src/main.js`)

```
each frame:
  stimulus.update(dt)

  if time_since_last_forward >= 1/24s:
    inputs = stimulus.getInputs()
    network.forward(inputs)
    visualizer.syncActivations()
    visualizer.triggerSignalFlow()   ← spawn particles

  if time_since_last_weight_sync >= 0.15s:
    visualizer.syncWeights()         ← update line colors

  visualizer.update(dt)              ← advance particles, orbit controls
  visualizer.render()                ← bloom composer
```

The forward pass runs at 24 Hz (decoupled from render rate) and weight colors refresh every 150 ms — frequent enough to feel live but not thrashing GPU buffers every frame.

---

## File Structure

```
net-viz/
├── index.html                       # Full-screen canvas + HUD overlay
├── package.json
└── src/
    ├── main.js                      # Entry point, animation loop, STL wiring
    ├── network/
    │   ├── NeuralNetwork.js         # Forward pass, eligibility traces, reinforce
    │   └── activations.js           # sigmoid / relu / tanh / linear + viz normalizers
    ├── viz/
    │   ├── NetworkVisualizer.js     # Three.js scene, bloom, neuron/connection/particle meshes
    │   └── STLLayout.js             # STL loader, 32³ voxelizer, force-directed neuron sim
    ├── stimulus/
    │   └── StimulusController.js    # Sine, noise, and manual input generators
    └── ui/
        └── UIController.js          # lil-gui panels, layer editor, STL panel
```

---

## Planned Features

- **Audio stimulus** — tap the microphone and map frequency bands to input neurons; the network will react to sound in real time
- **Weight snapshots** — save and restore trained weight states
- **Multiple layout modes** — circular layers, 3D grid, free-form scatter
- **Custom activation functions** — define your own via a text input
- **Export** — record the visualization as a video or sequence of frames

---

## Dependencies

| Package | Purpose |
|---|---|
| [three](https://threejs.org) | 3D rendering, post-processing bloom |
| [lil-gui](https://lil-gui.georgealways.com) | GUI panels |
| [vite](https://vitejs.dev) | Dev server and bundler |
