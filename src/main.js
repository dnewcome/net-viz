import * as THREE from 'three';
import { NeuralNetwork } from './network/NeuralNetwork.js';
import { NetworkVisualizer } from './viz/NetworkVisualizer.js';
import { StimulusController } from './stimulus/StimulusController.js';
import { UIController } from './ui/UIController.js';

// ─── Default network topology ────────────────────────────────────────────────
const DEFAULT_CONFIG = [
  { size: 4,  activation: 'sigmoid' },  // input
  { size: 10, activation: 'relu'    },  // hidden
  { size: 10, activation: 'relu'    },  // hidden
  { size: 4,  activation: 'sigmoid' },  // output
];

// ─── Initialize core objects ─────────────────────────────────────────────────
const canvas = document.getElementById('canvas');

const network    = new NeuralNetwork(DEFAULT_CONFIG);
const visualizer = new NetworkVisualizer(canvas);
const stimulus   = new StimulusController(DEFAULT_CONFIG[0].size);

// Build initial visualization
visualizer.buildFromNetwork(network);
visualizer.syncWeights();

// ─── UI ──────────────────────────────────────────────────────────────────────
const ui = new UIController({
  network,
  visualizer,
  stimulus,
  onReconfigure(configs) {
    network.configure(configs);
    visualizer.buildFromNetwork(network);
    visualizer.syncWeights();
    stimulus.setInputSize(configs[0].size);
  },
});

// ─── HUD helpers ─────────────────────────────────────────────────────────────
const statsEl = document.getElementById('stats');
let frameCount = 0;
let fpsTime = 0;
let fps = 0;

function updateHUD(elapsed) {
  frameCount++;
  if (elapsed - fpsTime >= 1.0) {
    fps = frameCount;
    frameCount = 0;
    fpsTime = elapsed;
  }
  statsEl.textContent =
    `${fps} fps  |  ` +
    `${network.numLayers} layers  |  ` +
    `${network.getTotalNeurons()} neurons  |  ` +
    `${network.getTotalConnections()} weights  |  ` +
    `${visualizer._particles.length} particles`;
}

// ─── Animation loop ───────────────────────────────────────────────────────────
const clock = new THREE.Clock();
let lastForwardAt  = 0;
let lastWeightSync = 0;

const FORWARD_HZ    = 24;   // forward pass rate
const WEIGHT_SYNC_S = 0.15; // how often to refresh weight colors (seconds)

function animate() {
  requestAnimationFrame(animate);

  const dt      = clock.getDelta();
  const elapsed = clock.getElapsedTime();

  // Advance stimulus time
  stimulus.update(dt);

  // Forward pass at a fixed rate
  if (elapsed - lastForwardAt >= 1 / FORWARD_HZ) {
    const inputs = stimulus.getInputs();
    network.forward(inputs);

    visualizer.syncActivations();

    if (visualizer.showParticles) {
      visualizer.triggerSignalFlow();
    }

    lastForwardAt = elapsed;
  }

  // Weight color sync (cheaper than per-frame)
  if (elapsed - lastWeightSync >= WEIGHT_SYNC_S) {
    visualizer.syncWeights();
    lastWeightSync = elapsed;
  }

  // Update particles & orbit controls
  visualizer.update(dt);

  // Render with bloom
  visualizer.render();

  // HUD
  updateHUD(elapsed);
}

animate();
