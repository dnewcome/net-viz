import GUI from 'lil-gui';

export class UIController {
  /**
   * @param {object} opts
   * @param {import('../network/NeuralNetwork.js').NeuralNetwork} opts.network
   * @param {import('../viz/NetworkVisualizer.js').NetworkVisualizer} opts.visualizer
   * @param {import('../stimulus/StimulusController.js').StimulusController} opts.stimulus
   * @param {function(Array)} opts.onReconfigure  - called when topology changes
   */
  constructor({ network, visualizer, stimulus, onReconfigure }) {
    this.network = network;
    this.visualizer = visualizer;
    this.stimulus = stimulus;
    this.onReconfigure = onReconfigure;

    // Mutable state tracked by GUI
    this.learningRate = 0.015;

    this.gui = new GUI({ title: 'NET·VIZ', width: 290 });
    // Mirror the current layer configs as editable objects
    this._layerConfigs = network.layerConfigs.map(l => ({ ...l }));

    this._build();
  }

  // ─── Public rebuild (called after reconfigure) ───────────────────────────────

  rebuild(newLayerConfigs) {
    this._layerConfigs = newLayerConfigs.map(l => ({ ...l }));
    this.gui.destroy();
    this.gui = new GUI({ title: 'NET·VIZ', width: 290 });
    this._build();
  }

  // ─── Build all panels ────────────────────────────────────────────────────────

  _build() {
    this._buildArchPanel();
    this._buildStimulusPanel();
    this._buildTrainingPanel();
    this._buildVizPanel();
    if (this._stlLayout) this._buildSTLPanel();
  }

  _buildArchPanel() {
    const folder = this.gui.addFolder('Architecture');

    this._layerConfigs.forEach((cfg, i) => {
      const isInput  = i === 0;
      const isOutput = i === this._layerConfigs.length - 1;
      const label = isInput ? '→ Input' : isOutput ? '← Output' : `Layer ${i}`;
      const lf = folder.addFolder(label);

      lf.add(cfg, 'size', 1, 32, 1).name('Nodes');
      lf.add(cfg, 'activation', ['sigmoid', 'relu', 'tanh', 'linear']).name('Activation');

      // Only hidden layers can be removed
      if (!isInput && !isOutput) {
        lf.add({
          remove: () => {
            this._layerConfigs.splice(i, 1);
            this._apply();
          },
        }, 'remove').name('✕ Remove');
      }
      lf.open();
    });

    folder.add({
      add: () => {
        // Insert new hidden layer before output
        this._layerConfigs.splice(this._layerConfigs.length - 1, 0, {
          size: 6,
          activation: 'relu',
        });
        this._apply();
      },
    }, 'add').name('+ Add Hidden Layer');

    folder.add({ apply: () => this._apply() }, 'apply').name('↺ Apply');
    folder.open();
  }

  _buildStimulusPanel() {
    const folder = this.gui.addFolder('Stimulus');

    folder.add(this.stimulus, 'mode', ['sine', 'noise', 'manual'])
      .name('Mode')
      .onChange(() => this._rebuildStimulusControls(subFolder));

    const subFolder = folder.addFolder('Controls');
    this._rebuildStimulusControls(subFolder);
    folder.open();
  }

  _rebuildStimulusControls(folder) {
    // Remove all child controllers
    [...folder.controllers].forEach(c => c.destroy());
    [...folder.folders].forEach(f => f.destroy());

    if (this.stimulus.mode === 'sine') {
      this.stimulus.sineParams.forEach((p, i) => {
        folder.add(p, 'frequency', 0.05, 8.0, 0.05).name(`#${i} freq`);
        folder.add(p, 'amplitude', 0.0, 2.0, 0.05).name(`#${i} amp`);
      });
    } else if (this.stimulus.mode === 'manual') {
      for (let i = 0; i < this.stimulus.inputSize; i++) {
        folder.add(this.stimulus.manualValues, String(i), 0, 1, 0.01).name(`Input ${i}`);
      }
    } else {
      folder.add({ info: 'Smooth random walk' }, 'info').name('Mode').disable();
    }

    folder.open();
  }

  _buildTrainingPanel() {
    const folder = this.gui.addFolder('Training');

    folder.add(this, 'learningRate', 0.001, 0.1, 0.001).name('Learning Rate');

    folder.add({
      reward: () => {
        this.network.reinforce(1.0, this.learningRate);
        this.visualizer.syncWeights();
      },
    }, 'reward').name('★  Reward  (+)');

    folder.add({
      punish: () => {
        this.network.reinforce(-1.0, this.learningRate);
        this.visualizer.syncWeights();
      },
    }, 'punish').name('✗  Punish  (−)');

    folder.add({
      reset: () => {
        this.network.configure(this._layerConfigs);
        this.visualizer.syncWeights();
      },
    }, 'reset').name('↺  Reset Weights');

    folder.open();
  }

  _buildVizPanel() {
    const folder = this.gui.addFolder('Visualization');

    folder.add(this.visualizer, 'showParticles').name('Particles');

    // Bloom controls — expose the pass directly
    const bp = this.visualizer.bloomPass;
    folder.add(bp, 'strength', 0, 4, 0.05).name('Bloom Strength');
    folder.add(bp, 'radius', 0, 2, 0.05).name('Bloom Radius');
    folder.add(bp, 'threshold', 0, 1, 0.01).name('Bloom Threshold');

    folder.close();
  }

  // ─── STL Layout panel ────────────────────────────────────────────────────────

  addSTLPanel(layout) {
    this._stlLayout = layout;
    this._setupSTLCallbacks(layout);
    this._buildSTLPanel();
  }

  /** Wire layout callbacks — called once; references class fields for late binding. */
  _setupSTLCallbacks(layout) {
    layout.onProgress = (pct, label) => {
      if (this._stlState) {
        this._stlState.status = label;
        this._stlStatusCtrl?.updateDisplay();
      }
    };

    layout.onReady = (geometry) => {
      this.visualizer.showSTLMesh(geometry);
      layout.initNeurons(this.network);
      layout.start();
      if (this._stlState) {
        this._stlState.status = 'Ready';
        this._stlStatusCtrl?.updateDisplay();
      }
      this._stlStartCtrl?.enable();
      this._stlStopCtrl?.enable();
    };

    layout.onTick = (positions) => {
      this.visualizer.setPositions(positions);
    };
  }

  /** Build (or rebuild) the STL Layout GUI folder. */
  _buildSTLPanel() {
    const layout = this._stlLayout;

    // Persist status across GUI rebuilds
    if (!this._stlState) this._stlState = { status: 'No file loaded' };

    // Clean up old hidden file input from previous build
    if (this._stlFileInput) {
      document.body.removeChild(this._stlFileInput);
      this._stlFileInput = null;
    }

    const folder = this.gui.addFolder('STL Layout');
    const state  = this._stlState;

    // Hidden file input element
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.stl';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);
    this._stlFileInput = fileInput;

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      state.status = 'Loading…';
      this._stlStatusCtrl?.updateDisplay();
      layout.loadFile(file);
      fileInput.value = '';
    });

    folder.add({ load: () => fileInput.click() }, 'load').name('Load STL…');

    this._stlStatusCtrl = folder.add(state, 'status').name('Status').disable();

    this._stlStartCtrl = folder.add({ start: () => { if (layout.isLoaded) layout.start(); } }, 'start').name('▶ Start');
    this._stlStopCtrl  = folder.add({ stop:  () => layout.stop() },  'stop').name('■ Stop');

    if (!layout.isLoaded) {
      this._stlStartCtrl.disable();
      this._stlStopCtrl.disable();
    }

    folder.add(layout, 'kRepel',    0.1, 3,    0.01 ).name('Repulsion');
    folder.add(layout, 'kLayer',    0,   0.2,  0.002).name('Layer Guidance');
    folder.add(layout, 'kBoundary', 0,   2,    0.02 ).name('Boundary');
    folder.add(layout, 'damping',   0.5, 0.99, 0.01 ).name('Damping');

    folder.add({
      reset: () => {
        layout.stop();
        this.visualizer.hideSTLMesh();
        this.visualizer.buildFromNetwork(this.network);
      },
    }, 'reset').name('Reset to Grid');

    folder.open();
  }

  // ─── Internal ────────────────────────────────────────────────────────────────

  _apply() {
    const configs = this._layerConfigs.map(l => ({ ...l }));
    this.onReconfigure(configs);
    // Rebuild the whole GUI so layer folders reflect new topology
    this.gui.destroy();
    this.gui = new GUI({ title: 'NET·VIZ', width: 290 });
    this._build();
  }
}
