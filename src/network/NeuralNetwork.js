import { ACTIVATIONS } from './activations.js';

export class NeuralNetwork {
  /**
   * @param {Array<{size: number, activation: string}>} layerConfigs
   */
  constructor(layerConfigs) {
    this.configure(layerConfigs);
  }

  configure(layerConfigs) {
    this.layerConfigs = layerConfigs.map(l => ({
      size: Math.max(1, l.size | 0),
      activation: l.activation || 'sigmoid',
    }));
    this.numLayers = this.layerConfigs.length;
    this._initWeightsAndState();
  }

  _initWeightsAndState() {
    this.activations = this.layerConfigs.map(l => new Float32Array(l.size).fill(0));
    this.weights = [];
    this.biases = [];
    this.eligibility = [];

    for (let l = 0; l < this.numLayers - 1; l++) {
      const inN = this.layerConfigs[l].size;
      const outN = this.layerConfigs[l + 1].size;
      // Xavier uniform initialization
      const limit = Math.sqrt(6.0 / (inN + outN));
      const w = new Float32Array(inN * outN);
      for (let k = 0; k < w.length; k++) {
        w[k] = (Math.random() * 2 - 1) * limit;
      }
      this.weights.push(w);
      this.biases.push(new Float32Array(outN).fill(0));
      this.eligibility.push(new Float32Array(inN * outN).fill(0));
    }
  }

  /**
   * Run a forward pass. Stores activations internally.
   * @param {number[]} inputs
   * @returns {Float32Array[]} activations per layer
   */
  forward(inputs) {
    // Set input layer
    const inp = this.activations[0];
    for (let i = 0; i < Math.min(inputs.length, inp.length); i++) {
      inp[i] = inputs[i];
    }

    for (let l = 0; l < this.numLayers - 1; l++) {
      const inN = this.layerConfigs[l].size;
      const outN = this.layerConfigs[l + 1].size;
      const actFn = ACTIVATIONS[this.layerConfigs[l + 1].activation]?.fn ?? ACTIVATIONS.sigmoid.fn;
      const w = this.weights[l];
      const b = this.biases[l];
      const pre = this.activations[l];
      const post = this.activations[l + 1];
      const elig = this.eligibility[l];

      for (let j = 0; j < outN; j++) {
        let sum = b[j];
        for (let i = 0; i < inN; i++) {
          sum += pre[i] * w[i * outN + j];
        }
        post[j] = actFn(sum);
      }

      // Eligibility trace: Hebbian correlation, decaying over time
      const decay = 0.92;
      for (let i = 0; i < inN; i++) {
        for (let j = 0; j < outN; j++) {
          elig[i * outN + j] = decay * elig[i * outN + j] + pre[i] * post[j];
        }
      }
    }

    return this.activations;
  }

  /**
   * Apply a reinforcement signal using eligibility traces.
   * Positive reward strengthens recently active connections,
   * negative reward weakens them.
   * @param {number} reward  - typically +1 or -1
   * @param {number} lr      - learning rate
   */
  reinforce(reward, lr = 0.01) {
    for (let l = 0; l < this.weights.length; l++) {
      const w = this.weights[l];
      const elig = this.eligibility[l];
      for (let k = 0; k < w.length; k++) {
        w[k] += lr * reward * elig[k];
        // Soft clamp to prevent runaway weights
        w[k] = Math.max(-4, Math.min(4, w[k]));
      }
    }
  }

  /**
   * Read a single weight value.
   * @param {number} layerIdx - index of the weight matrix (0 = input→hidden1)
   * @param {number} from     - pre-synaptic neuron index
   * @param {number} to       - post-synaptic neuron index
   */
  getWeight(layerIdx, from, to) {
    const outN = this.layerConfigs[layerIdx + 1].size;
    return this.weights[layerIdx][from * outN + to];
  }

  getTotalNeurons() {
    return this.layerConfigs.reduce((s, l) => s + l.size, 0);
  }

  getTotalConnections() {
    let total = 0;
    for (let l = 0; l < this.numLayers - 1; l++) {
      total += this.layerConfigs[l].size * this.layerConfigs[l + 1].size;
    }
    return total;
  }
}
