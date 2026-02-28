export class StimulusController {
  /**
   * @param {number} inputSize  - number of input neurons
   */
  constructor(inputSize) {
    this.mode = 'sine'; // 'sine' | 'noise' | 'manual'
    this.time = 0;
    this._setSize(inputSize);
  }

  get inputSize() { return this._inputSize; }

  setInputSize(size) {
    this._setSize(size);
  }

  _setSize(size) {
    this._inputSize = size;

    // Sine parameters — evenly spread in frequency/phase
    this.sineParams = Array.from({ length: size }, (_, i) => ({
      frequency: 0.3 + i * 0.25,
      amplitude: 1.0,
      phase: (i / Math.max(1, size)) * Math.PI * 2,
    }));

    // Manual values as plain object so lil-gui can bind to string keys
    // Only initialize if size changed (preserve existing values)
    const prev = this.manualValues ?? {};
    this.manualValues = {};
    for (let i = 0; i < size; i++) {
      this.manualValues[String(i)] = prev[String(i)] ?? 0.5;
    }

    // Noise state
    this._noiseState = new Float32Array(size).fill(0.5);
  }

  /**
   * Advance time by dt (call once per frame before getInputs).
   */
  update(dt) {
    this.time += dt;
  }

  /**
   * Returns current input values as number[].
   */
  getInputs() {
    const inputs = new Array(this._inputSize);

    switch (this.mode) {
      case 'sine':
        for (let i = 0; i < this._inputSize; i++) {
          const { frequency, amplitude, phase } = this.sineParams[i];
          inputs[i] = (Math.sin(this.time * frequency * Math.PI * 2 + phase) * amplitude + 1) * 0.5;
        }
        break;

      case 'noise':
        // Smooth random walk, bounded [0, 1]
        for (let i = 0; i < this._inputSize; i++) {
          this._noiseState[i] += (Math.random() - 0.5) * 0.12;
          this._noiseState[i] = Math.max(0, Math.min(1, this._noiseState[i]));
          inputs[i] = this._noiseState[i];
        }
        break;

      case 'manual':
        for (let i = 0; i < this._inputSize; i++) {
          inputs[i] = this.manualValues[String(i)] ?? 0.5;
        }
        break;

      default:
        inputs.fill(0);
    }

    return inputs;
  }
}
