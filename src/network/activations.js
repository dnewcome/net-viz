export const ACTIVATIONS = {
  sigmoid: {
    fn: x => 1 / (1 + Math.exp(-x)),
    // Maps output to [0, 1] — no normalization needed
    normalize: v => v,
    label: 'Sigmoid',
  },
  relu: {
    fn: x => Math.max(0, x),
    // Clamp to [0, 1] for visualization
    normalize: v => Math.min(1, v),
    label: 'ReLU',
  },
  tanh: {
    fn: x => Math.tanh(x),
    // Maps [-1, 1] → [0, 1]
    normalize: v => (v + 1) * 0.5,
    label: 'Tanh',
  },
  linear: {
    fn: x => x,
    // Soft clamp centered at 0 for visualization
    normalize: v => Math.min(1, Math.max(0, (v + 2) / 4)),
    label: 'Linear',
  },
};

export function normalizeForViz(value, activationName) {
  const act = ACTIVATIONS[activationName];
  return act ? Math.max(0, Math.min(1, act.normalize(value))) : Math.max(0, Math.min(1, value));
}
