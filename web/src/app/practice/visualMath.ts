/** Small, deterministic calculations used by the worked-example visuals. */
export function softmax(values: number[]) {
  const max = Math.max(...values);
  const exps = values.map(value => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map(value => value / total);
}

export function dropoutExample(values: number[], draws: number[], p: number, training: boolean) {
  const mask = draws.map(draw => Number(draw > p));
  return { mask, output: values.map((value, i) => !training || p === 0 ? value : value * mask[i] / (1 - p)) };
}

export function attentionExample(query: number) {
  const keys = [[-1, 0], [0, 1], [1, 0]];
  const values = [10, 20, 40];
  const scores = keys.map(key => (query * key[0] + key[1]) / Math.sqrt(2));
  const weights = softmax(scores);
  const contributions = weights.map((weight, i) => weight * values[i]);
  return { keys, values, scores, weights, contributions, output: contributions.reduce((sum, value) => sum + value, 0) };
}

export function attentionMask(causal: boolean, window: number, length = 6) {
  return Array.from({ length }, (_, i) => Array.from({ length }, (_, j) => causal ? j <= i : Math.abs(i - j) <= window));
}

export function normalizationExample(task: string, first: number, gamma: number, beta: number) {
  const batch = task === 'batchnorm';
  const rms = task === 'rmsnorm';
  const input = [[first, 4, 6], [1, 3, 5], ...(batch ? [[6, 2, 4]] : [])];
  const groups = batch ? [0, 1, 2].map(j => input.map(row => row[j])) : input;
  const means = groups.map(group => group.reduce((sum, x) => sum + x, 0) / group.length);
  const variances = groups.map((group, i) => group.reduce((sum, x) => sum + (x - means[i]) ** 2, 0) / group.length);
  const scales = groups.map((group, i) => Math.sqrt((rms ? group.reduce((sum, x) => sum + x * x, 0) / group.length : variances[i]) + (rms ? 1e-6 : 1e-5)));
  const normalized = input.map((row, i) => row.map((x, j) => (x - (rms ? 0 : means[batch ? j : i])) / scales[batch ? j : i]));
  const output = normalized.map(row => row.map(x => gamma * x + (rms ? 0 : beta)));
  return { input, means, variances, scales, normalized, output };
}

export function ropeExample(position: number) {
  const frequencies = [1, 0.01]; // D=4: 10000 ** (-[0, 2]/4)
  const angles = frequencies.map(frequency => position * frequency);
  const pairs = angles.map(angle => [Math.cos(angle), Math.sin(angle)]);
  return { frequencies, angles, pairs, output: pairs.flat() };
}

export function samplingExample(topK: number, topP: number, temperature: number) {
  const base = [0.5, 0.25, 0.15, 0.1];
  const logits = base.map(p => Math.log(p) / Math.max(temperature, 1e-8));
  const sorted = [...logits].sort((a, b) => b - a);
  const threshold = topK > 0 ? sorted[topK - 1] : -Infinity;
  const afterK = logits.map(logit => logit < threshold ? -Infinity : logit);
  const probabilities = softmax(afterK);
  // Inputs stay in descending order, so prefix mass implements cumsum - probs.
  let prefix = 0;
  const prefixMass = probabilities.map(p => { const before = prefix; prefix += p; return before; });
  const kept = afterK.map((logit, i) => Number.isFinite(logit) && (topP >= 1 || prefixMass[i] <= topP));
  const final = softmax(afterK.map((logit, i) => kept[i] ? logit : -Infinity));
  return { base, probabilities, prefixMass, kept, final };
}

export function ppoExample(ratio: number, advantage: number, epsilon: number) {
  const clampedRatio = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio));
  const unclipped = ratio * advantage;
  const clipped = clampedRatio * advantage;
  const objective = Math.min(unclipped, clipped);
  return { clampedRatio, unclipped, clipped, objective, loss: -objective };
}
