'use client';

import { useId, useState } from 'react';
import { ArrowRight, FlaskConical, RefreshCw } from 'lucide-react';
import { attentionExample, attentionMask, dropoutExample, normalizationExample, ppoExample, ropeExample, samplingExample } from './visualMath';
import s from './explanation-visuals.module.css';

export const VISUAL_KINDS = ['dropout', 'embedding', 'attention', 'attention-mask', 'normalization', 'rope', 'sampling', 'ppo'] as const;

function fmt(value: number) { return Math.abs(value) < 0.0005 ? '0' : Number(value.toFixed(3)).toString(); }

function Lab({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className={s.lab} aria-label={title}><div className={s.heading}><FlaskConical size={17} /><span>TRY THE IDEA</span><b>Interactive example</b></div><h3>{title}</h3>{children}</section>;
}

function Slider({ label, value, min, max, step = 1, onChange }: { label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void }) {
  const id = useId();
  return <label className={s.control} htmlFor={id}><span>{label}<output>{fmt(value)}</output></span><input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}

function Matrix({ values, label }: { values: number[][]; label: string }) {
  return <div className={s.matrixGroup}><span>{label}</span><div className={s.matrix} role="img" aria-label={`${label}: ${values.map(row => row.map(fmt).join(', ')).join('; ')}`} style={{ gridTemplateColumns: `repeat(${values[0].length}, minmax(42px, 1fr))` }}>{values.flatMap((row, i) => row.map((value, j) => <span key={`${i}-${j}`}>{fmt(value)}</span>))}</div></div>;
}

function DropoutLab() {
  const [p, setP] = useState(0.5);
  const [training, setTraining] = useState(true);
  const [draws, setDraws] = useState([0.12, 0.77, 0.43, 0.91, 0.6]);
  const input = [2, -4, 6, -8, 10];
  const { mask, output } = dropoutExample(input, draws, p, training);
  const active = training && p > 0;
  return <Lab title="Watch the mask gate each value">
    <div className={s.controls}><Slider label="Drop probability p" value={p} min={0} max={0.9} step={0.1} onChange={setP} /><label className={s.toggle}><input type="checkbox" checked={training} onChange={event => setTraining(event.target.checked)} />Training mode</label><button onClick={() => setDraws(input.map(() => Math.random()))}><RefreshCw size={14} />Draw a new mask</button></div>
    <div className={s.tableScroll}><table><thead><tr><th>Input x</th><th>Random draw</th><th>draw &gt; p</th><th>Output</th></tr></thead><tbody>{input.map((value, i) => <tr key={i}><td>{value}</td><td>{active ? fmt(draws[i]) : '—'}</td><td>{active ? <span className={mask[i] ? s.yes : s.no}>{mask[i] ? '1 · keep' : '0 · drop'}</span> : 'bypassed'}</td><td className={s.result}>{fmt(output[i])}</td></tr>)}</tbody></table></div>
    <p className={s.formula}>{!training || p === 0 ? 'Early return: output = x' : `output = x × mask ÷ ${fmt(1 - p)}`}</p>
    <p>{!training || p === 0 ? 'The reference skips the draw and mask steps in this mode. Every input passes through unchanged.' : `Each surviving value is multiplied by ${fmt(1 / (1 - p))}. Try moving p past one of the displayed draws: that value disappears. A fresh draw changes which values survive.`}</p>
  </Lab>;
}

function EmbeddingLab() {
  const [first, setFirst] = useState(1);
  const [second, setSecond] = useState(3);
  const weights = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]];
  return <Lab title="IDs select rows, even when an ID repeats">
    <div className={s.controls}><Slider label="First token ID" value={first} min={0} max={4} onChange={setFirst} /><Slider label="Second token ID" value={second} min={0} max={4} onChange={setSecond} /></div>
    <div className={s.tableScroll}><table><thead><tr><th>Row ID</th><th>Weight vector</th><th>Times selected</th><th>Gradient of output.sum()</th></tr></thead><tbody>{weights.map((row, i) => { const count = Number(i === first) + Number(i === second); return <tr key={i} className={count ? s.highlight : ''}><td>{i}</td><td>[{row.join(', ')}]</td><td>{count}</td><td>[{count}, {count}]</td></tr>; })}</tbody></table></div>
    <div className={s.flow}><Matrix values={[[first, second]]} label="indices · shape (1, 2)" /><ArrowRight size={20} /><Matrix values={[weights[first], weights[second]]} label="one batch's output · shape (2, 2)" /></div>
    <p>The full output has shape <strong>(1, 2, 2)</strong>: one batch, two token positions, two coordinates per vector. Choose the same ID twice. Its row appears twice, and its gradient receives two contributions; unused rows receive zero.</p>
  </Lab>;
}

function AttentionLab() {
  const [query, setQuery] = useState(1);
  const example = attentionExample(query);
  return <Lab title="A query turns scores into a weighted answer">
    <Slider label="First query coordinate q₀" value={query} min={-3} max={3} step={0.1} onChange={setQuery} />
    <p>One query <strong>[{fmt(query)}, 1]</strong>, three keys, and three scalar values. The feature width is 2, so each dot product is divided by √2.</p>
    <div className={s.tableScroll}><table><thead><tr><th>Key</th><th>Scaled score</th><th>Softmax weight</th><th>Value</th><th>Contribution</th></tr></thead><tbody>{example.keys.map((key, i) => <tr key={i}><td>[{key.join(', ')}]</td><td>{fmt(example.scores[i])}</td><td><div className={s.bar}><i style={{ width: `${example.weights[i] * 100}%` }} /><span>{fmt(example.weights[i])}</span></div></td><td>{example.values[i]}</td><td>{fmt(example.contributions[i])}</td></tr>)}</tbody></table></div>
    <p className={s.formula}>Output = sum(weight × value) = <strong>{fmt(example.output)}</strong></p>
    <p>Move the query left or right. The keys stay fixed, but the dot products change; softmax changes their shares of the answer. Weights sum to 1, so the answer stays between the smallest and largest value.</p>
  </Lab>;
}

function MaskLab({ taskId }: { taskId: string }) {
  const causal = taskId === 'causal_attention';
  const [row, setRow] = useState(2);
  const [window, setWindow] = useState(1);
  const allowed = attentionMask(causal, window);
  const count = allowed[row].filter(Boolean).length;
  return <Lab title={causal ? 'Future tokens are blocked before softmax' : 'The window permits nearby tokens in both directions'}>
    <div className={s.controls}><Slider label="Query position" value={row} min={0} max={5} onChange={setRow} />{!causal && <Slider label="Window radius" value={window} min={0} max={5} onChange={setWindow} />}</div>
    <div className={s.maskGrid} role="img" aria-label={`${causal ? 'Causal' : 'Sliding window'} mask. Query ${row} can attend to keys ${allowed[row].map((yes, i) => yes ? i : null).filter(i => i !== null).join(', ')}.`}><span>Q ↓ / K →</span>{allowed.map((_, j) => <b key={`header-${j}`}>{j}</b>)}{allowed.flatMap((cells, i) => [<b key={`row-${i}`}>{i}</b>, ...cells.map((yes, j) => <span key={`${i}-${j}`} className={`${yes ? s.allowed : s.blocked} ${i === row ? s.selectedRow : ''}`} title={`Query ${i}, key ${j}: ${yes ? 'allowed · mask=False' : 'blocked · mask=True'}`}>{yes ? '✓' : '×'}</span>)])}</div>
    <div className={s.legend}><span><i className={s.allowed} />Allowed · mask=False</span><span><i className={s.blocked} />Blocked · mask=True</span></div>
    <p>Suppose every raw score is zero. Blocked scores become <strong>−∞</strong>, so their softmax weights become zero. The {count} allowed keys in row {row} each receive weight <strong>{fmt(1 / count)}</strong>.</p>
    <div className={s.tokenStrip}>{allowed[row].map((yes, j) => <span key={j} className={yes ? s.allowed : s.blocked}>K{j}<b>{yes ? fmt(1 / count) : '0'}</b></span>)}</div>
    <p>{causal ? 'The diagonal remains available: a token may attend to itself and earlier tokens.' : 'The test is abs(query − key) > radius. A nearby future token is allowed too; this reference window is not a causal mask.'}</p>
  </Lab>;
}

function NormalizationLab({ taskId }: { taskId: string }) {
  const [first, setFirst] = useState(2);
  const [gamma, setGamma] = useState(1);
  const [beta, setBeta] = useState(0);
  const example = normalizationExample(taskId, first, gamma, beta);
  const rms = taskId === 'rmsnorm';
  const batch = taskId === 'batchnorm';
  return <Lab title={batch ? 'Normalize each feature across the batch' : rms ? 'Divide by row RMS without subtracting a mean' : 'Normalize each row across its features'}>
    <div className={s.controls}><Slider label="First input value" value={first} min={-6} max={8} onChange={setFirst} /><Slider label={rms ? 'Weight' : 'Scale γ'} value={gamma} min={0.5} max={2} step={0.25} onChange={setGamma} />{!rms && <Slider label="Shift β" value={beta} min={-2} max={2} step={0.5} onChange={setBeta} />}</div>
    <div className={s.flow}><Matrix values={example.input} label={`Input · ${batch ? 'three' : 'two'} rows, three features`} /><ArrowRight size={20} /><Matrix values={example.output} label="Output · same shape" /></div>
    <div className={s.tableScroll}><table><thead><tr><th>{batch ? 'Feature column' : 'Row'}</th>{!rms && <><th>Mean</th><th>Variance</th></>}<th>{rms ? 'RMS = √(mean(x²) + ε)' : '√(variance + ε)'}</th></tr></thead><tbody>{example.means.map((mean, i) => <tr key={i}><td>{i}</td>{!rms && <><td>{fmt(mean)}</td><td>{fmt(example.variances[i])}</td></>}<td>{fmt(example.scales[i])}</td></tr>)}</tbody></table></div>
    <p className={s.formula}>{rms ? 'Output = input ÷ RMS × weight' : 'Output = (input − mean) ÷ √(variance + ε) × γ + β'}</p>
    <p>{batch ? 'This is training mode: statistics come from the three rows in each column. Changing one input affects every row in that column. The reference also updates running statistics for later evaluation.' : rms ? 'Changing one coordinate changes its row’s RMS and therefore scales every coordinate in that row. Values do not become zero-centered.' : 'Changing one coordinate changes its row’s mean and variance. The second row stays independent. γ and β then adjust each normalized value.'}</p>
  </Lab>;
}

function RopeLab() {
  const [position, setPosition] = useState(0);
  const example = ropeExample(position);
  const id = useId().replaceAll(':', '');
  return <Lab title="Each pair rotates at its own frequency">
    <Slider label="Token position" value={position} min={0} max={12} onChange={setPosition} />
    <p>Use a four-coordinate vector <strong>[1, 0, 1, 0]</strong>. Its two pairs start on the positive x-axis. Position zero leaves both pairs unchanged.</p>
    <div className={s.circles}>{example.pairs.map((pair, i) => <div key={i}><svg viewBox="0 0 160 160" role="img" aria-label={`Pair ${i}: angle ${fmt(example.angles[i])} radians, output [${pair.map(fmt).join(', ')}]`}><defs><marker id={`${id}-${i}`} markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill={i === 0 ? '#4c7a3a' : '#996a23'} /></marker></defs><circle cx="80" cy="80" r="57" fill="none" stroke="#dce7cf" /><path d="M15,80 H145 M80,15 V145" stroke="#dce7cf" /><path d="M80,80 H137" stroke="#aebca4" strokeDasharray="4 4" /><path d={`M80,80 L${80 + 57 * pair[0]},${80 - 57 * pair[1]}`} stroke={i === 0 ? '#4c7a3a' : '#996a23'} strokeWidth="3" markerEnd={`url(#${id}-${i})`} /><circle cx={80 + 57 * pair[0]} cy={80 - 57 * pair[1]} r="3" fill={i === 0 ? '#4c7a3a' : '#996a23'} /></svg><b>Pair {i} · frequency {example.frequencies[i]}</b><span>Angle {fmt(example.angles[i])} rad · [{pair.map(fmt).join(', ')}]</span></div>)}</div>
    <p className={s.formula}>Rotated vector = [{example.output.map(fmt).join(', ')}]</p>
    <p>The first pair turns quickly; the second turns 100 times more slowly. The code interleaves each pair’s rotated x/y coordinates back into one vector and applies the same operation to queries and keys.</p>
  </Lab>;
}

function SamplingLab() {
  const [topK, setTopK] = useState(0);
  const [topP, setTopP] = useState(0.6);
  const [temperature, setTemperature] = useState(1);
  const [sample, setSample] = useState<number | null>(null);
  const example = samplingExample(topK, topP, temperature);
  function update(setter: (value: number) => void, value: number) { setter(value); setSample(null); }
  function draw(threshold: number) { let total = 0; const token = example.final.findIndex(p => { total += p; return threshold < total; }); setSample(token < 0 ? example.final.length - 1 : token); }
  return <Lab title="See which tokens survive top-k and top-p">
    <p>Start with token probabilities [0.5, 0.25, 0.15, 0.1], represented by their log probabilities as logits.</p>
    <div className={s.controls}><Slider label="Top-k (0 means off)" value={topK} min={0} max={4} onChange={value => update(setTopK, value)} /><Slider label="Top-p" value={topP} min={0.05} max={1} step={0.05} onChange={value => update(setTopP, value)} /><Slider label="Temperature" value={temperature} min={0.2} max={2} step={0.1} onChange={value => update(setTemperature, value)} /></div>
    <div className={s.tableScroll}><table><thead><tr><th>Token</th><th>After top-k</th><th>Mass before token</th><th>Kept?</th><th>Final chance</th></tr></thead><tbody>{example.base.map((_, i) => <tr key={i} className={example.kept[i] ? s.highlight : ''}><td>{i}</td><td>{fmt(example.probabilities[i])}</td><td>{fmt(example.prefixMass[i])}</td><td>{example.kept[i] ? 'Yes' : 'No'}</td><td><div className={s.bar}><i style={{ width: `${example.final[i] * 100}%` }} /><span>{fmt(example.final[i])}</span></div></td></tr>)}</tbody></table></div>
    <p>Top-p tests the probability mass <strong>before</strong> each token. A token that crosses the cutoff is retained; later tokens are removed when their prefix mass is greater than top-p. Surviving probabilities are renormalized before sampling.</p>
    <div className={s.sampleRow}><button onClick={() => draw(Math.random())}>Sample a token<ArrowRight size={14} /></button><output aria-live="polite">{sample === null ? 'Only a surviving token can be drawn.' : `Sampled token ${sample}`}</output></div>
  </Lab>;
}

function PpoLab() {
  const [ratio, setRatio] = useState(1.4);
  const [advantage, setAdvantage] = useState(1);
  const [epsilon, setEpsilon] = useState(0.2);
  const result = ppoExample(ratio, advantage, epsilon);
  const x = (r: number) => 30 + r / 2.2 * 340;
  const y = (value: number) => 115 - value / 2.4 * 80;
  const points = Array.from({ length: 89 }, (_, i) => i / 40);
  const objective = points.map((r, i) => `${i ? 'L' : 'M'}${x(r)},${y(ppoExample(r, advantage, epsilon).objective)}`).join(' ');
  const unclipped = points.map((r, i) => `${i ? 'L' : 'M'}${x(r)},${y(r * advantage)}`).join(' ');
  const capped = advantage >= 0 ? ratio > 1 + epsilon : ratio < 1 - epsilon;
  return <Lab title="The advantage sign decides which change gets capped">
    <div className={s.controls}><Slider label="Probability ratio r" value={ratio} min={0.2} max={2} step={0.05} onChange={setRatio} /><label className={s.control}><span>Advantage A</span><select aria-label="Advantage A" value={advantage} onChange={event => setAdvantage(Number(event.target.value))}><option value={1}>+1 · better than expected</option><option value={-1}>−1 · worse than expected</option></select></label><Slider label="Clip range ε" value={epsilon} min={0} max={0.5} step={0.05} onChange={setEpsilon} /></div>
    <svg className={s.objectiveChart} viewBox="0 0 400 220" role="img" aria-label={`PPO objective versus ratio. Ratio ${fmt(ratio)}, advantage ${advantage}, objective ${fmt(result.objective)}.`}><rect x={x(1 - epsilon)} y="15" width={x(1 + epsilon) - x(1 - epsilon)} height="188" rx="5" fill="#e6efd8" /><path d="M30,115 H375 M30,15 V203" stroke="#c5d5b5" /><path d={unclipped} fill="none" stroke="#929f86" strokeWidth="2" strokeDasharray="5 4" /><path d={objective} fill="none" stroke="#4c7a3a" strokeWidth="3" /><circle cx={x(ratio)} cy={y(result.objective)} r="5" fill="#b68434" stroke="#fcfef8" strokeWidth="2" /><text x="30" y="217">0</text><text x={x(1) - 3} y="217">1</text><text x="320" y="217">ratio r →</text><text x="38" y="27">objective</text></svg>
    <div className={s.legend}><span>Solid: min(unclipped, clipped)</span><span>Dashed: r × A</span><span><i className={s.allowed} />Band: 1 − ε to 1 + ε</span></div>
    <div className={s.tableScroll}><table><thead><tr><th>r × A</th><th>clamp(r) × A</th><th>Chosen minimum</th><th>One-sample loss</th></tr></thead><tbody><tr><td>{fmt(result.unclipped)}</td><td>{fmt(result.clipped)}</td><td>{fmt(result.objective)}</td><td className={s.result}>{fmt(result.loss)}</td></tr></tbody></table></div>
    <p>{capped ? 'The objective is on its flat section: this change would improve the objective too far, so clipping limits that improvement.' : 'The objective follows r × A here. Clipping does not remove the penalty for changing the policy in the unfavorable direction.'} The reference negates the mean of these chosen objectives across the batch.</p>
  </Lab>;
}

export default function ExplanationVisual({ kind, taskId }: { kind: string; taskId: string }) {
  switch (kind) {
    case 'dropout': return <DropoutLab />;
    case 'embedding': return <EmbeddingLab />;
    case 'attention': return <AttentionLab />;
    case 'attention-mask': return <MaskLab taskId={taskId} />;
    case 'normalization': return <NormalizationLab taskId={taskId} />;
    case 'rope': return <RopeLab />;
    case 'sampling': return <SamplingLab />;
    case 'ppo': return <PpoLab />;
    default: return null;
  }
}
