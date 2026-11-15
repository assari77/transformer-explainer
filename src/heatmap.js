/**
 * Vector strip and attention heatmap renderers for the Transformer Explainer.
 * Uses the diverging Orange (-1) <-> White (0) <-> Blue (+1) palette from myTFPlayground
 * for real-valued embeddings/QKV/MLP activations, and a crisp sequential blue scale
 * for [0, 1] Softmax attention probabilities.
 */

const NEG_COLOR = [245, 147, 34];  // #f59322 (orange)
const ZERO_COLOR = [248, 250, 252]; // #f8fafc (near-white)
const POS_COLOR = [8, 119, 189];   // #0877bd (blue)

export function valueToDivergingRgb(val) {
  const clamped = Math.max(-1, Math.min(1, val));
  const t = Math.abs(clamped);
  const target = clamped < 0 ? NEG_COLOR : POS_COLOR;
  const r = Math.round(ZERO_COLOR[0] + (target[0] - ZERO_COLOR[0]) * t);
  const g = Math.round(ZERO_COLOR[1] + (target[1] - ZERO_COLOR[1]) * t);
  const b = Math.round(ZERO_COLOR[2] + (target[2] - ZERO_COLOR[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Renders a 1D numeric vector as a horizontal or vertical color-coded strip on a <canvas>.
 */
export function renderVectorStrip(canvas, values, options = {}) {
  if (!canvas || !values || values.length === 0) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const scale = options.scale || 1.0;
  const vertical = Boolean(options.vertical);

  ctx.clearRect(0, 0, w, h);
  const n = values.length;

  if (vertical) {
    const cellH = h / n;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = valueToDivergingRgb(values[i] / scale);
      ctx.fillRect(0, i * cellH, w, Math.ceil(cellH));
    }
  } else {
    const cellW = w / n;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = valueToDivergingRgb(values[i] / scale);
      ctx.fillRect(i * cellW, 0, Math.ceil(cellW), h);
    }
  }
}

/**
 * Maps an attention probability in [0, 1] to a clean sequential blue fill and readable text color.
 */
export function attentionProbStyle(prob) {
  const p = Math.max(0, Math.min(1, prob));
  // Interpolate from #f8fafc (0) to #0877bd (1)
  const r = Math.round(248 + (8 - 248) * p);
  const g = Math.round(250 + (119 - 250) * p);
  const b = Math.round(252 + (189 - 252) * p);
  const textColor = p > 0.48 ? "#ffffff" : "#1e293b";
  return {
    backgroundColor: `rgb(${r}, ${g}, ${b})`,
    color: textColor,
  };
}
