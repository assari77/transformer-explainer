/**
 * Streamlined Pure-JS Transformer Engine.
 * Breaks any partial sentence into token + positional embeddings, applies a configurable
 * Context Window limit (so earlier tokens outside the window are forgotten), runs a Transformer
 * block forward pass, and computes temperature-scaled Softmax probabilities:
 *   P(token_i) = exp(logit_i / T) / sum_j exp(logit_j / T)
 */

import { predictContextualCandidates, extractContextEntities } from "./lm.js";

export const D_MODEL = 8;

/**
 * Splits a partial sentence string into word tokens.
 */
export function tokenize(text) {
  const cleaned = String(text || "").trim();
  if (!cleaned) return ["The"];
  return cleaned.split(/\s+/);
}

/**
 * Numerically stable Softmax with Temperature scaling on [0.0, 1.0].
 * - T = 0.0: Greedy argmax (100% on the #1 token, 0% on all others)
 * - T = 0.5: Balanced Softmax distribution
 * - T = 1.0: Creative / flatter distribution
 */
export function softmax(logits, temperature = 0.5) {
  const rawT = Number(temperature);
  let maxLogit = -Infinity;
  let maxIdx = 0;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
      maxIdx = i;
    }
  }
  if (!Number.isFinite(maxLogit)) {
    return logits.map(() => 0);
  }
  if (!Number.isNaN(rawT) && rawT <= 0.02) {
    return logits.map((_, i) => (i === maxIdx ? 1 : 0));
  }

  // Map UI temperature (0.1 .. 1.0) onto an expressive Softmax divisor (0.18 .. 1.80)
  const effectiveT = Math.max(0.05, (Number.isNaN(rawT) ? 0.5 : rawT) * 1.8);
  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const val = Math.exp((logits[i] - maxLogit) / effectiveT);
    exps[i] = val;
    sum += val;
  }
  return exps.map((v) => (sum > 0 ? v / sum : 0));
}

export function layerNorm(vec, eps = 1e-5) {
  const n = vec.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += vec[i];
  mean /= n;

  let variance = 0;
  for (let i = 0; i < n; i++) {
    const diff = vec[i] - mean;
    variance += diff * diff;
  }
  variance /= n;
  const invStd = 1 / Math.sqrt(variance + eps);
  return vec.map((x) => (x - mean) * invStd);
}

export function getPositionalEncoding(pos, dModel = D_MODEL) {
  const pe = new Array(dModel);
  for (let i = 0; i < dModel; i++) {
    const divTerm = Math.pow(100, (2 * Math.floor(i / 2)) / dModel);
    const angle = (pos + 1) / divTerm;
    pe[i] = (i % 2 === 0 ? Math.sin(angle) : Math.cos(angle)) * 0.35;
  }
  return pe;
}

/**
 * Deterministic 8D semantic embedding for any word token.
 */
export function getTokenEmbedding(token) {
  const s = String(token || "").toLowerCase();
  const raw = new Array(D_MODEL).fill(0);
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    for (let d = 0; d < D_MODEL; d++) {
      raw[d] += Math.sin(code * (d + 1) * 0.37 + i * 0.73);
    }
  }
  return layerNorm(raw).map((v) => v * 0.65);
}

export function dotProduct(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function addVectors(a, b, scaleB = 1.0) {
  return a.map((v, i) => v + b[i] * scaleB);
}

/**
 * Samples or selects a token from `topPredictions` according to `temperature` in [0.0, 1.0].
 */
export function selectNextToken(
  topPredictions,
  temperature = 0.5,
  rng = Math.random,
  forceSample = false
) {
  if (!topPredictions || topPredictions.length === 0) return null;
  if (temperature <= 0.02 || (!forceSample && temperature <= 0.6)) {
    return topPredictions[0];
  }
  const totalProb = topPredictions.reduce((acc, item) => acc + item.prob, 0);
  if (totalProb <= 0) return topPredictions[0];

  let r = rng() * totalProb;
  for (const item of topPredictions) {
    r -= item.prob;
    if (r <= 0) return item;
  }
  return topPredictions[0];
}

/**
 * Runs the Transformer forward pass for `tokens` at the given `temperature` and `contextWindow`.
 * Only the last `contextWindow` tokens are visible to Self-Attention and the Language Model;
 * earlier tokens are marked as out-of-context (forgotten).
 */
export function forwardTransformer(
  tokens,
  temperature = 0.5,
  rng = Math.random,
  contextWindow = 10,
  externalCandidates = null
) {
  const allTokens = Array.isArray(tokens) && tokens.length > 0 ? tokens : ["The"];
  const maxWin = Math.max(1, Math.floor(Number(contextWindow) || 10));
  const contextStartIndex = Math.max(0, allTokens.length - maxWin);
  const activeTokens = allTokens.slice(contextStartIndex);
  const seqLen = activeTokens.length;

  // Compute full-sentence embeddings so Block 1 can show both active and forgotten tokens
  const allTokenEmbeddings = allTokens.map((t) => getTokenEmbedding(t));
  const allPosEncodings = allTokens.map((_, pos) => getPositionalEncoding(pos, D_MODEL));
  const allInputEmbeddings = allTokenEmbeddings.map((te, i) =>
    addVectors(te, allPosEncodings[i], 1.0)
  );

  // Active window embeddings (what the Transformer actually sees)
  const tokenEmbeddings = activeTokens.map((t) => getTokenEmbedding(t));
  const posEncodings = activeTokens.map((_, pos) => getPositionalEncoding(pos, D_MODEL));
  const inputEmbeddings = tokenEmbeddings.map((te, i) => addVectors(te, posEncodings[i], 1.0));

  // 2. Transformer Block on `activeTokens` only
  const norm1 = inputEmbeddings.map((v) => layerNorm(v));
  const scale = Math.sqrt(D_MODEL);
  const { tokenIndices: entityIndices } = extractContextEntities(activeTokens);
  const entityIdxSet = new Set(entityIndices);

  const attnWeightsLastToken = new Array(seqLen).fill(0);
  const lastQuery = norm1[seqLen - 1];
  const rawAttnScores = norm1.map((kVec, j) => {
    let score = dotProduct(lastQuery, kVec) / scale;
    if (j === seqLen - 1) score += 0.35;
    if (entityIdxSet.has(j)) score += 1.25; // Self-attention focuses strongly on in-window entities
    return score;
  });
  const lastAttnProbs = softmax(rawAttnScores, 0.5);
  for (let j = 0; j < seqLen; j++) {
    attnWeightsLastToken[j] = lastAttnProbs[j];
  }

  // Context vector for the final token position
  const attnContext = new Array(D_MODEL).fill(0);
  for (let j = 0; j < seqLen; j++) {
    for (let d = 0; d < D_MODEL; d++) {
      attnContext[d] += lastAttnProbs[j] * norm1[j][d];
    }
  }
  const residual1 = addVectors(inputEmbeddings[seqLen - 1], attnContext, 0.6);
  const norm2 = layerNorm(residual1);
  const mlpOut = norm2.map((x, idx) => {
    const c = Math.sqrt(2 / Math.PI);
    const geluVal = 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    return geluVal * 0.7 + Math.sin((idx + 1) * x) * 0.2;
  });
  const finalHiddenState = layerNorm(addVectors(residual1, mlpOut, 0.5));

  // 3. Language Model Head: Uses real GPT-2 candidates when available, or built-in contextual LM
  const candidateEntries =
    Array.isArray(externalCandidates) && externalCandidates.length > 0
      ? externalCandidates
      : predictContextualCandidates(activeTokens);
  const rawLogits = candidateEntries.map((entry) => {
    const candEmb = getTokenEmbedding(entry.token);
    const smallAlignment = dotProduct(finalHiddenState, candEmb) * 0.04;
    return entry.baseLogit + smallAlignment;
  });

  // Add a small background tail mass representing the rest of the vocabulary
  const tailLogits = [-0.8, -1.1, -1.4, -1.7];
  const allLogits = [...rawLogits, ...tailLogits];
  const allProbs = softmax(allLogits, temperature);

  const topPredictions = candidateEntries
    .map((entry, idx) => ({
      token: entry.token,
      logit: rawLogits[idx],
      prob: allProbs[idx],
    }))
    .sort((a, b) => b.prob - a.prob);

  const selected = selectNextToken(topPredictions, temperature, rng, false);

  return {
    allTokens,
    allInputEmbeddings,
    contextWindow: maxWin,
    contextStartIndex,
    droppedCount: contextStartIndex,
    tokens: activeTokens,
    temperature,
    tokenEmbeddings,
    posEncodings,
    inputEmbeddings,
    attnWeightsLastToken,
    finalHiddenState,
    rawLogits,
    probabilities: topPredictions.map((p) => p.prob),
    topPredictions,
    selectedToken: selected ? selected.token : topPredictions[0].token,
    selectedProb: selected ? selected.prob : topPredictions[0].prob,
  };
}
