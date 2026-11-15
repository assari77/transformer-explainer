/**
 * Real In-Browser GPT-2 Engine powered by Hugging Face Transformers.js (Xenova/distilgpt2).
 * Loads quantized DistilGPT-2 (82M parameters) directly in the browser via WebAssembly,
 * caches weights in IndexedDB/CacheStorage, and extracts real next-word logits over the
 * 50,257-token vocabulary for the active Context Window with standard LLM Repetition Penalty
 * and No-Repeat N-gram blocking to prevent greedy repetition loops.
 */

import { extractContextEntities } from "./lm.js";

const MODEL_ID = "Xenova/distilgpt2";
const CDN_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

let tokenizer = null;
let model = null;
let isLoading = false;
let isReady = false;
let loadError = null;

// Cache recent active-window queries so dragging the Temperature slider is 0ms instant
const logitCache = new Map();

// Common function words allowed to repeat naturally across clauses
const ALLOWED_FUNCTION_REPEATS = new Set([
  "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "from", "by",
  "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "our", "their",
  "is", "was", "are", "were", "be", "had", "have", "has", "that", "this", "as",
]);

export function getGpt2Status() {
  return { isLoading, isReady, loadError };
}

/**
 * Dynamically imports Transformers.js from CDN and loads Xenova/distilgpt2.
 * Calls `onStatus(statusText, readyBoolean)` as download/initialization progresses.
 */
export async function initGpt2(onStatus) {
  if (isReady || isLoading) return;
  if (typeof window === "undefined") return;

  isLoading = true;
  if (onStatus) onStatus("Loading GPT-2 model…", false);

  try {
    const { AutoTokenizer, AutoModelForCausalLM, env } = await import(CDN_URL);
    env.allowLocalModels = false;

    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: "q8",
      progress_callback: (info) => {
        if (info.status === "progress" && typeof info.progress === "number" && onStatus) {
          onStatus(`Loading GPT-2 (${Math.round(info.progress)}%)…`, false);
        }
      },
    });

    isReady = true;
    isLoading = false;
    if (onStatus) onStatus("GPT-2 Ready (82M)", true);
  } catch (err) {
    console.warn("GPT-2 browser load fallback:", err);
    loadError = err?.message || "Failed to load GPT-2";
    isLoading = false;
    if (onStatus) onStatus("Built-in LM Active", false);
  }
}

/**
 * Given the active in-window tokens, runs a real GPT-2 forward pass and returns
 * the top 6 clean English word candidates `{ token, baseLogit }[]`.
 */
export async function predictGpt2Candidates(activeTokens) {
  if (!isReady || !tokenizer || !model) return null;

  const activeText = activeTokens.join(" ").trim();
  if (!activeText) return null;

  if (logitCache.has(activeText)) {
    return logitCache.get(activeText);
  }

  const inputs = await tokenizer(activeText);
  const outputs = await model(inputs);
  const { logits } = outputs;
  const [, seqLen, vocabSize] = logits.dims;
  const offset = (seqLen - 1) * vocabSize;
  const lastLogits = logits.data.subarray(offset, offset + vocabSize);

  // Find the top 140 raw token IDs in O(V) time
  const topK = 140;
  const topIds = [];
  const topVals = [];

  for (let id = 0; id < vocabSize; id++) {
    const val = lastLogits[id];
    if (topIds.length < topK) {
      topIds.push(id);
      topVals.push(val);
      if (topIds.length === topK) {
        sortPairsDesc(topIds, topVals);
      }
    } else if (val > topVals[topK - 1]) {
      topIds[topK - 1] = id;
      topVals[topK - 1] = val;
      sortPairsDesc(topIds, topVals);
    }
  }

  const cleanHistory = activeTokens.map((t) => t.toLowerCase().replace(/[^a-z]/g, ""));
  const n = cleanHistory.length;
  const lastWordLower = n > 0 ? cleanHistory[n - 1] : "";
  const prevWordLower = n > 1 ? cleanHistory[n - 2] : "";

  // Build a set of existing 2-grams ("prevWord -> nextWord") in the sentence to block 2-gram loops
  // e.g. if "the middle" already appeared, block "middle" after "the"!
  const existingBigramsFromLast = new Set();
  for (let i = 0; i < n - 1; i++) {
    if (cleanHistory[i] === lastWordLower) {
      existingBigramsFromLast.add(cleanHistory[i + 1]);
    }
  }

  // Count how many times each word has already appeared in the recent window
  const wordCounts = new Map();
  for (const w of cleanHistory) {
    if (!w) continue;
    wordCounts.set(w, (wordCounts.get(w) || 0) + 1);
  }

  // Check if any context entities were explicitly introduced (e.g. "I have a parrot ... feed my")
  // Only boost on possessive endings ("my", "your", "his", "her", "our", "their"), NOT "the"
  const { entities: contextEntities } = extractContextEntities(activeTokens);
  const isPossessiveEnd = new Set(["my", "your", "his", "her", "our", "their"]).has(lastWordLower);

  const candidates = [];
  const seenWords = new Set();

  for (let i = 0; i < topIds.length; i++) {
    const decoded = tokenizer.decode([topIds[i]]);
    // In GPT-2 BPE, whole words following a space begin with a leading space " "
    if (!/^\s+[a-zA-Z]{2,}$/.test(decoded) && decoded !== " I" && decoded !== " a") {
      continue;
    }
    const word = decoded.trim();
    const lower = word.toLowerCase();

    // 1. Block immediate repetition ("word word") or alternating ("word X word")
    if (lower === lastWordLower || lower === prevWordLower) continue;
    if (seenWords.has(lower)) continue;

    // 2. Block exact 2-gram repetition (prevents "in the middle of the middle")
    if (existingBigramsFromLast.has(lower) && !(isPossessiveEnd && contextEntities.includes(lower))) {
      continue;
    }

    let adjustedLogit = topVals[i];

    // 3. Apply standard LLM Frequency / Repetition Penalty to already-used content words
    const priorCount = wordCounts.get(lower) || 0;
    if (priorCount > 0 && !(isPossessiveEnd && contextEntities.includes(lower))) {
      if (ALLOWED_FUNCTION_REPEATS.has(lower)) {
        adjustedLogit -= priorCount * 0.6;
      } else {
        adjustedLogit -= priorCount * 3.2;
      }
    }

    seenWords.add(lower);
    candidates.push({
      token: word,
      rawLogit: adjustedLogit,
    });
    if (candidates.length >= 12) break;
  }

  // Sort after repetition penalty adjustments
  candidates.sort((a, b) => b.rawLogit - a.rawLogit);

  // If the sentence ends with a possessive ("my", "her", "his") and an entity was introduced earlier
  // in the active window (like "I have a parrot. Every morning I feed my"), boost that entity
  if (isPossessiveEnd && contextEntities.length > 0 && candidates.length > 0) {
    const topRaw = candidates[0].rawLogit;
    for (let eIdx = contextEntities.length - 1; eIdx >= 0; eIdx--) {
      const ent = contextEntities[eIdx];
      // Do not boost if the entity already appeared right after this same possessive
      if (existingBigramsFromLast.has(ent)) continue;
      const existing = candidates.find((c) => c.token.toLowerCase() === ent.toLowerCase());
      if (existing) {
        existing.rawLogit = Math.max(existing.rawLogit, topRaw + 0.85);
      } else {
        candidates.unshift({
          token: ent,
          rawLogit: topRaw + 0.85,
        });
      }
    }
    candidates.sort((a, b) => b.rawLogit - a.rawLogit);
  }

  if (candidates.length === 0) return null;

  // Normalize top 6 logits relative to top choice so temperature [0.0, 1.0] behaves consistently
  const top6 = candidates.slice(0, 6);
  const maxL = top6[0].rawLogit;
  const result = top6.map((c) => ({
    token: c.token,
    baseLogit: 3.65 + (c.rawLogit - maxL),
  }));

  if (logitCache.size > 120) {
    logitCache.clear();
  }
  logitCache.set(activeText, result);
  return result;
}

function sortPairsDesc(ids, vals) {
  for (let i = 1; i < vals.length; i++) {
    const v = vals[i];
    const id = ids[i];
    let j = i - 1;
    while (j >= 0 && vals[j] < v) {
      vals[j + 1] = vals[j];
      ids[j + 1] = ids[j];
      j--;
    }
    vals[j + 1] = v;
    ids[j + 1] = id;
  }
}
