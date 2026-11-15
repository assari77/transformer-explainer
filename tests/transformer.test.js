import test from "node:test";
import assert from "node:assert/strict";
import {
  D_MODEL,
  tokenize,
  forwardTransformer,
} from "../src/transformer.js";
import { PRESET_PROMPTS } from "../src/presets.js";

test("Tokenizes partial sentence and produces 8D token + positional embeddings", () => {
  const tokens = tokenize("Data visualization empowers users to");
  assert.deepEqual(tokens, ["Data", "visualization", "empowers", "users", "to"]);

  const trace = forwardTransformer(tokens, 1.0);
  assert.equal(trace.inputEmbeddings.length, 5);
  assert.equal(trace.inputEmbeddings[0].length, D_MODEL);
  assert.equal(trace.attnWeightsLastToken.length, 5);

  const attnSum = trace.attnWeightsLastToken.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(attnSum - 1.0) < 1e-6, `Expected attention weights to sum to 1.0, got ${attnSum}`);
});

test("Temperature slider (0.0 to 1.0) gives 100% at T=0.0, balanced at T=0.5, and flatter at T=1.0", () => {
  const tokens = tokenize(PRESET_PROMPTS[0].text);
  const zeroTemp = forwardTransformer(tokens, 0.0);
  const midTemp = forwardTransformer(tokens, 0.5);
  const maxTemp = forwardTransformer(tokens, 1.0);

  const pZero = zeroTemp.topPredictions[0].prob;
  const pMid = midTemp.topPredictions[0].prob;
  const pMax = maxTemp.topPredictions[0].prob;

  assert.equal(pZero, 1.0, `Expected T=0.0 top probability to be 1.0 (100%), got ${pZero}`);
  assert.ok(pZero > pMid, `Expected T=0.0 prob (${pZero}) > T=0.5 prob (${pMid})`);
  assert.ok(pMid > pMax, `Expected T=0.5 prob (${pMid}) > T=1.0 prob (${pMax})`);
});

test("Generates natural, grammatically coherent continuations for custom everyday sentences", () => {
  // 1. "every morning i feed my" -> should predict pets/animals ("cat", "dog", "pets", "fish", ...)
  const trace1 = forwardTransformer(tokenize("every morning i feed my"), 0.5);
  const cands1 = trace1.topPredictions.map((p) => p.token);
  assert.equal(trace1.selectedToken, "cat");
  assert.deepEqual(cands1.slice(0, 4), ["cat", "dog", "pets", "fish"]);

  // 2. "every morning i feed my cat" -> natural continuation ("and", "some", "before", ...)
  const trace2 = forwardTransformer(tokenize("every morning i feed my cat"), 0.5);
  assert.equal(trace2.selectedToken, "and");

  // 3. "every morning i drink my" -> beverages ("coffee", "tea", "water", "milk", ...)
  const trace3 = forwardTransformer(tokenize("every morning i drink my"), 0.5);
  assert.equal(trace3.selectedToken, "coffee");

  // 4. Self-attention entity recall: "i have a parrot. every morning i feed my" -> "parrot" at #1 when contextWindow=10
  const trace4 = forwardTransformer(tokenize("i have a parrot. every morning i feed my"), 0.5, Math.random, 10);
  const cands4 = trace4.topPredictions.map((p) => p.token);
  assert.equal(trace4.selectedToken, "parrot");
  assert.equal(trace4.droppedCount, 0);
  assert.ok(cands4.includes("parrot"), `Expected candidates to include "parrot", got ${cands4}`);
  assert.ok(cands4.includes("bird"), `Expected candidates to include companion "bird", got ${cands4}`);

  // 5. Context Window Forgetting: when contextWindow=5, "i have a parrot." is dropped (4 tokens forgotten)
  //    so the model only sees "every morning i feed my" and reverts to predicting "cat"!
  const trace5 = forwardTransformer(tokenize("i have a parrot. every morning i feed my"), 0.5, Math.random, 5);
  assert.equal(trace5.droppedCount, 4);
  assert.deepEqual(trace5.tokens, ["every", "morning", "i", "feed", "my"]);
  assert.equal(trace5.selectedToken, "cat");
  assert.ok(!trace5.topPredictions.map((p) => p.token).includes("parrot"));

  // 6. Classic idiom: "it was a dark and stormy" -> "night"
  const trace6 = forwardTransformer(tokenize("it was a dark and stormy"), 0.5, Math.random, 15);
  assert.equal(trace6.selectedToken, "night");
});
