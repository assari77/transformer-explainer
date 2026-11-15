# Transformer Explainer — Project Writeup

## 1. Overview
We built an ultra-simple, zero-dependency version of the [Polo Club Transformer Explainer](https://poloclub.github.io/transformer-explainer/) using pure HTML5, CSS3, and ES6+ JavaScript modules. Users can type or select any partial sentence, watch it break into embeddings, view a live visual diagram of the Transformer in action, and adjust a **Temperature** slider (`0.1` to `2.0`) to see its real-time impact on next-token probabilities and word selection.

- **Project Directory**: `C:\data\AGY\transformer-explainer`

---

## 2. Architecture & Codebase

| File | Purpose |
| :--- | :--- |
| [index.html](file:///C:/data/AGY/transformer-explainer/index.html) | App layout: header, Top Control Bar (Partial Sentence input, `Append Selected` / `Reset`, and `Temperature` slider), **1. EMBEDDING**, **2. TRANSFORMER IN ACTION** visual diagram, and **3. PROBABILITIES** panel |
| [styles.css](file:///C:/data/AGY/transformer-explainer/styles.css) | Responsive 3-column layout matching the Neural Network Playground theme (`#183d4e`), diverging orange-to-blue vector strips, animated SVG flow lines, and probability bar styling |
| [src/transformer.js](file:///C:/data/AGY/transformer-explainer/src/transformer.js) | Pure-JS Transformer engine: word tokenization, 8D Token + Positional Embeddings, Causal Self-Attention + Feed-Forward MLP forward pass, and temperature-scaled Softmax $\text{Softmax}(\text{logits} / T)$ + token sampling |
| [src/presets.js](file:///C:/data/AGY/transformer-explainer/src/presets.js) | 3 example partial sentences and candidate vocabulary |
| [src/heatmap.js](file:///C:/data/AGY/transformer-explainer/src/heatmap.js) | Diverging color-scale (`#f59322` $\leftrightarrow$ `#ffffff` $\leftrightarrow$ `#0877bd`) vector strip renderer for embeddings |
| [src/main.js](file:///C:/data/AGY/transformer-explainer/src/main.js) | Live sentence editing, Temperature slider binding, dynamic SVG **Transformer in Action** diagram renderer, and probability/selection updates |
| [tests/transformer.test.js](file:///C:/data/AGY/transformer-explainer/tests/transformer.test.js) | Automated unit tests (`node --test`) verifying tokenization, embedding shapes, and Temperature sharpening/flattening on Softmax probabilities |

---

## 3. UI Customizations & Simplifications
1. **Partial Sentence Input + Presets**: Users can type any custom partial sentence (or click `Example 1–3`) and press **Append Selected** to grow the sentence token by token.
2. **Interactive Context Window Slider (`2` – `15` tokens, default `10`)**:
   - Demonstrates how a Transformer's finite context window works.
   - In `"I have a parrot. Every morning I feed my"`, at `Context Window = 10`, all 9 tokens are in memory and Self-Attention recalls `"parrot"` (`#1 Selected`).
   - Sliding `Context Window` down to `5 tokens` drops `"I have a parrot."` (marked `FORGOTTEN` and struck through in Block 1), so the model only sees `"Every morning I feed my"` and reverts to predicting `"cat"`.
3. **Interactive Temperature Slider (`0.0` – `1.0`)**:
   - **`0.0` (Deterministic / Greedy)**: Gives `100%` probability to the #1 candidate and always selects it.
   - **`0.5` (Balanced, default)**: Shows a natural Softmax distribution across the top candidates.
   - **`0.7 – 1.0` (Creative)**: Flattens the probability bars so lower-ranked candidates gain probability mass and can be sampled automatically or via **Resample**.
3. **Clean 3-Block Layout**:
   - **1. EMBEDDING**: Shows the sentence broken into numbered word tokens and their 8D `Token + Positional` vector strips.
   - **2. TRANSFORMER IN ACTION**: Shows a live, animated SVG diagram of the Transformer architecture (`Input Tokens` $\to$ `Masked Self-Attention` $\to$ `Feed-Forward MLP` + `Residual & LayerNorm` $\to$ `Softmax(T)` $\to$ `Selected Word`).
   - **3. PROBABILITIES**: Displays the highlighted **Selected Next Token** card (`Resample` button) and the ranked candidate probability bars.
