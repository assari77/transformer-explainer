import { PRESET_PROMPTS } from "./presets.js";
import { tokenize, forwardTransformer, selectNextToken } from "./transformer.js";
import { renderVectorStrip } from "./heatmap.js";
import { initGpt2, predictGpt2Candidates, getGpt2Status } from "./gpt2.js";

const state = {
  baseText: PRESET_PROMPTS[0].text,
  currentText: PRESET_PROMPTS[0].text,
  temperature: 0.5,
  contextWindow: 10,
  gpt2Candidates: null,
  lastActiveKey: "",
  reqSeq: 0,
  trace: null,
};

const el = {
  modelStatusBadge: document.getElementById("model-status-badge"),
  promptInput: document.getElementById("prompt-input"),
  sentenceTokenCount: document.getElementById("sentence-token-count"),
  presetPills: document.getElementById("preset-pills"),
  btnGenerate: document.getElementById("btn-generate"),
  btnReset: document.getElementById("btn-reset"),
  contextSlider: document.getElementById("context-slider"),
  contextValue: document.getElementById("context-value"),
  contextStatus: document.getElementById("context-status"),
  tempSlider: document.getElementById("temp-slider"),
  tempValue: document.getElementById("temp-value"),
  tempRegime: document.getElementById("temp-regime"),
  embeddingTokenList: document.getElementById("embedding-token-list"),
  diagramSvg: document.getElementById("transformer-diagram-svg"),
  selectedTokenText: document.getElementById("selected-token-text"),
  selectedTokenProb: document.getElementById("selected-token-prob"),
  btnResample: document.getElementById("btn-resample"),
  probList: document.getElementById("prob-list"),
};

function getTemperatureLabel(t) {
  if (t <= 0.02) return "Deterministic / Greedy";
  if (t <= 0.3) return "Focused";
  if (t <= 0.6) return "Balanced";
  return "Creative / Diverse";
}

function updateAndRender() {
  const tokens = tokenize(state.currentText);
  const maxWin = Math.max(1, Math.floor(Number(state.contextWindow) || 10));
  const activeTokens = tokens.slice(Math.max(0, tokens.length - maxWin));
  const activeKey = activeTokens.join(" ").trim();

  // If the active window text changed and GPT-2 is ready, query real GPT-2 logits
  const { isReady } = getGpt2Status();
  if (isReady && activeKey !== state.lastActiveKey) {
    state.lastActiveKey = activeKey;
    const seq = ++state.reqSeq;
    predictGpt2Candidates(activeTokens).then((cands) => {
      if (seq === state.reqSeq && cands) {
        state.gpt2Candidates = cands;
        renderFromCurrentState();
      }
    });
  } else if (!isReady) {
    state.gpt2Candidates = null;
  }

  renderFromCurrentState();
}

function renderFromCurrentState() {
  const tokens = tokenize(state.currentText);
  state.trace = forwardTransformer(
    tokens,
    state.temperature,
    Math.random,
    state.contextWindow,
    state.gpt2Candidates
  );

  const totalTokens = state.trace.allTokens.length;
  const activeCount = state.trace.tokens.length;
  const dropped = state.trace.droppedCount;

  if (el.sentenceTokenCount) {
    const wordLabel = totalTokens === 1 ? "token" : "tokens";
    if (dropped > 0) {
      el.sentenceTokenCount.innerHTML = `Token count: <strong>${totalTokens} ${wordLabel}</strong> &middot; <span>${activeCount} in memory</span>, <span class="forgotten-inline">${dropped} forgotten</span>`;
    } else {
      el.sentenceTokenCount.innerHTML = `Token count: <strong>${totalTokens} ${wordLabel}</strong> &middot; <span>all ${totalTokens} in memory</span>`;
    }
  }

  if (el.contextValue) {
    el.contextValue.textContent = `${state.contextWindow} tokens`;
  }
  if (el.contextStatus) {
    if (state.trace.droppedCount > 0) {
      el.contextStatus.textContent = `${state.trace.droppedCount} forgotten`;
      el.contextStatus.style.color = "#b91c1c";
    } else {
      el.contextStatus.textContent = "All in memory";
      el.contextStatus.style.color = "";
    }
  }

  el.tempValue.textContent = state.temperature.toFixed(1);
  el.tempRegime.textContent = getTemperatureLabel(state.temperature);

  renderPresetPills();
  renderEmbeddings();
  renderTransformerDiagram();
  renderProbabilities();
  scrollPromptInputToEnd();
}

function scrollPromptInputToEnd() {
  if (!el.promptInput) return;
  el.promptInput.scrollTop = el.promptInput.scrollHeight;
  el.promptInput.scrollLeft = el.promptInput.scrollWidth;
}

function renderPresetPills() {
  el.presetPills.innerHTML = "";
  PRESET_PROMPTS.forEach((p, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    const isActive = state.currentText.trim().toLowerCase().startsWith(p.text.toLowerCase());
    btn.className = `preset-pill ${isActive ? "active" : ""}`;
    btn.textContent = p.label || `Example ${idx + 1}`;
    btn.title = p.text;
    btn.addEventListener("click", () => {
      state.baseText = p.text;
      state.currentText = p.text;
      el.promptInput.value = p.text;
      updateAndRender();
    });
    el.presetPills.appendChild(btn);
  });
}

function renderEmbeddings() {
  el.embeddingTokenList.innerHTML = "";
  const { allTokens, allInputEmbeddings, contextStartIndex } = state.trace;

  allTokens.forEach((tok, idx) => {
    if (contextStartIndex > 0 && idx === contextStartIndex) {
      const divider = document.createElement("div");
      divider.className = "context-cutoff-divider";
      divider.textContent = `Active Context Window (Last ${state.contextWindow})`;
      el.embeddingTokenList.appendChild(divider);
    }

    const isOutOfContext = idx < contextStartIndex;
    const isLast = idx === allTokens.length - 1;
    const row = document.createElement("div");
    row.className = "embedding-row";
    if (isOutOfContext) row.classList.add("out-of-context");
    if (isLast) row.classList.add("last-token");

    const info = document.createElement("div");
    info.className = "emb-token-info";
    info.innerHTML = `
      <span class="emb-pos-badge">#${idx}</span>
      <span class="emb-token-text">${escapeXml(tok)}</span>
    `;

    row.appendChild(info);

    if (isOutOfContext) {
      const badge = document.createElement("span");
      badge.className = "forgotten-badge";
      badge.textContent = "Forgotten";
      row.appendChild(badge);
    } else {
      const canvas = document.createElement("canvas");
      canvas.width = 96;
      canvas.height = 18;
      renderVectorStrip(canvas, allInputEmbeddings[idx], { scale: 1.2 });
      row.appendChild(canvas);
    }

    el.embeddingTokenList.appendChild(row);
  });
  // Keep newest tokens in view when list grows
  el.embeddingTokenList.scrollTop = el.embeddingTokenList.scrollHeight;
}

/**
 * Renders the Polo-Club-inspired "Transformer in Action" visual diagram inside Block 2.
 */
function renderTransformerDiagram() {
  const {
    tokens,
    attnWeightsLastToken,
    temperature,
    selectedToken,
    selectedProb,
    droppedCount,
    contextWindow,
  } = state.trace;
  // Show up to the last 6 active in-window tokens in the visual diagram
  const visibleTokens = tokens.slice(-6);
  const visibleWeights = attnWeightsLastToken.slice(-6);
  const n = visibleTokens.length;

  const yTop = 106;
  const yBottom = 368;
  const stepY = n > 1 ? (yBottom - yTop) / (n - 1) : 0;
  const tokenYs = visibleTokens.map((_, i) => (n === 1 ? 215 : Math.round(yTop + i * stepY)));
  const lastY = tokenYs[n - 1];

  let tokenNodesSvg = "";
  let attentionArcsSvg = "";
  let mlpFlowSvg = "";

  visibleTokens.forEach((tok, i) => {
    const y = tokenYs[i];
    const isLast = i === n - 1;
    const label = tok.length > 10 ? tok.slice(0, 9) + "…" : tok;
    const w = visibleWeights[i] || 0;
    const strokeW = (1.2 + w * 4.5).toFixed(1);
    const opacity = (0.22 + w * 0.75).toFixed(2);

    tokenNodesSvg += `
      <g transform="translate(14, ${y - 14})">
        <rect x="0" y="0" width="86" height="28" rx="6"
          fill="${isLast ? "#e0f2fe" : "#ffffff"}"
          stroke="${isLast ? "#0877bd" : "#cbd5e1"}"
          stroke-width="${isLast ? "2" : "1.2"}" />
        <text x="43" y="18" text-anchor="middle" font-size="11" font-weight="${isLast ? "700" : "600"}" fill="#1e293b">${escapeXml(label)}</text>
      </g>
      <line x1="100" y1="${y}" x2="146" y2="${y}" stroke="${isLast ? "#0877bd" : "#94a3b8"}" stroke-width="1.5" class="flow-animated" />
    `;

    attentionArcsSvg += `
      <path d="M 164 ${y} C 215 ${y}, 225 ${lastY}, 276 ${lastY}"
        fill="none"
        stroke="${isLast ? "#0877bd" : "#f59322"}"
        stroke-width="${strokeW}"
        stroke-opacity="${opacity}" />
      <circle cx="164" cy="${y}" r="4" fill="${isLast ? "#0877bd" : "#f59322"}" />
    `;
  });

  mlpFlowSvg = `
    <circle cx="276" cy="${lastY}" r="5" fill="#0877bd" />
    <path d="M 281 ${lastY} C 310 ${lastY}, 315 215, 342 215" fill="none" stroke="#0877bd" stroke-width="2.5" class="flow-animated" />
  `;

  const pctText = `${(selectedProb * 100).toFixed(1)}%`;
  const memorySubtext =
    droppedCount > 0
      ? `WINDOW=${contextWindow} (${droppedCount} EARLIER TOKEN${droppedCount > 1 ? "S" : ""} FORGOTTEN)`
      : `WINDOW=${contextWindow} (ALL TOKENS IN MEMORY)`;
  const memoryColor = droppedCount > 0 ? "#b91c1c" : "#183d4e";

  el.diagramSvg.innerHTML = `
    <defs>
      <linearGradient id="blockGrad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#f8fafc" />
        <stop offset="100%" stop-color="#f1f5f9" />
      </linearGradient>
      <marker id="arrowBlue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 8 5 L 0 9 z" fill="#0877bd" />
      </marker>
    </defs>

    <!-- Stage Header Labels -->
    <text x="57" y="26" text-anchor="middle" font-size="10" font-weight="700" fill="#64748b" letter-spacing="0.05em">IN-WINDOW TOKENS</text>
    <text x="335" y="26" text-anchor="middle" font-size="10" font-weight="700" fill="${memoryColor}" letter-spacing="0.04em">${memorySubtext}</text>
    <text x="605" y="26" text-anchor="middle" font-size="10" font-weight="700" fill="#64748b" letter-spacing="0.05em">SOFTMAX (T=${temperature.toFixed(1)})</text>

    <!-- Outer Transformer Block Container -->
    <rect x="132" y="38" width="404" height="364" rx="12" fill="url(#blockGrad)" stroke="#cbd5e1" stroke-width="1.5" />

    <!-- Sub-box 1: Masked Multi-Head Self-Attention -->
    <rect x="148" y="54" width="144" height="332" rx="8" fill="#ffffff" stroke="#93c5fd" stroke-width="1.5" />
    <text x="220" y="74" text-anchor="middle" font-size="11" font-weight="700" fill="#0369a1">Masked Self-Attention</text>
    <text x="220" y="89" text-anchor="middle" font-size="9.5" fill="#64748b">Q · Kᵀ / √d → Softmax</text>

    <!-- Dynamic Token Pills & Causal Attention Convergence Lines -->
    ${tokenNodesSvg}
    ${attentionArcsSvg}
    ${mlpFlowSvg}

    <!-- Sub-box 2: Feed-Forward Network (MLP + GELU) -->
    <rect x="342" y="130" width="174" height="170" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="429" y="154" text-anchor="middle" font-size="11" font-weight="700" fill="#183d4e">Feed-Forward MLP</text>
    <text x="429" y="170" text-anchor="middle" font-size="9.5" fill="#64748b">Linear → GELU → Linear</text>

    <!-- Mini Neural Network Graphic inside the MLP Box -->
    <g stroke="#94a3b8" stroke-width="1" opacity="0.65">
      <line x1="374" y1="196" x2="429" y2="188" /><line x1="374" y1="196" x2="429" y2="212" /><line x1="374" y1="196" x2="429" y2="236" />
      <line x1="374" y1="226" x2="429" y2="188" /><line x1="374" y1="226" x2="429" y2="212" /><line x1="374" y1="226" x2="429" y2="236" /><line x1="374" y1="226" x2="429" y2="260" />
      <line x1="374" y1="256" x2="429" y2="212" /><line x1="374" y1="256" x2="429" y2="236" /><line x1="374" y1="256" x2="429" y2="260" />

      <line x1="429" y1="188" x2="484" y2="202" /><line x1="429" y1="188" x2="484" y2="230" />
      <line x1="429" y1="212" x2="484" y2="202" /><line x1="429" y1="212" x2="484" y2="230" /><line x1="429" y1="212" x2="484" y2="254" />
      <line x1="429" y1="236" x2="484" y2="202" /><line x1="429" y1="236" x2="484" y2="230" /><line x1="429" y1="236" x2="484" y2="254" />
      <line x1="429" y1="260" x2="484" y2="230" /><line x1="429" y1="260" x2="484" y2="254" />
    </g>
    <!-- Input nodes of MLP -->
    <circle cx="374" cy="196" r="6" fill="#0877bd" />
    <circle cx="374" cy="226" r="6" fill="#38bdf8" />
    <circle cx="374" cy="256" r="6" fill="#0877bd" />
    <!-- Expanded GELU hidden nodes -->
    <circle cx="429" cy="188" r="6" fill="#f59322" />
    <circle cx="429" cy="212" r="6" fill="#f59322" />
    <circle cx="429" cy="236" r="6" fill="#f59322" />
    <circle cx="429" cy="260" r="6" fill="#f59322" />
    <!-- Output nodes of MLP -->
    <circle cx="484" cy="202" r="6" fill="#183d4e" />
    <circle cx="484" cy="230" r="6" fill="#0877bd" />
    <circle cx="484" cy="254" r="6" fill="#183d4e" />

    <!-- Residual Bypass Arc Label -->
    <path d="M 312 215 C 312 102, 495 102, 518 190" fill="none" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="4 3" />
    <text x="418" y="108" text-anchor="middle" font-size="9.5" font-weight="600" fill="#64748b">+ Residual &amp; LayerNorm</text>

    <!-- Output flow from MLP to Softmax & Selected Token -->
    <line x1="516" y1="215" x2="554" y2="215" stroke="#0877bd" stroke-width="2.5" marker-end="url(#arrowBlue)" />

    <!-- Right Output Box: Temperature Scaling & Selected Next Word -->
    <rect x="558" y="148" width="110" height="134" rx="10" fill="#fff7ed" stroke="#f59322" stroke-width="2" />
    <text x="613" y="172" text-anchor="middle" font-size="9.5" font-weight="700" fill="#c2410c">SELECTED WORD</text>
    <text x="613" y="212" text-anchor="middle" font-size="16" font-weight="800" fill="#183d4e">"${escapeXml(selectedToken)}"</text>
    <rect x="578" y="228" width="70" height="22" rx="11" fill="#f59322" />
    <text x="613" y="243" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">${pctText}</text>
    <text x="613" y="270" text-anchor="middle" font-size="9.5" fill="#9a3412">at T = ${temperature.toFixed(1)}</text>
  `;
}

function renderProbabilities() {
  const { topPredictions, selectedToken, selectedProb } = state.trace;

  el.selectedTokenText.textContent = `"${selectedToken}"`;
  el.selectedTokenProb.textContent = `${(selectedProb * 100).toFixed(1)}%`;

  el.probList.innerHTML = "";
  topPredictions.forEach((item) => {
    const isSelected = item.token === selectedToken;
    const pct = (item.prob * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = `prob-row ${isSelected ? "is-selected" : ""}`;
    row.innerHTML = `
      <div class="prob-row-top">
        <div class="prob-token-wrap">
          <span class="prob-token">${escapeXml(item.token)}</span>
          ${isSelected ? '<span class="selected-pill">Selected</span>' : ""}
        </div>
        <span class="prob-pct">${pct}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${Math.max(2, item.prob * 100).toFixed(1)}%"></div>
      </div>
    `;
    row.style.cursor = "pointer";
    row.title = `Click to select "${item.token}"`;
    row.addEventListener("click", () => {
      state.trace.selectedToken = item.token;
      state.trace.selectedProb = item.prob;
      renderTransformerDiagram();
      renderProbabilities();
    });
    el.probList.appendChild(row);
  });
}

function escapeXml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Event Listeners
el.promptInput.addEventListener("input", () => {
  state.currentText = el.promptInput.value;
  updateAndRender();
});

el.contextSlider?.addEventListener("input", () => {
  state.contextWindow = parseInt(el.contextSlider.value, 10);
  updateAndRender();
});

el.tempSlider.addEventListener("input", () => {
  state.temperature = parseFloat(el.tempSlider.value);
  updateAndRender();
});

el.btnResample.addEventListener("click", () => {
  if (!state.trace) return;
  const picked = selectNextToken(state.trace.topPredictions, state.temperature, Math.random, true);
  if (picked) {
    state.trace.selectedToken = picked.token;
    state.trace.selectedProb = picked.prob;
    renderTransformerDiagram();
    renderProbabilities();
  }
});

el.btnGenerate.addEventListener("click", () => {
  if (!state.trace) return;
  const nextWord = state.trace.selectedToken;
  state.currentText = `${state.currentText.trim()} ${nextWord}`.trim();
  el.promptInput.value = state.currentText;
  updateAndRender();
});

el.btnReset.addEventListener("click", () => {
  state.currentText = state.baseText;
  el.promptInput.value = state.currentText;
  updateAndRender();
});

// Initialize slider ranges & labels (guards against browser form/HTML caching)
if (!el.sentenceTokenCount && el.promptInput) {
  const row = el.promptInput.closest(".sentence-editor");
  if (row) {
    const countDiv = document.createElement("div");
    countDiv.id = "sentence-token-count";
    countDiv.className = "sentence-token-count";
    row.appendChild(countDiv);
    el.sentenceTokenCount = countDiv;
  }
}
if (el.tempSlider) {
  el.tempSlider.min = "0.0";
  el.tempSlider.max = "1.0";
  el.tempSlider.step = "0.1";
  el.tempSlider.value = String(state.temperature);
  const scaleContainer = el.tempSlider.parentElement?.querySelector(".temp-scale-labels");
  if (scaleContainer) {
    scaleContainer.innerHTML = `
      <span>0.0 (Focused)</span>
      <span>0.5 (Balanced)</span>
      <span>1.0 (Creative)</span>
    `;
  }
}
if (el.contextSlider) {
  el.contextSlider.value = String(state.contextWindow);
}
updateAndRender();

// Start loading real in-browser GPT-2 (Xenova/distilgpt2) in the background
initGpt2((statusText, ready) => {
  if (el.modelStatusBadge) {
    el.modelStatusBadge.textContent = statusText;
    el.modelStatusBadge.classList.toggle("ready", Boolean(ready));
  }
  if (ready) {
    state.lastActiveKey = "";
    updateAndRender();
  }
});
