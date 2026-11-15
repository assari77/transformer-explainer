# How a Transformer Works — Beginner Slide Deck & Presenter Guide

This 10-slide deck is designed to be presented side-by-side with the **Transformer Explainer** web app (`index.html` and `slides.html`) for complete beginners.

---

## Slide 1: How Do AI Models Like ChatGPT Actually Work?
- **Visual / Headline**: Behind all the magic of modern AI is a single invention called the **Transformer** — and its core job is surprisingly simple: **predicting the very next word**.
- **Bullet Points**:
  - You give the AI an unfinished sentence: `"The cat sat on the ___"`
  - The AI scores every word it knows and picks the most likely continuation: `"warm"` (`55.1%`) or `"mat"` (`22.4%`).
- **Live App Action**: Point to the **Partial Sentence** input box on the top left and the **Selected Next Token** box on the right.
- **Speaker Script**: *"When you chat with ChatGPT, it feels like it's writing whole paragraphs at once. But under the hood, it's playing a super-smart game of autocomplete—predicting just one word at a time."*

---

## Slide 2: The 3 Building Blocks of Our App
- **Visual / Headline**: Text In $\to$ Numbers $\to$ Context & Thinking $\to$ Probabilities $\to$ Next Word Out
- **Bullet Points**:
  1. **1. EMBEDDING (Left Panel)**: Breaks your sentence into word **tokens** and turns each word into a list of numbers (shown as colored squares).
  2. **2. TRANSFORMER IN ACTION (Center Panel)**: Connects words together using **Self-Attention** (context) and a **Neural Network (MLP)**.
  3. **3. PROBABILITIES (Right Panel)**: Converts the final numbers into percentage scores and selects the winning next word.
- **Speaker Script**: *"Our screen is divided into three columns that match the exact three steps inside every Transformer."*

---

## Slide 3: Step 1 — Turning Words Into Numbers (`1. EMBEDDING`)
- **Visual / Headline**: Computers Can't Read Letters — They Read Vectors (Lists of Numbers)
- **Bullet Points**:
  - **Tokenization**: Each word is split into a token and numbered `#0`, `#1`, `#2`, …
  - **Embedding Vector (`-1` to `+1`)**: Each word becomes a list of 8 numbers:
    - **Blue** = Positive (`+1`)
    - **White** = Near zero (`0`)
    - **Orange** = Negative (`-1`)
  - *(Note: Real GPT-2 uses a list of 768 numbers per word; we show 8 blocks so you can see them clearly!)*
- **Live App Action**: Point to **1. EMBEDDING** on the left and the **Embedding Vector Scale (`-1` to `+1`)** at the bottom right.

---

## Slide 4: Why Word Order Matters (`TOKEN + POS`)
- **Visual / Headline**: *"The dog bit the man"* $\neq$ *"The man bit the dog"*
- **Bullet Points**:
  - If we only looked up a word's meaning, the computer wouldn't know what order the words came in.
  - So the model adds a **Position signal (`POS`)** based on whether the word is at spot `#0`, `#1`, or `#4`.
- **Live App Demo #1**:
  1. Click **`Example 3`** (`"The cat sat on the"`).
  2. Look at **`#0 The`** (first row) and **`#4 the`** (last row) in Block 1.
  3. Notice they have the same overall pattern (blue on the left, orange on the right), **but slightly different shades** because one is at position `#0` and the other is at position `#4`!

---

## Slide 5: Step 2A — Masked Self-Attention (`2. TRANSFORMER IN ACTION`)
- **Visual / Headline**: How the AI Connects Clues Across Your Sentence
- **Bullet Points**:
  - Suppose you type: `"I have a parrot. Every morning I feed my ___"`
  - How does the AI know what follows `"my"`? The final word looks backward at earlier words using **Self-Attention** (the orange curves).
  - **Thicker orange curve** = The model is paying strong attention to that earlier word (`"parrot."`).
  - **Why "Masked"?** When predicting the next word, the model can only look at *past* words, never future words.
- **Live App Demo #2**:
  1. Click **`Parrot (Context Test)`** (`"I have a parrot. Every morning I feed my"`).
  2. Show how **3. PROBABILITIES** predicts **`"parrot"`** (`#1 Selected`) instead of generic pets like `"cat"` or `"dog"`.

---

## Slide 6: Context Window — How Much Can AI Remember?
- **Visual / Headline**: Every AI Has a Finite Working Memory (**Context Window**)
- **Bullet Points**:
  - The **Context Window** is the maximum number of recent tokens the AI can hold in memory at one time.
  - Any words earlier than that cutoff fall out of memory and are **completely forgotten**.
- **Live App Demo #3 (Crowd Favorite!)**:
  1. Keep **`Parrot (Context Test)`** active (`9 tokens` total).
  2. At **`Context Window = 10 tokens`**, all 9 words are in memory $\to$ the model remembers `"parrot"` and predicts **`"parrot"`**.
  3. Now slowly drag the **Context Window** slider down to **`5 tokens`**.
  4. Watch tokens `#0`–`#3` (`"I have a parrot."`) get crossed out and tagged **`FORGOTTEN`** in Block 1—and watch the prediction in Block 3 immediately switch from **`"parrot"`** to **`"cat"`**!

---

## Slide 7: Step 2B — The Feed-Forward MLP & Residual Shortcut
- **Visual / Headline**: Self-Attention Gathers Context; the MLP Recalls World Knowledge
- **Bullet Points**:
  - **Self-Attention** looks *sideways* at the words in your prompt (finding clues like `"parrot"`).
  - **Feed-Forward MLP (`Linear → GELU → Linear`)** is a classic neural network that recalls *world knowledge* learned during training (e.g., parrots eat seeds/fruit; cats sit on warm mats).
  - **`+ Residual & LayerNorm`** (the dotted arch on top): A shortcut highway that preserves the original word signal so information doesn't fade away.
- **Live App Action**: Trace the blue dashed line in Block 2 from **Masked Self-Attention** through **Feed-Forward MLP** to the orange **SELECTED WORD** box.

---

## Slide 8: Step 3 — Softmax & Next-Word Probabilities (`3. PROBABILITIES`)
- **Visual / Headline**: The AI Thinks in Probabilities, Not Single Answers
- **Bullet Points**:
  - At the end of the Transformer block, the model produces raw scores (**logits**) for candidate words.
  - A math formula called **Softmax** converts those scores into percentages (`0%` to `100%`) that add up to `100%`.
  - Panel 3 shows the winning **Selected Next Token** at the top and the top 6 candidates below.

---

## Slide 9: Temperature (`0.0` to `1.0`) — Focus vs. Creativity
- **Visual / Headline**: One Slider Controls How Predictable or Creative the AI Is
- **Bullet Points**:
  - **`0.0 (Focused / Deterministic)`**: Puts **100%** on the #1 word. Always gives the exact same safe answer (great for math, facts, and coding).
  - **`0.5 (Balanced)`**: Natural probability spread (~55% on the top choice).
  - **`1.0 (Creative)`**: Flattens the probability bars so #2, #3, and #4 words have a real chance of being picked (great for brainstorming and storytelling).
- **Live App Demo #4**:
  1. Drag **Temperature** to **`0.0`** $\to$ show the top bar jump to **`100.0%`**.
  2. Drag **Temperature** to **`1.0`** $\to$ show the bars flatten out, then click **↻ Resample** several times to watch different words get chosen!

---

## Slide 10: How One Word Becomes a Whole Paragraph
- **Visual / Headline**: The Autoregressive Loop — Appending & Repeating
- **Bullet Points**:
  - How does a single-word predictor write an entire sentence or essay?
  - It takes the **Selected Word**, appends it to the end of your sentence, and runs the whole 3-step pipeline again!
    - Pass 1: `"Data visualization empowers users to"` $\to$ **`"explore"`**
    - Pass 2: `"... users to explore"` $\to$ **`"complex"`**
    - Pass 3: `"... to explore complex"` $\to$ **`"models"`**
- **Live App Demo #5**:
  1. Click **`Example 1`**.
  2. Click **▶ Append Selected** 3 times in a row and watch the sentence and token counter grow step-by-step!
