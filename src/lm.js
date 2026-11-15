/**
 * Contextual English Language Model (Self-Attention Entity Memory + POS + Semantic Frames).
 * Tracks entities/nouns introduced earlier in the prompt (e.g., "i have a parrot. every morning i feed my ___" -> "parrot")
 * and combines them with verb-object collocations and grammatical part-of-speech slots.
 */

const VERB_OBJECT_FRAMES = {
  feed: ["cat", "dog", "pets", "fish", "birds", "chickens"],
  fed: ["cat", "dog", "pets", "fish", "birds", "chickens"],
  feeding: ["cat", "dog", "pets", "fish", "birds", "chickens"],
  pet: ["cat", "dog", "kitten", "puppy", "rabbit", "horse"],
  walk: ["dog", "puppy", "pet", "horse", "kids", "friends"],
  walked: ["dog", "puppy", "pet", "horse", "kids", "friends"],
  walking: ["dog", "puppy", "pet", "horse", "kids", "friends"],
  water: ["plants", "garden", "flowers", "trees", "lawn", "herbs"],
  watered: ["plants", "garden", "flowers", "trees", "lawn", "herbs"],
  drink: ["coffee", "tea", "water", "milk", "juice", "smoothie"],
  drank: ["coffee", "tea", "water", "milk", "juice", "smoothie"],
  drinking: ["coffee", "tea", "water", "milk", "juice", "smoothie"],
  sip: ["coffee", "tea", "water", "espresso", "latte", "juice"],
  pour: ["coffee", "tea", "water", "milk", "juice", "soup"],
  eat: ["breakfast", "lunch", "dinner", "toast", "eggs", "fruit"],
  ate: ["breakfast", "lunch", "dinner", "toast", "eggs", "fruit"],
  eating: ["breakfast", "lunch", "dinner", "toast", "eggs", "fruit"],
  cook: ["dinner", "breakfast", "lunch", "pasta", "eggs", "soup"],
  cooked: ["dinner", "breakfast", "lunch", "pasta", "eggs", "soup"],
  make: ["coffee", "breakfast", "dinner", "lunch", "tea", "plans"],
  made: ["coffee", "breakfast", "dinner", "lunch", "tea", "plans"],
  brush: ["teeth", "hair", "coat", "dog", "cat", "shoes"],
  brushed: ["teeth", "hair", "coat", "dog", "cat", "shoes"],
  wash: ["hands", "face", "dishes", "car", "clothes", "hair"],
  washed: ["hands", "face", "dishes", "car", "clothes", "hair"],
  clean: ["room", "kitchen", "house", "desk", "car", "cage"],
  cleaned: ["room", "kitchen", "house", "desk", "car", "cage"],
  drive: ["car", "truck", "van", "kids", "friends", "family"],
  drove: ["car", "truck", "van", "kids", "friends", "family"],
  ride: ["bike", "bicycle", "bus", "train", "horse", "scooter"],
  rode: ["bike", "bicycle", "bus", "train", "horse", "scooter"],
  read: ["book", "newspaper", "notes", "messages", "email", "story"],
  reading: ["book", "newspaper", "notes", "messages", "email", "story"],
  write: ["code", "notes", "stories", "emails", "letters", "reports"],
  wrote: ["code", "notes", "stories", "emails", "letters", "reports"],
  watch: ["movie", "show", "news", "sunset", "game", "video"],
  watched: ["movie", "show", "news", "sunset", "game", "video"],
  open: ["window", "door", "book", "laptop", "curtains", "cage"],
  opened: ["window", "door", "book", "laptop", "curtains", "cage"],
  close: ["door", "window", "book", "laptop", "gate", "eyes"],
  closed: ["door", "window", "book", "laptop", "gate", "eyes"],
  wear: ["jacket", "coat", "shoes", "hat", "glasses", "sweater"],
  wore: ["jacket", "coat", "shoes", "hat", "glasses", "sweater"],
  check: ["phone", "email", "messages", "calendar", "watch", "notes"],
  checked: ["phone", "email", "messages", "calendar", "watch", "notes"],
  call: ["mom", "friend", "parents", "sister", "brother", "doctor"],
  called: ["mom", "friend", "parents", "sister", "brother", "doctor"],
  love: ["family", "dog", "cat", "friends", "job", "home"],
  like: ["coffee", "music", "books", "dogs", "cats", "movies"],
  play: ["music", "games", "guitar", "piano", "soccer", "chess"],
  played: ["music", "games", "guitar", "piano", "soccer", "chess"],
  pack: ["bag", "lunch", "backpack", "suitcase", "books", "laptop"],
  grab: ["coffee", "keys", "jacket", "bag", "phone", "umbrella"],
  take: ["dog", "bus", "train", "walk", "coffee", "notes"],
  build: ["models", "systems", "apps", "tools", "projects", "websites"],
  train: ["models", "networks", "dogs", "teams", "students", "systems"],
  explore: ["complex", "new", "data", "models", "ideas", "patterns"],
  analyze: ["complex", "data", "patterns", "models", "trends", "results"],
  understand: ["complex", "how", "the", "new", "human", "data"],
};

// Related semantic companions when a specific noun is mentioned in the context
const ENTITY_COMPANIONS = {
  parrot: ["bird", "pet"],
  canary: ["bird", "pet"],
  parakeet: ["bird", "pet"],
  cockatiel: ["bird", "pet"],
  hamster: ["pet", "mouse"],
  rabbit: ["bunny", "pet"],
  bunny: ["rabbit", "pet"],
  turtle: ["tortoise", "pet"],
  tortoise: ["turtle", "pet"],
  goldfish: ["fish", "pet"],
  puppy: ["dog", "pet"],
  kitten: ["cat", "pet"],
  pony: ["horse", "pet"],
  iguana: ["lizard", "pet"],
  gecko: ["lizard", "pet"],
  ferret: ["pet"],
  bicycle: ["bike"],
  motorcycle: ["bike"],
  sedan: ["car"],
  suv: ["car", "truck"],
  espresso: ["coffee"],
  latte: ["coffee"],
  cappuccino: ["coffee"],
  novel: ["book", "story"],
};

const EXACT_PHRASE_CONTINUATIONS = {
  "data visualization empowers users to": ["explore", "understand", "analyze", "discover", "see", "build"],
  "data visualization empowers users to explore": ["complex", "new", "important", "hidden", "large", "rich"],
  "data visualization empowers users to explore complex": ["models", "patterns", "data", "systems", "datasets", "relationships"],
  "artificial intelligence transforms how we": ["learn", "work", "think", "build", "create", "communicate"],
  "artificial intelligence transforms how we learn": ["and", "new", "from", "about", "complex", "today"],
  "artificial intelligence transforms how we learn and": ["create", "work", "think", "build", "solve", "collaborate"],
  "the cat sat on the": ["warm", "mat", "soft", "sunny", "couch", "windowsill"],
  "the cat sat on the warm": ["mat", "rug", "sunny", "blanket", "couch", "floor"],
  "the cat sat on the warm sunny": ["mat", "windowsill", "porch", "rug", "floor", "bench"],
  "the cat sat on the mat": ["and", "purring", "quietly", "in", "while", "near"],
};

const PREPOSITION_NOUNS = {
  on: ["mat", "table", "couch", "floor", "chair", "bed", "desk", "perch"],
  in: ["morning", "kitchen", "park", "garden", "house", "cage", "evening", "world"],
  at: ["home", "work", "school", "night", "sunrise", "noon", "desk", "table"],
  under: ["table", "bed", "tree", "blanket", "bridge", "shade"],
  near: ["window", "door", "park", "river", "fire", "garden"],
  into: ["kitchen", "room", "garden", "house", "park", "car"],
  from: ["home", "work", "school", "kitchen", "garden", "window"],
  to: ["work", "school", "bed", "home", "class", "town"],
};

const POSSESSIVE_WORDS = new Set(["my", "your", "his", "her", "our", "their", "its"]);
const DETERMINERS = new Set(["a", "an", "the", "my", "your", "his", "her", "our", "their", "this", "that", "one", "two"]);
const TIME_NOUNS = new Set(["morning", "evening", "night", "afternoon", "day", "weekend", "week", "month", "year", "today", "tomorrow", "yesterday", "time"]);
const SUBJECT_PRONOUNS_PLURAL_OR_1ST = new Set(["i", "we", "you", "they"]);
const SUBJECT_PRONOUNS_3RD_SING = new Set(["he", "she", "it"]);
const MODAL_AUX = new Set(["can", "will", "would", "should", "could", "must", "may", "might", "do", "did", "dont", "didnt", "cannot"]);
const PREPOSITIONS = new Set(["in", "on", "at", "under", "near", "by", "with", "from", "into", "inside", "outside", "before", "after", "during", "without"]);
const ADJECTIVES = new Set([
  "warm", "soft", "sunny", "quiet", "cozy", "complex", "new", "important", "hidden",
  "large", "small", "fresh", "hot", "cold", "bright", "happy", "little", "big",
  "old", "young", "smart", "fast", "gentle", "sweet", "delicious", "clean", "modern",
  "green", "colorful", "clever", "friendly", "cute", "lovely",
  "dark", "stormy", "rainy", "windy", "cloudy", "foggy", "chilly", "freezing",
  "deep", "wide", "tall", "short", "long", "beautiful", "peaceful", "silent", "loud",
]);

const KNOWN_NOUNS = new Set([
  "cat", "dog", "pets", "pet", "fish", "birds", "bird", "chickens", "kitten", "puppy", "horse",
  "parrot", "canary", "parakeet", "hamster", "rabbit", "bunny", "turtle", "tortoise", "goldfish", "iguana", "ferret",
  "coffee", "tea", "water", "milk", "juice", "breakfast", "lunch", "dinner", "toast", "eggs", "fruit", "food", "seeds",
  "teeth", "hair", "hands", "face", "room", "kitchen", "house", "desk", "car", "bike", "bicycle", "bus", "train",
  "book", "newspaper", "notes", "messages", "email", "story", "movie", "show", "news", "sunset", "game",
  "window", "door", "laptop", "jacket", "coat", "shoes", "phone", "family", "friend", "friends", "kids",
  "mat", "rug", "floor", "chair", "bed", "couch", "table", "park", "garden", "plants", "flowers", "morning", "evening", "night",
  "models", "patterns", "data", "systems", "ideas", "users", "intelligence", "visualization",
]);

const KNOWN_VERBS = new Set([
  "have", "has", "had", "got", "own", "owned", "bought", "adopted", "found", "kept",
  "feed", "fed", "pet", "walk", "walked", "water", "watered", "drink", "drank", "sip", "pour", "eat", "ate", "cook", "cooked",
  "make", "made", "brush", "brushed", "wash", "washed", "clean", "cleaned", "drive", "drove", "ride", "rode",
  "read", "write", "wrote", "watch", "watched", "open", "opened", "close", "closed", "wear", "wore",
  "check", "checked", "call", "called", "love", "like", "play", "played", "pack", "grab", "take",
  "build", "train", "explore", "analyze", "understand", "learn", "think", "create", "solve", "sat", "sit",
  "sleep", "slept", "run", "ran", "jump", "jumped", "go", "went", "work", "worked",
]);

const FUNCTION_STOPWORDS = new Set([
  ...POSSESSIVE_WORDS,
  ...DETERMINERS,
  ...TIME_NOUNS,
  ...SUBJECT_PRONOUNS_PLURAL_OR_1ST,
  ...SUBJECT_PRONOUNS_3RD_SING,
  ...MODAL_AUX,
  ...PREPOSITIONS,
  ...KNOWN_VERBS,
  "every", "each", "and", "or", "but", "so", "because", "while", "when", "then", "to", "for", "of", "is", "are", "was", "were", "be", "been", "very", "really", "always", "usually", "often", "sometimes",
]);

export function cleanWord(w) {
  return String(w || "").toLowerCase().replace(/[^a-z]/g, "");
}

/**
 * Self-Attention Context Entity Extractor:
 * Scans earlier tokens in the prompt for nouns/entities introduced via patterns like
 * "i have a parrot", "bought a new bicycle", "my little hamster", or known entity nouns.
 * Returns `{ entities: string[], tokenIndices: number[] }` in most-salient-first order.
 */
export function extractContextEntities(tokens) {
  const cleanTokens = tokens.map(cleanWord);
  const entities = [];
  const tokenIndices = [];
  const seen = new Set();

  const addEntity = (word, idx) => {
    if (!word || word.length < 2) return;
    if (FUNCTION_STOPWORDS.has(word) || ADJECTIVES.has(word)) return;
    if (!seen.has(word)) {
      seen.add(word);
      entities.push(word);
    }
    if (idx !== undefined && !tokenIndices.includes(idx)) {
      tokenIndices.push(idx);
    }
  };

  // Pass 1: Look for nouns introduced after determiners ("a parrot", "the cute puppy", "my guitar")
  for (let i = 0; i < cleanTokens.length - 1; i++) {
    const w = cleanTokens[i];
    if (DETERMINERS.has(w)) {
      const next1 = cleanTokens[i + 1];
      const next2 = i + 2 < cleanTokens.length ? cleanTokens[i + 2] : "";
      if (ADJECTIVES.has(next1) && next2) {
        addEntity(next2, i + 2);
      } else {
        addEntity(next1, i + 1);
      }
    }
  }

  // Pass 2: Any other known non-time noun mentioned earlier in the sentence
  for (let i = 0; i < cleanTokens.length - 1; i++) {
    const w = cleanTokens[i];
    if (KNOWN_NOUNS.has(w) && !TIME_NOUNS.has(w)) {
      addEntity(w, i);
    }
  }

  return { entities, tokenIndices };
}

function findRecentVerb(cleanTokens) {
  for (let i = cleanTokens.length - 1; i >= 0; i--) {
    const w = cleanTokens[i];
    if (VERB_OBJECT_FRAMES[w]) return w;
  }
  return null;
}

function findRecentPreposition(cleanTokens) {
  for (let i = cleanTokens.length - 1; i >= Math.max(0, cleanTokens.length - 3); i--) {
    const w = cleanTokens[i];
    if (PREPOSITION_NOUNS[w]) return w;
  }
  return null;
}

/**
 * Merges context-extracted entities (from earlier in the prompt) with default frame candidates
 * so that e.g. "i have a parrot. every morning i feed my ___" places "parrot" (and "bird") at the top!
 */
function mergeContextEntitiesWithFrame(contextEntities, frameDefaults) {
  const merged = [];
  for (const ent of contextEntities) {
    merged.push(ent);
    if (ENTITY_COMPANIONS[ent]) {
      merged.push(...ENTITY_COMPANIONS[ent]);
    }
  }
  merged.push(...frameDefaults);
  return merged;
}

/**
 * Given an array of input tokens, returns an ordered list of 6 contextually appropriate
 * candidate next tokens along with their raw base logits.
 */
export function predictContextualCandidates(tokens) {
  const cleanTokens = tokens.map(cleanWord).filter(Boolean);
  const n = cleanTokens.length;
  if (n === 0) {
    return buildRankedList(["The", "Every", "In", "Data", "Artificial", "We"]);
  }

  const fullKey = cleanTokens.join(" ");
  if (EXACT_PHRASE_CONTINUATIONS[fullKey]) {
    return buildRankedList(EXACT_PHRASE_CONTINUATIONS[fullKey]);
  }

  const { entities: contextEntities } = extractContextEntities(tokens);
  const last = cleanTokens[n - 1];
  const prev = n >= 2 ? cleanTokens[n - 2] : "";
  const prev2 = n >= 3 ? cleanTokens[n - 3] : "";
  const recentVerb = findRecentVerb(cleanTokens);
  const recentPrep = findRecentPreposition(cleanTokens);
  const hasMorningContext =
    cleanTokens.includes("morning") ||
    cleanTokens.includes("wake") ||
    cleanTokens.includes("breakfast");

  // 1. After "every" / "each" -> time / routine nouns
  if (last === "every" || last === "each") {
    return buildRankedList(["morning", "day", "evening", "night", "weekend", "week"]);
  }

  // 2. Time phrase opener: "every morning", "in the morning", "every day"
  if (
    (prev === "every" && (last === "morning" || last === "day" || last === "evening" || last === "night")) ||
    (prev2 === "in" && prev === "the" && (last === "morning" || last === "evening"))
  ) {
    return buildRankedList(["I", "we", "the", "she", "he", "my"]);
  }

  // 3. After Possessive Pronoun ("my", "your", "his", "her", "our", "their")
  //    Combine any entity introduced earlier in the prompt (e.g. "parrot") with the verb's object frame!
  if (POSSESSIVE_WORDS.has(last)) {
    const baseFrame =
      recentVerb && VERB_OBJECT_FRAMES[recentVerb]
        ? VERB_OBJECT_FRAMES[recentVerb]
        : hasMorningContext
        ? ["coffee", "breakfast", "cat", "dog", "teeth", "bag"]
        : ["cat", "dog", "friend", "family", "coffee", "car", "house", "phone"];

    return buildRankedList(mergeContextEntitiesWithFrame(contextEntities, baseFrame));
  }

  // 4. After Article ("the", "a", "an")
  if (last === "the" || last === "a" || last === "an") {
    if (PREPOSITION_NOUNS[prev]) {
      return buildRankedList(PREPOSITION_NOUNS[prev]);
    }
    if (recentVerb && VERB_OBJECT_FRAMES[recentVerb] && prev === recentVerb) {
      const defaults = VERB_OBJECT_FRAMES[recentVerb];
      // If "the" is used ("i have a parrot. i feed the ___"), also prioritize context entities!
      const list = last === "the" ? mergeContextEntitiesWithFrame(contextEntities, defaults) : defaults;
      return buildRankedList(list);
    }
    if (prev === "have" || prev === "has" || prev === "had" || prev === "got" || prev === "adopted" || prev === "bought") {
      return buildRankedList(["parrot", "cat", "dog", "puppy", "kitten", "car", "book", "house"]);
    }
    if (n === 1) {
      return buildRankedList(["cat", "dog", "sun", "morning", "model", "system", "world"]);
    }
    return buildRankedList(
      mergeContextEntitiesWithFrame(contextEntities, ["cat", "dog", "morning", "house", "table", "park", "warm", "new"])
    );
  }

  // 5. After an Adjective ("warm", "sunny", "complex", "new", "fresh", "hot", "soft", "quiet", "stormy", "dark")
  if (ADJECTIVES.has(last)) {
    if (last === "stormy" || last === "dark" || last === "rainy" || last === "windy" || last === "foggy") {
      return buildRankedList(["night", "evening", "sky", "day", "morning", "road"]);
    }
    if (last === "complex" || last === "new" || last === "important" || last === "hidden" || last === "large") {
      return buildRankedList(["models", "patterns", "systems", "data", "ideas", "problems"]);
    }
    if (last === "hot" || last === "fresh") {
      return buildRankedList(["coffee", "tea", "breakfast", "bread", "food", "water"]);
    }
    if (recentPrep && PREPOSITION_NOUNS[recentPrep]) {
      return buildRankedList(PREPOSITION_NOUNS[recentPrep]);
    }
    return buildRankedList(
      mergeContextEntitiesWithFrame(contextEntities, ["mat", "rug", "couch", "morning", "room", "bed", "floor", "chair"])
    );
  }

  // 6. After 1st/2nd/Plural Subject Pronouns ("i", "we", "you", "they")
  if (SUBJECT_PRONOUNS_PLURAL_OR_1ST.has(last)) {
    if (hasMorningContext) {
      return buildRankedList(["feed", "drink", "eat", "make", "walk", "brush", "go"]);
    }
    return buildRankedList(["have", "like", "love", "feed", "drink", "eat", "read", "learn"]);
  }

  // 7. After 3rd-person singular Pronouns ("he", "she", "it")
  if (SUBJECT_PRONOUNS_3RD_SING.has(last)) {
    if (hasMorningContext) {
      return buildRankedList(["feeds", "drinks", "eats", "walks", "makes", "goes", "wakes"]);
    }
    return buildRankedList(["likes", "loves", "is", "runs", "sleeps", "eats", "sits", "works"]);
  }

  // 8. After Modal / Auxiliary ("can", "will", "would", "should", "must", "do", "did")
  if (MODAL_AUX.has(last)) {
    return buildRankedList(["learn", "explore", "help", "build", "create", "make", "solve", "see"]);
  }

  // 9. After Possession / Acquisition Verbs ("have", "has", "had", "bought", "adopted", "got")
  if (last === "have" || last === "has" || last === "had" || last === "bought" || last === "adopted") {
    return buildRankedList(["a", "two", "some", "my", "the", "many"]);
  }

  // 10. After Action Verbs ("feed", "drink", "eat", "walk", "read", "watch", "brush", "drive", ...)
  if (VERB_OBJECT_FRAMES[last]) {
    const objList = mergeContextEntitiesWithFrame(contextEntities, VERB_OBJECT_FRAMES[last]);
    if (last === "feed" || last === "fed" || last === "walk" || last === "walked" || last === "brush" || last === "pet") {
      return buildRankedList(["my", "the", ...objList.slice(0, 4)]);
    }
    if (last === "drink" || last === "eat" || last === "make" || last === "cook" || last === "pour") {
      return buildRankedList([...objList.slice(0, 4), "my", "some", "hot", "fresh"]);
    }
    return buildRankedList(["the", "my", "a", ...objList.slice(0, 4)]);
  }

  // 11. After "go" / "went" / "walk" / "head"
  if (last === "go" || last === "went" || last === "goes" || last === "headed") {
    return buildRankedList(["to", "outside", "home", "inside", "back", "for"]);
  }

  // 12. After "to" (Infinitive vs Preposition after motion verb)
  if (last === "to") {
    if (prev === "go" || prev === "went" || prev === "goes" || prev === "walk" || prev === "drive" || prev === "drove") {
      return buildRankedList(["work", "school", "bed", "the", "home", "class"]);
    }
    return buildRankedList(["explore", "learn", "understand", "build", "create", "help", "make", "eat"]);
  }

  // 13. After Other Prepositions ("on", "in", "at", "under", "near", "before", "after", "with")
  if (PREPOSITIONS.has(last)) {
    if (last === "before" || last === "after") {
      return buildRankedList(["breakfast", "work", "school", "dinner", "going", "leaving", "I"]);
    }
    if (last === "with") {
      return buildRankedList(["my", "the", "a", "some", "seeds", "friends", "care"]);
    }
    return buildRankedList(["the", "my", "a", ...(PREPOSITION_NOUNS[last] || ["home", "work", "morning"]).slice(0, 4)]);
  }

  // 14. After Conjunction ("and", "then", "while", "because", "so")
  if (last === "and" || last === "then") {
    if (prev === "dark") {
      return buildRankedList(["stormy", "quiet", "cold", "windy", "silent", "rainy"]);
    }
    if (ADJECTIVES.has(prev)) {
      return buildRankedList(["sunny", "warm", "quiet", "bright", "clear", "peaceful"]);
    }
    if (recentVerb === "feed" || recentVerb === "fed") {
      return buildRankedList(["clean", "give", "water", "talk", "play", "drink", "eat"]);
    }
    if (hasMorningContext) {
      return buildRankedList(["drink", "eat", "make", "head", "walk", "read", "check"]);
    }
    return buildRankedList(["create", "build", "explore", "learn", "then", "I", "we"]);
  }

  // 15. After "some" (e.g. "i have a parrot. every morning i feed my parrot some ___")
  if (last === "some") {
    if (cleanTokens.includes("parrot") || cleanTokens.includes("bird") || cleanTokens.includes("birds")) {
      return buildRankedList(["seeds", "fruit", "nuts", "berries", "water", "treats"]);
    }
    if (recentVerb === "feed" || recentVerb === "fed" || cleanTokens.includes("cat") || cleanTokens.includes("dog")) {
      return buildRankedList(["food", "treats", "milk", "fish", "water", "kibble"]);
    }
    return buildRankedList(["coffee", "food", "water", "tea", "fruit", "time"]);
  }

  // 16. After a Known Noun OR a Context Entity Noun (e.g. "i have a parrot. every morning i feed my parrot ___")
  if (KNOWN_NOUNS.has(last) || contextEntities.includes(last) || DETERMINERS.has(prev)) {
    if (n <= 3 && !recentVerb) {
      if (last === "parrot" || last === "bird" || last === "canary") {
        return buildRankedList(["sings", "talks", "flew", "sat", "chirps", "loves", "is"]);
      }
      if (last === "cat" || last === "dog" || last === "kitten" || last === "puppy" || last === "pet") {
        return buildRankedList(["sat", "slept", "ran", "jumped", "played", "ate", "is"]);
      }
      return buildRankedList(["is", "was", "can", "will", "helps", "makes", "works"]);
    }
    if (last === "parrot" || last === "bird" || last === "birds") {
      return buildRankedList(["some", "seeds", "fruit", "and", "before", "every", "in"]);
    }
    if (last === "cat" || last === "dog" || last === "pets" || last === "fish" || contextEntities.includes(last)) {
      return buildRankedList(["and", "some", "before", "in", "every", "with", "fresh"]);
    }
    if (last === "coffee" || last === "tea" || last === "breakfast" || last === "lunch" || last === "dinner") {
      return buildRankedList(["before", "in", "and", "with", "at", "while", "every"]);
    }
    return buildRankedList(["and", "in", "with", "before", "on", "to", "every"]);
  }

  // 17. General fallback
  if (KNOWN_VERBS.has(last)) {
    return buildRankedList(["the", "my", "a", "and", "in", "to"]);
  }
  return buildRankedList(["and", "the", "to", "in", "with", "for"]);
}

function buildRankedList(words) {
  const baseLogits = [3.65, 2.75, 2.20, 1.75, 1.35, 0.95, 0.55, 0.20];
  const unique = [];
  const seen = new Set();
  for (const w of words) {
    if (!seen.has(w.toLowerCase())) {
      seen.add(w.toLowerCase());
      unique.push(w);
    }
  }
  return unique.slice(0, 6).map((token, i) => ({
    token,
    baseLogit: baseLogits[i] ?? 0.1,
  }));
}
