/**
 * harper-fabric-embeddings
 *
 * Minimal llama.cpp embedding wrapper for Harper Fabric.
 * Talks directly to the native N-API addon — no build tools,
 * no CLI, no chat wrappers, no model downloaders.
 *
 * ~19 MB installed (native binary only) vs ~250 MB+ for node-llama-cpp.
 */

import { createRequire } from "node:module";
import { createWriteStream, existsSync, mkdirSync, readdirSync } from "node:fs";
import { pipeline } from "node:stream/promises";
import path from "node:path";

const require = createRequire(import.meta.url);

// ─── Model registry ─────────────────────────────────────────────────────────

const MODELS = {
  "nomic-embed-text": {
    repo: "nomic-ai/nomic-embed-text-v1.5-GGUF",
    file: "nomic-embed-text-v1.5.Q4_K_M.gguf",
  },
  "nomic-embed-text-v2-moe": {
    repo: "nomic-ai/nomic-embed-text-v2-moe-GGUF",
    file: "nomic-embed-text-v2-moe.Q4_K_M.gguf",
  },
};

// ─── State ──────────────────────────────────────────────────────────────────

let binding = null;
let model = null;
let context = null;
let bosToken = -1;
let eosToken = -1;
let disposed = false;

// ─── Public API ─────────────────────────────────────────────────────────────

/**
 * Initialize the embedding engine.
 *
 * Provide either `modelPath` (absolute path to a .gguf file) or
 * `modelsDir` (directory to search/download into) + optional `modelName`.
 *
 * @param {object} options
 * @param {string} [options.modelPath]        - Absolute path to a .gguf model file.
 * @param {string} [options.modelsDir]        - Directory containing (or to download to) model files.
 * @param {string} [options.modelName="nomic-embed-text"] - Model name from the registry.
 * @param {number} [options.contextSize=2048] - Token context window size.
 * @param {number} [options.batchSize=512]    - Batch processing size.
 * @param {number} [options.threads=6]        - CPU threads for inference.
 * @param {number} [options.gpuLayers=0]      - Layers to offload to GPU (0 = CPU only).
 * @param {string} [options.addonPath]        - Override path to llama-addon.node.
 */
export async function init(options) {
  if (binding) return; // Already initialized

  const {
    modelPath: explicitPath,
    modelsDir,
    modelName = "nomic-embed-text",
    contextSize = 2048,
    batchSize = 512,
    threads = 6,
    gpuLayers = 0,
    addonPath,
  } = options;

  // Resolve model path: explicit path, or find/download in modelsDir
  let modelPath = explicitPath;
  if (!modelPath) {
    if (!modelsDir) throw new Error("Either modelPath or modelsDir is required");
    modelPath = await resolveModelPath(modelsDir, modelName);
  }

  if (!existsSync(modelPath))
    throw new Error(`Model file not found: ${modelPath}`);

  // ── Load native addon ──────────────────────────────────────────────────

  const resolvedAddonPath = addonPath || findAddonBinary();
  binding = loadAddon(resolvedAddonPath);

  await binding.init();

  // Load GPU/CPU backends from the same directory as the addon binary
  const backendsDir = path.dirname(resolvedAddonPath);
  binding.loadBackends();
  binding.loadBackends(backendsDir);

  // ── Load model ─────────────────────────────────────────────────────────

  model = new binding.AddonModel(modelPath, {
    gpuLayers,
    useMmap: true,
    useMlock: false,
    checkTensors: false,
  });

  const modelLoaded = await model.init();
  if (!modelLoaded) throw new Error("Failed to load model");

  // Cache special tokens
  bosToken = model.tokenBos();
  eosToken = model.tokenEos();

  // ── Create embedding context ───────────────────────────────────────────

  context = new binding.AddonContext(model, {
    contextSize,
    batchSize,
    sequences: 1,
    embeddings: true,
    threads,
  });

  const ctxLoaded = await context.init();
  if (!ctxLoaded) throw new Error("Failed to create embedding context");

  disposed = false;
}

/**
 * Generate an embedding vector for the given text.
 *
 * @param {string} text - Input text to embed.
 * @returns {Promise<number[]>} Embedding vector.
 */
export async function embed(text) {
  if (!binding || !model || !context) {
    throw new Error("Not initialized. Call init() first.");
  }
  if (disposed) {
    throw new Error("Engine has been disposed.");
  }

  // Tokenize
  const tokens = model.tokenize(text, false);
  if (tokens.length === 0) return [];

  // Prepend BOS / append EOS if the model uses them
  const input = buildTokenSequence(tokens);

  // Batch evaluate
  context.initBatch(input.length);

  // logitIndexes: indexes of tokens that should produce outputs
  // For embeddings, all tokens must be marked as outputs
  const logitIndexes = new Uint32Array(input.length);
  for (let i = 0; i < input.length; i++) logitIndexes[i] = i;

  context.addToBatch(0, 0, input, logitIndexes);
  await context.decodeBatch();

  // Extract embedding
  const raw = context.getEmbedding(input.length);

  // Normalize to unit vector (L2) for cosine similarity
  return normalize(raw);
}

/**
 * Get the embedding vector dimensionality.
 *
 * @returns {number}
 */
export function dimensions() {
  if (!model) throw new Error("Not initialized. Call init() first.");
  return model.getEmbeddingVectorSize();
}

/**
 * Clean up native resources.
 */
export async function dispose() {
  disposed = true;
  if (context) {
    await context.dispose();
    context = null;
  }
  if (model) {
    await model.dispose();
    model = null;
  }
  if (binding) {
    await binding.dispose();
    binding = null;
  }
  bosToken = -1;
  eosToken = -1;
}

// ─── Internals ──────────────────────────────────────────────────────────────

/**
 * Build the full token sequence with BOS/EOS markers.
 */
function buildTokenSequence(tokens) {
  const parts = [];

  if (bosToken >= 0 && tokens[0] !== bosToken) {
    parts.push(bosToken);
  }

  for (let i = 0; i < tokens.length; i++) {
    parts.push(tokens[i]);
  }

  if (eosToken >= 0 && tokens[tokens.length - 1] !== eosToken) {
    parts.push(eosToken);
  }

  return new Uint32Array(parts);
}

/**
 * L2-normalize a vector to unit length.
 */
function normalize(vec) {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) {
    sumSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return Array.from(vec);

  const out = new Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] / norm;
  }
  return out;
}

// ─── Model resolution & download ────────────────────────────────────────────

/**
 * Find an existing model file or download it.
 *
 * @param {string} dir - Models directory.
 * @param {string} modelName - Key in the MODELS registry.
 * @returns {Promise<string>} Absolute path to the model file.
 */
async function resolveModelPath(dir, modelName) {
  const config = MODELS[modelName];
  if (!config) {
    throw new Error(
      `Unknown model: ${modelName}. Available: ${Object.keys(MODELS).join(", ")}`,
    );
  }

  mkdirSync(dir, { recursive: true });

  // Check for existing file (hf-prefixed name from node-llama-cpp, or bare name)
  const hfName = `hf_${config.repo.replace("/", "_")}_${config.file}`;
  const hfPath = path.join(dir, hfName);
  if (existsSync(hfPath)) return hfPath;

  const barePath = path.join(dir, config.file);
  if (existsSync(barePath)) return barePath;

  // Scan for any matching .gguf
  const stem = config.file.replace(".gguf", "");
  for (const entry of readdirSync(dir)) {
    if (entry.endsWith(".gguf") && entry.includes(stem)) {
      return path.join(dir, entry);
    }
  }

  // Download from Hugging Face
  return downloadModel(dir, modelName);
}

/**
 * Download a model from Hugging Face.
 *
 * @param {string} dir - Directory to save the model file.
 * @param {string} [modelName="nomic-embed-text"] - Model name from registry.
 * @returns {Promise<string>} Absolute path to the downloaded file.
 */
export async function downloadModel(dir, modelName = "nomic-embed-text") {
  const config = MODELS[modelName];
  if (!config) {
    throw new Error(
      `Unknown model: ${modelName}. Available: ${Object.keys(MODELS).join(", ")}`,
    );
  }

  mkdirSync(dir, { recursive: true });

  const url = `https://huggingface.co/${config.repo}/resolve/main/${config.file}`;
  const destPath = path.join(dir, config.file);
  const tmpPath = destPath + ".downloading";

  console.log(`[fabric-llama-embeddings] Downloading ${config.file} from Hugging Face...`);

  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText} — ${url}`);
  }

  // Stream to disk
  const fileStream = createWriteStream(tmpPath);
  await pipeline(response.body, fileStream);

  // Rename to final path (atomic on same filesystem)
  const { rename } = await import("node:fs/promises");
  await rename(tmpPath, destPath);

  console.log(`[fabric-llama-embeddings] Downloaded ${config.file} to ${destPath}`);
  return destPath;
}

/**
 * Find the llama-addon.node binary from installed platform packages.
 */
function findAddonBinary() {
  // Platform-specific package names
  const candidates = [
    "@node-llama-cpp/linux-x64",
    "@node-llama-cpp/mac-arm64-metal",
    "@node-llama-cpp/mac-x64",
    "@node-llama-cpp/linux-arm64",
  ];

  for (const pkg of candidates) {
    try {
      const pkgMain = require.resolve(pkg);
      const pkgDir = path.dirname(pkgMain);

      // Binary lives in ../bins/<folder>/llama-addon.node
      const binsDir = path.join(pkgDir, "..", "bins");
      if (!existsSync(binsDir)) continue;

      // Find the first directory in bins/ that contains the addon
      const entries = require("node:fs").readdirSync(binsDir);
      for (const entry of entries) {
        const addonPath = path.join(binsDir, entry, "llama-addon.node");
        if (existsSync(addonPath)) return addonPath;
      }
    } catch {
      // Package not installed
    }
  }

  throw new Error(
    "No llama-addon.node binary found. Install a @node-llama-cpp platform package " +
      "(e.g., @node-llama-cpp/linux-x64).",
  );
}

/**
 * Load the native addon with cache isolation.
 */
function loadAddon(addonPath) {
  // Clear require cache to ensure fresh native state
  // (multiple Llama instances need independent state)
  try {
    delete require.cache[require.resolve(addonPath)];
  } catch {
    // Not cached
  }

  const addon = require(addonPath);

  try {
    delete require.cache[require.resolve(addonPath)];
  } catch {
    // Ignore
  }

  return addon;
}
