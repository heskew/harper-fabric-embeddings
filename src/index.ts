/**
 * harper-fabric-embeddings
 *
 * Minimal llama.cpp embedding wrapper for Harper Fabric.
 * Talks directly to the native N-API addon — no build tools,
 * no CLI, no chat wrappers, no model downloaders.
 *
 * ~19 MB installed (native binary only) vs ~250 MB+ for node-llama-cpp.
 */

import { createWriteStream, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { rename, unlink } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ModelConfig {
	repo: string;
	file: string;
}

export interface InitOptions {
	/** Absolute path to a .gguf model file. */
	modelPath?: string;
	/** Directory containing (or to download to) model files. */
	modelsDir?: string;
	/** Model name from the built-in registry. */
	modelName?: string;
	/** Token context window size. */
	contextSize?: number;
	/** Batch processing size. */
	batchSize?: number;
	/** CPU threads for inference. */
	threads?: number;
	/** Layers to offload to GPU (0 = CPU only). */
	gpuLayers?: number;
	/** Override path to llama-addon.node. */
	addonPath?: string;
}

/** Native llama.cpp binding loaded from the platform-specific addon. */
interface LlamaBinding {
	init(): Promise<void>;
	dispose(): Promise<void>;
	loadBackends(dir?: string): void;
	AddonModel: new (
		path: string,
		options: {
			gpuLayers: number;
			useMmap: boolean;
			useMlock: boolean;
			checkTensors: boolean;
		}
	) => LlamaModel;
	AddonContext: new (
		model: LlamaModel,
		options: {
			contextSize: number;
			batchSize: number;
			sequences: number;
			embeddings: boolean;
			threads: number;
		}
	) => LlamaContext;
}

/** Native model handle. */
interface LlamaModel {
	init(): Promise<boolean>;
	dispose(): Promise<void>;
	tokenBos(): number;
	tokenEos(): number;
	tokenize(text: string, addSpecial: boolean): Uint32Array;
	getEmbeddingVectorSize(): number;
}

/** Native embedding context handle. */
interface LlamaContext {
	init(): Promise<boolean>;
	dispose(): Promise<void>;
	initBatch(size: number): void;
	addToBatch(seq: number, pos: number, tokens: Uint32Array, logitIndexes: Uint32Array): void;
	decodeBatch(): Promise<void>;
	getEmbedding(tokenCount: number): Float32Array;
}

// ─── Model registry ─────────────────────────────────────────────────────────

const MODELS: Record<string, ModelConfig> = {
	'nomic-embed-text': {
		repo: 'nomic-ai/nomic-embed-text-v1.5-GGUF',
		file: 'nomic-embed-text-v1.5.Q4_K_M.gguf',
	},
	'nomic-embed-text-v2-moe': {
		repo: 'nomic-ai/nomic-embed-text-v2-moe-GGUF',
		file: 'nomic-embed-text-v2-moe.Q4_K_M.gguf',
	},
};

// ─── State ──────────────────────────────────────────────────────────────────

let binding: LlamaBinding | null = null;
let model: LlamaModel | null = null;
let context: LlamaContext | null = null;
let bosToken = -1;
let eosToken = -1;
let disposed = false;
let initPromise: Promise<void> | null = null;

// Serial queue for embed calls — llama.cpp context is not safe for concurrent use
let embedQueue: Promise<unknown> = Promise.resolve();

// ─── Public API ─────────────────────────────────────────────────────────────

/**
 * Initialize the embedding engine.
 *
 * Provide either `modelPath` (absolute path to a .gguf file) or
 * `modelsDir` (directory to search/download into) + optional `modelName`.
 *
 * Safe to call concurrently — concurrent callers share the same initialization.
 */
export async function init(options: InitOptions): Promise<void> {
	if (initPromise) return initPromise;
	initPromise = doInit(options);
	try {
		await initPromise;
	} catch (err) {
		// Allow retry on failure
		initPromise = null;
		throw err;
	}
}

async function doInit(options: InitOptions): Promise<void> {
	if (binding) return; // Already initialized

	const {
		modelPath: explicitPath,
		modelsDir,
		modelName = 'nomic-embed-text',
		contextSize = 2048,
		batchSize = 512,
		threads = 6,
		gpuLayers = 0,
		addonPath,
	} = options;

	// Resolve model path: explicit path, or find/download in modelsDir
	let modelPath = explicitPath;
	if (!modelPath) {
		if (!modelsDir) throw new Error('Either modelPath or modelsDir is required');
		modelPath = await resolveModelPath(modelsDir, modelName);
	}

	if (!existsSync(modelPath)) throw new Error(`Model file not found: ${modelPath}`);

	// ── Load native addon ──────────────────────────────────────────────────

	const resolvedAddonPath = addonPath || findAddonBinary();
	const b = loadAddon(resolvedAddonPath);

	await b.init();

	// Load GPU/CPU backends from the same directory as the addon binary
	const backendsDir = path.dirname(resolvedAddonPath);
	b.loadBackends();
	b.loadBackends(backendsDir);

	// ── Load model ─────────────────────────────────────────────────────────

	const m = new b.AddonModel(modelPath, {
		gpuLayers,
		useMmap: true,
		useMlock: false,
		checkTensors: false,
	});

	const modelLoaded = await m.init();
	if (!modelLoaded) throw new Error('Failed to load model');

	// Cache special tokens
	bosToken = m.tokenBos();
	eosToken = m.tokenEos();

	// ── Create embedding context ───────────────────────────────────────────

	const ctx = new b.AddonContext(m, {
		contextSize,
		batchSize,
		sequences: 1,
		embeddings: true,
		threads,
	});

	const ctxLoaded = await ctx.init();
	if (!ctxLoaded) throw new Error('Failed to create embedding context');

	// Commit state only after everything succeeds
	binding = b;
	model = m;
	context = ctx;
	disposed = false;
}

/**
 * Generate an embedding vector for the given text.
 *
 * Calls are serialized internally — concurrent callers wait in queue
 * rather than hitting the llama.cpp context simultaneously.
 */
export function embed(text: string): Promise<number[]> {
	const result = embedQueue.then(() => {
		assertReady();
		return embedOne(text);
	});
	embedQueue = result.catch(() => {});
	return result;
}

/**
 * Generate embedding vectors for multiple texts.
 *
 * More efficient than calling embed() in a loop — texts are processed
 * sequentially through the native context without queue overhead per item.
 */
export function embedBatch(texts: string[]): Promise<number[][]> {
	const result = embedQueue.then(async () => {
		assertReady();
		const results: number[][] = [];
		for (const text of texts) {
			results.push(await embedOne(text));
		}
		return results;
	});
	embedQueue = result.catch(() => {});
	return result;
}

function assertReady(): void {
	if (!binding || !model || !context) {
		throw new Error('Not initialized. Call init() first.');
	}
	if (disposed) {
		throw new Error('Engine has been disposed.');
	}
}

/**
 * Internal: generate a single embedding (must be called within the serial queue).
 */
async function embedOne(text: string): Promise<number[]> {
	const tokens: Uint32Array = model!.tokenize(text, false);
	if (tokens.length === 0) return [];

	const input = buildTokenSequence(tokens);
	context!.initBatch(input.length);

	const logitIndexes = new Uint32Array(input.length);
	for (let i = 0; i < input.length; i++) logitIndexes[i] = i;

	context!.addToBatch(0, 0, input, logitIndexes);
	await context!.decodeBatch();

	return normalize(context!.getEmbedding(input.length));
}

/**
 * Get the embedding vector dimensionality.
 */
export function dimensions(): number {
	if (!model) throw new Error('Not initialized. Call init() first.');
	return model.getEmbeddingVectorSize();
}

/**
 * Clean up native resources.
 */
export async function dispose(): Promise<void> {
	disposed = true;
	initPromise = null;
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
function buildTokenSequence(tokens: Uint32Array): Uint32Array {
	const parts: number[] = [];

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
 * L2-normalize a Float32Array in-place and return as number[].
 *
 * Normalizes in the typed array first (avoiding a second allocation),
 * then converts to number[] for consumer compatibility.
 */
function normalize(vec: Float32Array): number[] {
	let sumSq = 0;
	for (let i = 0; i < vec.length; i++) {
		sumSq += vec[i] * vec[i];
	}
	const norm = Math.sqrt(sumSq);
	if (norm === 0) return Array.from(vec);

	for (let i = 0; i < vec.length; i++) {
		vec[i] /= norm;
	}
	return Array.from(vec);
}

// ─── Model resolution & download ────────────────────────────────────────────

/**
 * Find an existing model file or download it.
 */
async function resolveModelPath(dir: string, modelName: string): Promise<string> {
	const config = MODELS[modelName];
	if (!config) {
		throw new Error(`Unknown model: ${modelName}. Available: ${Object.keys(MODELS).join(', ')}`);
	}

	mkdirSync(dir, { recursive: true });

	// Check for existing file (hf-prefixed name from node-llama-cpp, or bare name)
	const hfName = `hf_${config.repo.replace('/', '_')}_${config.file}`;
	const hfPath = path.join(dir, hfName);
	if (existsSync(hfPath)) return hfPath;

	const barePath = path.join(dir, config.file);
	if (existsSync(barePath)) return barePath;

	// Scan for any matching .gguf
	const stem = config.file.replace('.gguf', '');
	for (const entry of readdirSync(dir)) {
		if (entry.endsWith('.gguf') && entry.includes(stem)) {
			return path.join(dir, entry);
		}
	}

	// Download from Hugging Face
	return downloadModel(dir, modelName);
}

/**
 * Download a model from Hugging Face.
 */
export async function downloadModel(dir: string, modelName = 'nomic-embed-text'): Promise<string> {
	const config = MODELS[modelName];
	if (!config) {
		throw new Error(`Unknown model: ${modelName}. Available: ${Object.keys(MODELS).join(', ')}`);
	}

	mkdirSync(dir, { recursive: true });

	const url = `https://huggingface.co/${config.repo}/resolve/main/${config.file}`;
	const destPath = path.join(dir, config.file);
	const tmpPath = destPath + '.downloading';

	console.log(`[harper-fabric-embeddings] Downloading ${config.file} from Hugging Face...`);

	const response = await fetch(url, { redirect: 'follow' });
	if (!response.ok) {
		throw new Error(`Download failed: ${response.status} ${response.statusText} — ${url}`);
	}

	// Stream to disk, clean up temp file on failure
	try {
		const fileStream = createWriteStream(tmpPath);
		await pipeline(response.body!, fileStream);
	} catch (err) {
		await unlink(tmpPath).catch(() => {});
		throw err;
	}

	// Rename to final path (atomic on same filesystem)
	await rename(tmpPath, destPath);

	console.log(`[harper-fabric-embeddings] Downloaded ${config.file} to ${destPath}`);
	return destPath;
}

/**
 * Find the llama-addon.node binary from installed platform packages.
 *
 * Scans node_modules on the filesystem rather than using require.resolve,
 * since Harper's sandbox blocks node:module.
 */
function findAddonBinary(): string {
	const candidates = [
		'@node-llama-cpp/linux-x64',
		'@node-llama-cpp/mac-arm64-metal',
		'@node-llama-cpp/mac-x64',
		'@node-llama-cpp/linux-arm64',
	];

	const searchRoots = [
		path.join(process.cwd(), 'node_modules'),
		path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'node_modules'),
	];

	console.log(
		`[harper-fabric-embeddings] findAddonBinary: cwd=${process.cwd()}, searching ${searchRoots.length} roots`
	);

	for (const nmDir of searchRoots) {
		if (!existsSync(nmDir)) {
			console.log(`[harper-fabric-embeddings] findAddonBinary: skip ${nmDir} (not found)`);
			continue;
		}
		for (const pkg of candidates) {
			const binsDir = path.join(nmDir, pkg, 'bins');
			if (!existsSync(binsDir)) {
				console.log(`[harper-fabric-embeddings] findAddonBinary: skip ${binsDir} (not found)`);
				continue;
			}

			for (const entry of readdirSync(binsDir)) {
				const addonPath = path.join(binsDir, entry, 'llama-addon.node');
				if (existsSync(addonPath)) {
					console.log(`[harper-fabric-embeddings] findAddonBinary: found ${addonPath}`);
					return addonPath;
				}
			}
		}
	}

	throw new Error(
		'No llama-addon.node binary found. Install a @node-llama-cpp platform package ' +
			'(e.g., @node-llama-cpp/linux-x64).'
	);
}

/**
 * Load the native addon via process.dlopen.
 *
 * Each call gets an independent instance — no require cache to worry about.
 */
function loadAddon(addonPath: string): LlamaBinding {
	const mod = { exports: {} } as { exports: LlamaBinding };
	process.dlopen(mod, addonPath);
	return mod.exports;
}
