/**
 * Embedding engine internals.
 *
 * `EmbeddingEngine` owns one GGUF model + llama.cpp context and serializes
 * embed calls against it. Multiple engines can coexist in one process — one
 * per registered Harper model entry — so the native addon binding is shared
 * through a refcounted registry keyed by addon path (dlopen of the same path
 * returns the same handle; llama.cpp keeps all state in heap objects, so
 * engines stay independent through the shared native code).
 */

import { createWriteStream, existsSync, mkdirSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import { open, rename, unlink } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ModelConfig {
	repo: string;
	file: string;
}

export interface EngineOptions {
	/** Absolute path to a .gguf model file. */
	modelPath?: string;
	/** Directory containing (or to download to) model files. */
	modelsDir?: string;
	/** Model name from the built-in registry. */
	modelName?: string;
	/** Token context window size. */
	contextSize?: number;
	/**
	 * Batch processing size. In node-llama-cpp this sets both `n_batch` AND
	 * `n_ubatch` on the llama.cpp context. llama.cpp's encoder asserts
	 * `n_ubatch >= n_tokens` for every input — inputs that tokenize above
	 * `batchSize` trigger `GGML_ASSERT` → `ggml_abort`, killing the host
	 * process. Defaults to `contextSize` so the full context window is
	 * usable out of the box.
	 */
	batchSize?: number;
	/** CPU threads for inference. */
	threads?: number;
	/** Layers to offload to GPU (0 = CPU only). */
	gpuLayers?: number;
	/** Override path to llama-addon.node. */
	addonPath?: string;
}

export interface EmbedManyOptions {
	/**
	 * For models that distinguish document-vs-query embeddings. Applies the
	 * model's task prefix (`search_document: ` / `search_query: ` for
	 * nomic-embed-text); ignored for models without a known prefix convention.
	 */
	inputType?: 'document' | 'query';
	/** Best-effort cancellation — checked between inputs, not mid-decode. */
	signal?: AbortSignal;
}

export interface EmbedManyResult {
	/** One L2-normalized vector per input, in input order. */
	vectors: Float32Array[];
	/** Total tokens decoded across all inputs (including BOS/EOS). */
	tokens: number;
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

// ─── Shared native binding registry ─────────────────────────────────────────

interface BindingEntry {
	promise: Promise<LlamaBinding>;
	refs: number;
}

// One dlopen + backend load per addon path, shared across engines. Refcounted
// so the last engine's dispose() tears the addon down (matching the pre-engine
// module behavior); a failed load is evicted so the next acquire retries.
//
// Not atomic across the dispose await: an acquire racing the final release can
// dlopen a handle the in-flight dispose is tearing down. Accepted — it needs
// two engines on one addon path with one disposing while the other inits, and
// the models-backend path never disposes engines today.
const bindings = new Map<string, BindingEntry>();

function acquireBinding(addonPath: string): Promise<LlamaBinding> {
	let entry = bindings.get(addonPath);
	if (!entry) {
		const created: BindingEntry = { refs: 0, promise: loadBinding(addonPath) };
		created.promise.catch(() => {
			if (bindings.get(addonPath) === created) bindings.delete(addonPath);
		});
		bindings.set(addonPath, created);
		entry = created;
	}
	entry.refs++;
	return entry.promise;
}

async function releaseBinding(addonPath: string): Promise<void> {
	const entry = bindings.get(addonPath);
	if (!entry) return;
	if (--entry.refs > 0) return;
	bindings.delete(addonPath);
	try {
		const binding = await entry.promise;
		await binding.dispose();
	} catch {
		// Load failed — nothing to dispose.
	}
}

async function loadBinding(addonPath: string): Promise<LlamaBinding> {
	const binding = loadAddon(addonPath);
	await binding.init();
	// Load GPU/CPU backends from the same directory as the addon binary
	binding.loadBackends();
	binding.loadBackends(path.dirname(addonPath));
	return binding;
}

// ─── Engine ─────────────────────────────────────────────────────────────────

/**
 * One GGUF model + llama.cpp embedding context.
 *
 * - Construction validates configuration synchronously (missing model source,
 *   unknown model name) so misconfiguration fails at registration, not first use.
 * - Heavy work (download, addon + model load) happens in `ensureReady()`, which
 *   is lazy, shared across concurrent callers, and retryable after failure.
 * - `embedMany()` serializes against a per-engine queue — the llama.cpp context
 *   is not safe for concurrent use.
 */
export class EmbeddingEngine {
	#options: EngineOptions;
	#modelIdentity: string;
	#binding: LlamaBinding | null = null;
	#addonPathUsed: string | null = null;
	#model: LlamaModel | null = null;
	#context: LlamaContext | null = null;
	#bosToken = -1;
	#eosToken = -1;
	// Effective n_ubatch the live context was created with. embedOne truncates
	// against this so oversized inputs return a (shorter) embedding instead of
	// tripping the llama.cpp GGML_ASSERT that kills the host process.
	#maxInputTokens = 0;
	#truncationWarned = false;
	#disposed = false;
	// Set at dispose() entry, before the queue drain. Embeds already queued on an
	// initialized engine complete; anything needing a fresh init (or submitted
	// after) rejects instead of starting native work during shutdown.
	#disposeStarted = false;
	#initPromise: Promise<void> | null = null;
	// Serial queue for embed calls — llama.cpp context is not safe for concurrent use
	#queue: Promise<unknown> = Promise.resolve();

	constructor(options: EngineOptions) {
		if (!options.modelPath && !options.modelsDir) {
			throw new Error('Either modelPath or modelsDir is required');
		}
		const modelName = options.modelName ?? 'nomic-embed-text';
		if (!options.modelPath && !MODELS[modelName]) {
			throw new Error(`Unknown model: ${modelName}. Available: ${Object.keys(MODELS).join(', ')}`);
		}
		// The truncation guard in #embedOne reserves 2 slots for BOS/EOS; an
		// effective batch below 3 can still build a 3-token sequence and trip the
		// llama.cpp GGML_ASSERT that kills the host process. Reject up front.
		const effectiveBatchSize = options.batchSize ?? options.contextSize ?? 2048;
		if (effectiveBatchSize < 3) {
			throw new Error(`batchSize must be at least 3 (one body token plus BOS/EOS), got ${effectiveBatchSize}`);
		}
		this.#options = options;
		this.#modelIdentity = options.modelPath ? path.basename(options.modelPath) : modelName;
	}

	/** Model name (or model file basename) — used for backend naming and prefix detection. */
	get modelIdentity(): string {
		return this.#modelIdentity;
	}

	/**
	 * Kick off (or join) initialization. Lazy and retryable: a failed attempt
	 * resets so the next call tries again (e.g. a transient download failure).
	 */
	ensureReady(): Promise<void> {
		if (this.#initPromise) return this.#initPromise;
		const attempt = this.#doInit();
		this.#initPromise = attempt;
		attempt.catch(() => {
			if (this.#initPromise === attempt) this.#initPromise = null;
		});
		return attempt;
	}

	async #doInit(): Promise<void> {
		if (this.#disposed || this.#disposeStarted) throw new Error('Engine has been disposed.');

		const {
			modelPath: explicitPath,
			modelsDir,
			modelName = 'nomic-embed-text',
			contextSize = 2048,
			batchSize: explicitBatchSize,
			threads = 6,
			gpuLayers = 0,
			addonPath,
		} = this.#options;

		// node-llama-cpp's AddonContext ties n_batch and n_ubatch to `batchSize`,
		// and llama.cpp aborts (GGML_ASSERT -> ggml_abort, kills the host process)
		// when any input tokenizes above n_ubatch. Default to contextSize so the
		// full declared context window is actually embeddable.
		const batchSize = explicitBatchSize ?? contextSize;

		// Resolve the model BEFORE touching the addon: the common misconfiguration
		// (bad path, missing file) never churns the shared binding refcount.
		let modelPath = explicitPath;
		if (!modelPath) {
			modelPath = await resolveModelPath(modelsDir!, modelName);
		}
		if (!existsSync(modelPath)) throw new Error(`Model file not found: ${modelPath}`);

		const resolvedAddonPath = addonPath || findAddonBinary();
		const binding = await acquireBinding(resolvedAddonPath);
		let model: LlamaModel | null = null;
		let context: LlamaContext | null = null;
		try {
			model = new binding.AddonModel(modelPath, {
				gpuLayers,
				useMmap: true,
				useMlock: false,
				checkTensors: false,
			});
			if (!(await model.init())) {
				// Dispose before nulling: a false init() may still hold native
				// resources, and the catch below skips dispose for a null model.
				await model.dispose().catch(() => {});
				model = null;
				throw new Error('Failed to load model');
			}
			context = new binding.AddonContext(model, {
				contextSize,
				batchSize,
				sequences: 1,
				embeddings: true,
				threads,
			});
			if (!(await context.init())) {
				context = null;
				throw new Error('Failed to create embedding context');
			}
			if (this.#disposed) throw new Error('Engine has been disposed.');
		} catch (err) {
			if (context) await context.dispose().catch(() => {});
			if (model) await model.dispose().catch(() => {});
			await releaseBinding(resolvedAddonPath);
			throw err;
		}

		// Commit state only after everything succeeds
		this.#binding = binding;
		this.#addonPathUsed = resolvedAddonPath;
		this.#model = model;
		this.#context = context;
		this.#bosToken = model.tokenBos();
		this.#eosToken = model.tokenEos();
		this.#maxInputTokens = batchSize;
	}

	/**
	 * Generate one L2-normalized vector per input, in input order.
	 *
	 * Serialized against the engine's queue; init is awaited lazily on first
	 * call. `opts.signal` is checked between inputs (best-effort — a decode in
	 * flight can't be interrupted).
	 */
	embedMany(texts: string[], opts: EmbedManyOptions = {}): Promise<EmbedManyResult> {
		// Fail an already-aborted call synchronously instead of parking it in the queue.
		opts.signal?.throwIfAborted();
		const result = this.#queue.then(async () => {
			await this.ensureReady();
			this.#assertReady();
			const vectors: Float32Array[] = [];
			let tokens = 0;
			for (const text of texts) {
				opts.signal?.throwIfAborted();
				const one = await this.#embedOne(this.#applyPrefix(text, opts.inputType));
				vectors.push(one.vector);
				tokens += one.tokens;
			}
			return { vectors, tokens };
		});
		this.#queue = result.catch(() => {});
		return result;
	}

	/** Get the embedding vector dimensionality. */
	dimensions(): number {
		if (!this.#model) throw new Error('Not initialized. Call init() first.');
		return this.#model.getEmbeddingVectorSize();
	}

	/**
	 * Clean up native resources.
	 *
	 * Drains the embed queue before touching native handles — disposing the
	 * llama.cpp context while a decodeBatch is executing is a use-after-free
	 * that can kill the host process. Embeds accepted before this call complete
	 * (when the engine is initialized); embeds that would need a fresh init
	 * during disposal, and any submitted after, reject with a disposed error.
	 */
	async dispose(): Promise<void> {
		this.#disposeStarted = true;
		await this.#queue.catch(() => {});
		this.#disposed = true;
		this.#initPromise = null;
		if (this.#context) {
			await this.#context.dispose();
			this.#context = null;
		}
		if (this.#model) {
			await this.#model.dispose();
			this.#model = null;
		}
		if (this.#binding) {
			this.#binding = null;
			const addonPath = this.#addonPathUsed;
			this.#addonPathUsed = null;
			if (addonPath) await releaseBinding(addonPath);
		}
		this.#bosToken = -1;
		this.#eosToken = -1;
		this.#maxInputTokens = 0;
		this.#truncationWarned = false;
	}

	#assertReady(): void {
		if (this.#disposed) throw new Error('Engine has been disposed.');
		if (!this.#binding || !this.#model || !this.#context) {
			throw new Error('Not initialized. Call init() first.');
		}
	}

	#applyPrefix(text: string, inputType?: 'document' | 'query'): string {
		if (!inputType) return text;
		// nomic-embed-text v1.5+ uses application-layer task prefixes to distinguish
		// document-corpus encodings from query encodings. Mirrors the convention in
		// Harper's built-in ollama backend; other model families (BGE, e5, ...) use
		// their own conventions — add cases as we validate them.
		if (/nomic-embed-text/i.test(this.#modelIdentity)) {
			return (inputType === 'document' ? 'search_document: ' : 'search_query: ') + text;
		}
		return text;
	}

	/** Generate a single embedding (must be called within the serial queue). */
	async #embedOne(text: string): Promise<{ vector: Float32Array; tokens: number }> {
		let tokens: Uint32Array = this.#model!.tokenize(text, false);
		if (tokens.length === 0) return { vector: new Float32Array(0), tokens: 0 };

		// Belt-and-suspenders: reserve 2 slots for BOS/EOS and truncate anything
		// that would exceed the live context's n_ubatch. Prevents GGML_ASSERT when
		// callers pass an explicit `batchSize` smaller than their real inputs.
		const maxBodyTokens = Math.max(1, this.#maxInputTokens - 2);
		if (tokens.length > maxBodyTokens) {
			if (!this.#truncationWarned) {
				console.warn(
					`[harper-fabric-embeddings] Input tokenized to ${tokens.length} tokens, ` +
						`truncating to ${maxBodyTokens} (batchSize=${this.#maxInputTokens}). ` +
						`Increase batchSize/contextSize if you need longer inputs embedded in full.`
				);
				this.#truncationWarned = true;
			}
			tokens = tokens.subarray(0, maxBodyTokens);
		}

		const input = this.#buildTokenSequence(tokens);
		this.#context!.initBatch(input.length);

		const logitIndexes = new Uint32Array(input.length);
		for (let i = 0; i < input.length; i++) logitIndexes[i] = i;

		this.#context!.addToBatch(0, 0, input, logitIndexes);
		await this.#context!.decodeBatch();

		// Copy before normalizing — don't assume the addon's returned view stays
		// stable across the next decode.
		const vector = Float32Array.from(this.#context!.getEmbedding(input.length));
		l2NormalizeInPlace(vector);
		return { vector, tokens: input.length };
	}

	/** Build the full token sequence with BOS/EOS markers. */
	#buildTokenSequence(tokens: Uint32Array): Uint32Array {
		const parts: number[] = [];

		if (this.#bosToken >= 0 && tokens[0] !== this.#bosToken) {
			parts.push(this.#bosToken);
		}

		for (let i = 0; i < tokens.length; i++) {
			parts.push(tokens[i]);
		}

		if (this.#eosToken >= 0 && tokens[tokens.length - 1] !== this.#eosToken) {
			parts.push(this.#eosToken);
		}

		return new Uint32Array(parts);
	}
}

/** L2-normalize in place. A zero vector is left unchanged. */
function l2NormalizeInPlace(vec: Float32Array): void {
	let sumSq = 0;
	for (let i = 0; i < vec.length; i++) {
		sumSq += vec[i] * vec[i];
	}
	const norm = Math.sqrt(sumSq);
	if (norm === 0) return;
	for (let i = 0; i < vec.length; i++) {
		vec[i] /= norm;
	}
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
 * A `.downloading` lock untouched for this long is considered abandoned and is
 * reclaimed. An active download keeps the lock file's mtime fresh — the bytes
 * stream into it — so only a dead owner (SIGKILL / OOM / power loss mid-download)
 * leaves it unmoving.
 */
const LOCK_STALE_MS = 60_000;

/** Total time a worker will wait on other workers before giving up. */
const DOWNLOAD_WAIT_TIMEOUT_MS = 300_000;

/**
 * Download a model from Hugging Face.
 *
 * Cross-worker coordination: exclusive creation of `<file>.downloading` elects
 * one downloader; the rest poll. A lock whose owner died is reclaimed after
 * `LOCK_STALE_MS`, and a waiter whose winner failed (lock vanished, no final
 * file) retakes the lock and retries rather than timing out.
 */
export async function downloadModel(dir: string, modelName = 'nomic-embed-text'): Promise<string> {
	const config = MODELS[modelName];
	if (!config) {
		throw new Error(`Unknown model: ${modelName}. Available: ${Object.keys(MODELS).join(', ')}`);
	}

	mkdirSync(dir, { recursive: true });

	const destPath = path.join(dir, config.file);

	// Already downloaded
	if (existsSync(destPath)) return destPath;

	const tmpPath = destPath + '.downloading';
	const deadline = Date.now() + DOWNLOAD_WAIT_TIMEOUT_MS;

	while (true) {
		reclaimStaleLock(tmpPath);

		// Try to exclusively create the temp file — only one worker wins
		let lockHandle;
		try {
			lockHandle = await open(tmpPath, 'wx');
		} catch {
			// Another worker is downloading — wait for it to finish or fail
			console.log(`[harper-fabric-embeddings] Waiting for another worker to finish downloading ${config.file}...`);
			const outcome = await waitForDownload(destPath, tmpPath, deadline);
			if (outcome === 'complete') return destPath;
			continue; // lock vanished or went stale without a final file — take over
		}

		// We won the lock — download the model
		try {
			console.log(`[harper-fabric-embeddings] Downloading ${config.file} from Hugging Face...`);
			await lockHandle.close();

			const url = `https://huggingface.co/${config.repo}/resolve/main/${config.file}`;
			const response = await fetch(url, { redirect: 'follow' });
			if (!response.ok) {
				throw new Error(`Download failed: ${response.status} ${response.statusText} — ${url}`);
			}

			const fileStream = createWriteStream(tmpPath);
			await pipeline(response.body!, fileStream);

			// Rename to final path (atomic on same filesystem)
			await rename(tmpPath, destPath);
			console.log(`[harper-fabric-embeddings] Downloaded ${config.file} to ${destPath}`);
		} catch (err) {
			await unlink(tmpPath).catch(() => {});
			throw err;
		}

		return destPath;
	}
}

/** Remove a `.downloading` lock whose owner died (no mtime movement for `LOCK_STALE_MS`). */
function reclaimStaleLock(tmpPath: string): void {
	try {
		if (Date.now() - statSync(tmpPath).mtimeMs > LOCK_STALE_MS) {
			console.warn(`[harper-fabric-embeddings] Reclaiming stale download lock ${tmpPath}`);
			unlinkSync(tmpPath);
		}
	} catch {
		// No lock file (or it vanished between stat and unlink) — nothing to reclaim.
	}
}

/**
 * Poll until the download completes (`'complete'`), the lock disappears or goes
 * stale without a final file (`'retry'` — the caller retakes the lock), or the
 * deadline passes (throws).
 */
async function waitForDownload(destPath: string, tmpPath: string, deadline: number): Promise<'complete' | 'retry'> {
	while (true) {
		if (existsSync(destPath)) return 'complete';
		let lockAlive: boolean;
		try {
			lockAlive = Date.now() - statSync(tmpPath).mtimeMs <= LOCK_STALE_MS;
		} catch {
			// Lock gone: the winner either renamed to destPath (check again) or failed.
			return existsSync(destPath) ? 'complete' : 'retry';
		}
		if (!lockAlive) return 'retry';
		if (Date.now() > deadline) {
			throw new Error(`Timed out waiting for model download: ${destPath}`);
		}
		await sleep(500);
	}
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

	// fileURLToPath, not URL.pathname: pathname is percent-encoded (spaces → %20)
	// and carries a broken leading slash on Windows, so existsSync would miss.
	const moduleDir = path.dirname(fileURLToPath(import.meta.url));
	const searchRoots = [
		path.join(process.cwd(), 'node_modules'),
		// Sibling packages in the same node_modules as harper-fabric-embeddings
		path.resolve(moduleDir, '../..'),
		// harper-fabric-embeddings own nested node_modules (hoisted installs)
		path.resolve(moduleDir, '..', 'node_modules'),
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
 * On Linux, dlopen of the same path returns the same handle (OS deduplicates
 * by inode), so all worker threads share the same native .so in memory.
 * This is safe because llama.cpp stores all state in heap-allocated objects
 * (LlamaModel, LlamaContext) rather than C++ globals — each thread operates
 * on its own object instances through the shared native code.
 */
function loadAddon(addonPath: string): LlamaBinding {
	const mod = { exports: {} } as { exports: LlamaBinding };
	process.dlopen(mod, addonPath);
	return mod.exports;
}
