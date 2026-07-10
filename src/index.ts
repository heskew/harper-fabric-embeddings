/**
 * harper-fabric-embeddings
 *
 * Minimal llama.cpp embedding wrapper for Harper Fabric. Talks directly to
 * the `@node-llama-cpp` native N-API addon — no build tools, no CLI, no chat
 * wrappers, no model downloaders beyond a simple HuggingFace fetch.
 *
 * ~19 MB installed (native binary only) vs ~250 MB+ for node-llama-cpp.
 *
 * Three ways in:
 * - Raw API — `init` / `embed` / `embedBatch` / `dimensions` / `dispose`
 *   against a module-level default engine (what harper-kb consumes).
 * - Harper sub-component — `handleApplication(scope)` when loaded via
 *   `package:` in a parent component's config.yaml.
 * - Harper models backend — the `register` factory, invoked by Harper's
 *   models bootstrap for `backend: harper-fabric-embeddings` config entries.
 *   Wires the engine into `models.embed()`, `@embed` table directives, and
 *   model-call analytics.
 */

import path from 'node:path';
import { EmbeddingEngine, type EngineOptions } from './engine.js';

export { EmbeddingEngine, downloadModel } from './engine.js';
export type { EngineOptions, EmbedManyOptions, EmbedManyResult } from './engine.js';

/** Options for `init()`. Same shape as `EngineOptions`. */
export type InitOptions = EngineOptions;

// ─── Module-level default engine (raw API) ──────────────────────────────────

let defaultEngine: EmbeddingEngine | null = null;
let defaultInit: Promise<void> | null = null;

/**
 * Initialize the default embedding engine. Call once before using `embed()`.
 *
 * Provide either `modelPath` (absolute path to a .gguf file) or
 * `modelsDir` (directory to search/download into) + optional `modelName`.
 *
 * Safe to call concurrently — concurrent callers share the same initialization.
 * A failed attempt resets so the next call retries.
 */
export async function init(options: InitOptions): Promise<void> {
	if (defaultInit) return defaultInit;
	const attempt = (async () => {
		const engine = new EmbeddingEngine(options);
		await engine.ensureReady();
		defaultEngine = engine;
	})();
	defaultInit = attempt;
	try {
		await attempt;
	} catch (err) {
		if (defaultInit === attempt) defaultInit = null;
		throw err;
	}
}

function requireDefaultEngine(): EmbeddingEngine {
	if (!defaultEngine) throw new Error('Not initialized. Call init() first.');
	return defaultEngine;
}

/**
 * Generate an L2-normalized embedding vector for the given text.
 *
 * Calls are serialized internally — concurrent callers wait in queue
 * rather than hitting the llama.cpp context simultaneously.
 */
export async function embed(text: string): Promise<number[]> {
	const { vectors } = await requireDefaultEngine().embedMany([text]);
	const vector = vectors[0];
	return vector ? Array.from(vector) : [];
}

/**
 * Generate embedding vectors for multiple texts.
 *
 * More efficient than calling embed() in a loop — texts are processed
 * sequentially through the native context without queue overhead per item.
 */
export async function embedBatch(texts: string[]): Promise<number[][]> {
	const { vectors } = await requireDefaultEngine().embedMany(texts);
	return vectors.map((vector) => Array.from(vector));
}

/**
 * Get the embedding vector dimensionality.
 */
export function dimensions(): number {
	if (!defaultEngine) throw new Error('Not initialized. Call init() first.');
	return defaultEngine.dimensions();
}

/**
 * Clean up native resources for the default engine.
 */
export async function dispose(): Promise<void> {
	defaultInit = null;
	const engine = defaultEngine;
	defaultEngine = null;
	if (engine) await engine.dispose();
}

// ─── Harper sub-component plugin entry point ────────────────────────────────

/**
 * Harper plugin hook — called on each worker thread when loaded as a
 * sub-component via `package:` in the parent's config.yaml.
 *
 * Reads config from scope.options, initializes the GGUF engine, and
 * handles close/change events for cleanup and hot-reload.
 *
 * Uses the module-level default engine: two sub-components loading this
 * package in the same worker share one engine (the second's config is
 * ignored, and either's close tears it down for both). When you need
 * per-entry engines, use the models-backend `register` path instead.
 *
 * Config options (in parent config.yaml):
 *   modelName   — model from the built-in registry (default: nomic-embed-text)
 *   modelsDir   — override models directory (default: <plugin dir>/models)
 *   contextSize — token context window size
 *   batchSize   — batch processing size
 *   threads     — CPU threads for inference
 *   gpuLayers   — layers to offload to GPU (0 = CPU only)
 *   addonPath   — override path to llama-addon.node
 */
export async function handleApplication(scope: {
	directory: string;
	options: {
		getAll?: () => Record<string, unknown>;
		on(event: 'change', fn: () => void): void;
	} & Record<string, unknown>;
	on(event: 'close', fn: () => void): void;
}): Promise<void> {
	function resolveConfig(): InitOptions {
		// Harper's `scope.options` is an OptionsWatcher (EventEmitter); config values
		// are NOT exposed as direct properties — they live behind `.getAll()` /
		// `.get([key])`. Without this, every config.yaml override (modelName,
		// batchSize, contextSize, threads, gpuLayers, addonPath) silently falls
		// through to defaults. Fall back to direct property access if a caller
		// passes a plain-object scope (tests, non-Harper hosts).
		const opts =
			typeof scope.options.getAll === 'function' ? scope.options.getAll() : (scope.options as Record<string, unknown>);
		return {
			modelsDir: (opts.modelsDir as string) || path.join(scope.directory, 'models'),
			modelName: (opts.modelName as string) || 'nomic-embed-text',
			contextSize: toFiniteNumber(opts.contextSize, 'contextSize'),
			batchSize: toFiniteNumber(opts.batchSize, 'batchSize'),
			threads: toFiniteNumber(opts.threads, 'threads'),
			gpuLayers: toFiniteNumber(opts.gpuLayers, 'gpuLayers'),
			addonPath: opts.addonPath as string | undefined,
		};
	}

	// Await init so the model is ready before Harper routes requests to this worker
	await init(resolveConfig());

	scope.on('close', () => {
		dispose().catch((err) => {
			console.error('[harper-fabric-embeddings] Error during dispose:', (err as Error).message);
		});
	});

	scope.options.on('change', () => {
		dispose()
			.then(() => init(resolveConfig()))
			.catch((err) => {
				console.error(
					'[harper-fabric-embeddings] Failed to re-initialize after config change:',
					(err as Error).message
				);
			});
	});
}

// ─── Harper models-backend factory ──────────────────────────────────────────

/** Registration args Harper's models bootstrap passes to a backend module factory. */
export interface RegisterArgs {
	/** Logical name of the config entry (`models.embedding.<name>`); callers select it via `opts.model`. */
	logicalName: string;
	/** `'embedding'` is the only kind this package supports. */
	kind: string;
	/** The env-expanded config entry from `harperdb-config.yaml`. */
	config: Record<string, unknown>;
}

/** Per-call options Harper's models facade hands the backend (the subset we use). */
interface BackendEmbedOpts {
	inputType?: 'document' | 'query';
	signal?: AbortSignal;
}

interface EmbedUsage {
	embeddingTokens: number;
	latencyMs: number;
}

type EmbedCallResult = { status: 'completed'; output: Float32Array[]; usage: EmbedUsage };

/** The slice of Harper's global `models` API this factory needs. */
interface ModelsApi {
	registerBackend(kind: 'embedding' | 'generative', id: string, backend: unknown): void;
	defineBackend(spec: {
		name: string;
		embed: (input: string | string[], opts: BackendEmbedOpts) => Promise<EmbedCallResult>;
	}): unknown;
}

/**
 * Harper models-backend factory. Harper's `bootstrapModels` imports this
 * package when a `models:` config entry names it and invokes this export
 * (a named `register` export is probed before the default export):
 *
 *   models:
 *     embedding:
 *       default:
 *         backend: harper-fabric-embeddings
 *         modelName: nomic-embed-text
 *         modelsDir: ./models
 *
 * `modelsDir` (or `modelPath`) is required — there is no instance-root default
 * on this path. `model` is accepted as an alias for `modelName`, matching the
 * field Harper's built-in backends use.
 *
 * Registration is fast-boot: model resolution / download / load kicks off in
 * the background here and the FIRST embed call awaits it (a failed attempt
 * retries on the next call). Misconfiguration — wrong kind, missing model
 * source, unknown model name — throws right here so Harper's bootstrap logs
 * and skips the entry at boot instead of surfacing at first use.
 *
 * Returns the engine so tests and advanced callers can dispose it; Harper
 * ignores the return value.
 */
export async function register({ logicalName, kind, config }: RegisterArgs): Promise<EmbeddingEngine> {
	if (kind !== 'embedding') {
		throw new Error(`harper-fabric-embeddings is an embedding backend; cannot register models.${kind}.${logicalName}`);
	}
	const models = (globalThis as { models?: ModelsApi }).models;
	if (typeof models?.registerBackend !== 'function' || typeof models?.defineBackend !== 'function') {
		throw new Error(
			'global `models` API not available — harper-fabric-embeddings requires a Harper version with model-backend support'
		);
	}

	const engine = new EmbeddingEngine(engineOptionsFromConfig(config));

	// Fast boot: start the model load/download now, but don't block Harper boot
	// on it — the first embed call awaits readiness instead.
	engine.ensureReady().catch((err: Error) => {
		console.error(
			`[harper-fabric-embeddings] models.embedding.${logicalName}: model init failed (will retry on first embed): ${err.message}`
		);
	});

	models.registerBackend(
		'embedding',
		logicalName,
		models.defineBackend({
			name: `fabric-embeddings:${engine.modelIdentity}`,
			embed: async (input, opts) => {
				const texts = Array.isArray(input) ? input : [input];
				const started = performance.now();
				const { vectors, tokens } = await engine.embedMany(texts, {
					inputType: opts?.inputType,
					signal: opts?.signal,
				});
				return {
					status: 'completed',
					output: vectors,
					usage: { embeddingTokens: tokens, latencyMs: Math.round(performance.now() - started) },
				};
			},
		})
	);

	return engine;
}

export default register;

function engineOptionsFromConfig(config: Record<string, unknown>): EngineOptions {
	const c = config ?? {};
	return {
		modelPath: c.modelPath as string | undefined,
		modelsDir: c.modelsDir as string | undefined,
		// `model` is the conventional per-entry field on Harper's built-in
		// backends; `modelName` is this package's native option. Either works.
		modelName: (c.modelName as string | undefined) ?? (c.model as string | undefined),
		contextSize: toFiniteNumber(c.contextSize, 'contextSize'),
		batchSize: toFiniteNumber(c.batchSize, 'batchSize'),
		threads: toFiniteNumber(c.threads, 'threads'),
		gpuLayers: toFiniteNumber(c.gpuLayers, 'gpuLayers'),
		addonPath: c.addonPath as string | undefined,
	};
}

/**
 * Coerce a numeric config value. YAML env-var expansion (`threads: ${THREADS}`)
 * and quoted values (`"2048"`) deliver strings; these fields feed native
 * constructors and the truncation math, so a non-finite value must fail at
 * registration, not inside the addon.
 */
function toFiniteNumber(value: unknown, field: string): number | undefined {
	if (value === undefined || value === null || value === '') return undefined;
	const n = Number(value);
	if (!Number.isFinite(n)) {
		throw new Error(`${field} must be a finite number, got '${String(value)}'`);
	}
	return n;
}
