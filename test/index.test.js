/**
 * Basic tests for harper-fabric-embeddings.
 *
 * Run with a model file:
 *   MODEL_PATH=/path/to/model.gguf npm test
 *
 * Without MODEL_PATH, only unit tests (error handling, binary discovery) run.
 */

import { describe, it, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, unlinkSync, utimesSync, writeFileSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import {
	init,
	embed,
	embedBatch,
	dimensions,
	dispose,
	downloadModel,
	handleApplication,
	register,
	EmbeddingEngine,
} from '../dist/index.js';

// ─── Unit tests (no model needed) ──────────────────────────────────────────

describe('findAddonBinary', () => {
	it('finds a platform binary if installed', async () => {
		// This will throw if no binary is found — that's fine for CI
		// where @node-llama-cpp/* may not be installed
		try {
			await init({ modelPath: '/nonexistent.gguf' });
		} catch (err) {
			// Expected: either "Model file not found" (binary found, model missing)
			// or "No llama-addon.node binary found" (no platform package)
			assert.ok(
				err.message.includes('Model file not found') || err.message.includes('No llama-addon.node binary found'),
				`Unexpected error: ${err.message}`
			);
		} finally {
			await dispose();
		}
	});
});

describe('error handling', () => {
	it('throws when not initialized', async () => {
		await assert.rejects(() => embed('hello'), /Not initialized/);
	});

	it('throws for missing modelPath', async () => {
		await assert.rejects(() => init({}), /Either modelPath or modelsDir is required/);
	});

	it('throws for nonexistent model file', async () => {
		await assert.rejects(() => init({ modelPath: '/nonexistent.gguf' }), /Model file not found/);
	});

	it('embedBatch throws when not initialized', async () => {
		await assert.rejects(() => embedBatch(['hello']), /Not initialized/);
	});
});

describe('downloadModel', () => {
	it('throws for unknown model name', async () => {
		await assert.rejects(() => downloadModel('/tmp', 'nonexistent-model'), /Unknown model/);
	});

	it('returns existing path if file already downloaded', async () => {
		const dir = mkdtempSync(join(tmpdir(), 'hfe-test-'));
		try {
			// Create a fake model file matching the registry filename
			const fakeModel = join(dir, 'nomic-embed-text-v1.5.Q4_K_M.gguf');
			writeFileSync(fakeModel, 'fake');

			const result = await downloadModel(dir, 'nomic-embed-text');
			assert.equal(result, fakeModel);
		} finally {
			await rm(dir, { recursive: true });
		}
	});

	it('only one worker downloads when lock file exists', async () => {
		const dir = mkdtempSync(join(tmpdir(), 'hfe-test-'));
		try {
			// Create the .downloading lock file to simulate another worker downloading
			const lockFile = join(dir, 'nomic-embed-text-v1.5.Q4_K_M.gguf.downloading');
			writeFileSync(lockFile, '');

			// Simulate the other worker finishing by creating the final file after a delay
			const destFile = join(dir, 'nomic-embed-text-v1.5.Q4_K_M.gguf');
			setTimeout(() => writeFileSync(destFile, 'fake'), 600);

			const result = await downloadModel(dir, 'nomic-embed-text');
			assert.equal(result, destFile);
		} finally {
			await rm(dir, { recursive: true });
		}
	});
});

describe('handleApplication', () => {
	it('is async and returns a promise', () => {
		assert.equal(typeof handleApplication, 'function');
		// Verify it returns a promise (will reject since no model, but it's a promise)
		const scope = {
			directory: '/nonexistent',
			options: Object.assign({}, { on() {} }),
			on() {},
		};
		const result = handleApplication(scope);
		assert.ok(result instanceof Promise);
		// Let it reject gracefully
		result.catch(() => {});
	});

	it('reads options via getAll() when available (Harper OptionsWatcher interface)', async () => {
		// Harper's scope.options is an OptionsWatcher — config values live behind
		// .getAll() / .get([key]), not direct property access. Assert the module
		// calls getAll() to read config so it actually honors config.yaml overrides.
		let getAllCalls = 0;
		const scope = {
			directory: '/nonexistent-dir-that-wont-exist',
			options: {
				getAll() {
					getAllCalls++;
					return { modelName: 'nomic-embed-text-v2-moe', modelsDir: '/nonexistent-models-dir' };
				},
				on() {},
			},
			on() {},
		};
		await handleApplication(scope).catch(() => {
			// Expected to reject — we don't care about the error, only that getAll()
			// was consulted before the failure.
		});
		assert.ok(getAllCalls > 0, 'Expected getAll() to be called at least once when reading config');
	});
});

// ─── Harper models-backend factory (unit) ──────────────────────────────────

describe('register (Harper models backend factory)', () => {
	const originalModels = globalThis.models;

	function fakeModels() {
		const calls = { registered: [] };
		globalThis.models = {
			defineBackend(spec) {
				calls.spec = spec;
				return { __defined: spec };
			},
			registerBackend(kind, id, backend) {
				calls.registered.push({ kind, id, backend });
			},
		};
		return calls;
	}

	afterEach(() => {
		if (originalModels === undefined) delete globalThis.models;
		else globalThis.models = originalModels;
	});

	it('rejects a non-embedding kind', async () => {
		fakeModels();
		await assert.rejects(
			() => register({ logicalName: 'gen', kind: 'generative', config: { modelPath: '/nonexistent.gguf' } }),
			/embedding backend/
		);
	});

	it('rejects when the models global is unavailable', async () => {
		delete globalThis.models;
		await assert.rejects(
			() => register({ logicalName: 'default', kind: 'embedding', config: { modelPath: '/nonexistent.gguf' } }),
			/models.*not available/
		);
	});

	it('rejects at registration when modelPath/modelsDir are missing', async () => {
		fakeModels();
		await assert.rejects(
			() => register({ logicalName: 'default', kind: 'embedding', config: {} }),
			/Either modelPath or modelsDir is required/
		);
	});

	it('rejects at registration on an unknown model name (via the `model` alias)', async () => {
		fakeModels();
		await assert.rejects(
			() =>
				register({ logicalName: 'default', kind: 'embedding', config: { modelsDir: '/tmp', model: 'not-a-model' } }),
			/Unknown model: not-a-model/
		);
	});

	it('registers under the logical name without awaiting model load (fast boot)', async () => {
		const calls = fakeModels();
		await register({ logicalName: 'local', kind: 'embedding', config: { modelPath: '/nonexistent.gguf' } });
		assert.equal(calls.registered.length, 1);
		assert.equal(calls.registered[0].kind, 'embedding');
		assert.equal(calls.registered[0].id, 'local');
		assert.equal(calls.spec.name, 'fabric-embeddings:nonexistent.gguf');
		assert.equal(typeof calls.spec.embed, 'function');
	});

	it('first embed call surfaces the init failure, and later calls retry', async () => {
		const calls = fakeModels();
		await register({ logicalName: 'local', kind: 'embedding', config: { modelPath: '/nonexistent.gguf' } });
		await assert.rejects(() => calls.spec.embed('hello', {}), /Model file not found/);
		// A failed attempt resets — the next call retries (and fails the same way here).
		await assert.rejects(() => calls.spec.embed(['a', 'b'], {}), /Model file not found/);
	});

	it('supports multiple registrations with independent engines', async () => {
		const calls = fakeModels();
		await register({ logicalName: 'one', kind: 'embedding', config: { modelPath: '/one.gguf' } });
		await register({ logicalName: 'two', kind: 'embedding', config: { modelPath: '/two.gguf' } });
		assert.deepEqual(
			calls.registered.map((r) => r.id),
			['one', 'two']
		);
	});

	it('coerces string numeric config values (YAML env-var expansion)', async () => {
		const calls = fakeModels();
		await register({
			logicalName: 'coerced',
			kind: 'embedding',
			config: { modelPath: '/nonexistent.gguf', threads: '4', contextSize: '2048', batchSize: '1024', gpuLayers: '0' },
		});
		assert.equal(calls.registered.length, 1);
	});

	it('rejects a non-numeric config value at registration', async () => {
		fakeModels();
		await assert.rejects(
			() => register({ logicalName: 'bad', kind: 'embedding', config: { modelPath: '/x.gguf', threads: 'lots' } }),
			/threads must be a finite number/
		);
	});

	it('rejects a batchSize too small for BOS + token + EOS', async () => {
		fakeModels();
		await assert.rejects(
			() => register({ logicalName: 'tiny', kind: 'embedding', config: { modelPath: '/x.gguf', batchSize: 2 } }),
			/batchSize must be at least 3/
		);
	});
});

describe('EmbeddingEngine dispose', () => {
	it('embedMany after dispose fails fast without touching native resources', async () => {
		const engine = new EmbeddingEngine({ modelPath: '/nonexistent.gguf' });
		await engine.dispose();
		await assert.rejects(() => engine.embedMany(['hello']), /disposed/);
	});
});

describe('download lock recovery', () => {
	const originalFetch = globalThis.fetch;

	function fakeFetch(content) {
		globalThis.fetch = async () => new Response(content);
	}

	afterEach(() => {
		globalThis.fetch = originalFetch;
	});

	it('reclaims a stale .downloading lock left by a dead worker', async () => {
		const dir = mkdtempSync(join(tmpdir(), 'hfe-test-'));
		try {
			// A live download keeps the lock's mtime fresh; backdate it to simulate
			// a worker killed mid-download (SIGKILL/OOM leaves no catch cleanup).
			const lockFile = join(dir, 'nomic-embed-text-v1.5.Q4_K_M.gguf.downloading');
			writeFileSync(lockFile, 'partial');
			const past = new Date(Date.now() - 120_000);
			utimesSync(lockFile, past, past);

			fakeFetch('fake-model-bytes');
			const result = await downloadModel(dir, 'nomic-embed-text');
			assert.equal(result, join(dir, 'nomic-embed-text-v1.5.Q4_K_M.gguf'));
			assert.equal(readFileSync(result, 'utf8'), 'fake-model-bytes');
		} finally {
			await rm(dir, { recursive: true });
		}
	});

	it('sends an HF token as a bearer when the env var is set, none otherwise', async () => {
		const savedToken = process.env.HF_TOKEN;
		const savedHub = process.env.HUGGING_FACE_HUB_TOKEN;
		const seen = [];
		globalThis.fetch = async (url, init) => {
			seen.push(init?.headers?.authorization);
			return new Response('fake-model-bytes');
		};
		const dir1 = mkdtempSync(join(tmpdir(), 'hfe-test-'));
		const dir2 = mkdtempSync(join(tmpdir(), 'hfe-test-'));
		try {
			// Clear the fallback var too, or a developer machine that exports it
			// makes the no-token half of this test fail spuriously.
			delete process.env.HUGGING_FACE_HUB_TOKEN;
			process.env.HF_TOKEN = 'hf_unit_test_token';
			await downloadModel(dir1, 'nomic-embed-text');
			delete process.env.HF_TOKEN;
			await downloadModel(dir2, 'nomic-embed-text');
			assert.deepEqual(seen, ['Bearer hf_unit_test_token', undefined]);
		} finally {
			if (savedToken === undefined) delete process.env.HF_TOKEN;
			else process.env.HF_TOKEN = savedToken;
			if (savedHub !== undefined) process.env.HUGGING_FACE_HUB_TOKEN = savedHub;
			await rm(dir1, { recursive: true });
			await rm(dir2, { recursive: true });
		}
	});

	it('a 403 without a token hints at HF_TOKEN', async () => {
		const savedToken = process.env.HF_TOKEN;
		const savedHub = process.env.HUGGING_FACE_HUB_TOKEN;
		globalThis.fetch = async () => new Response('denied', { status: 403, statusText: 'Forbidden' });
		const dir = mkdtempSync(join(tmpdir(), 'hfe-test-'));
		try {
			delete process.env.HF_TOKEN;
			delete process.env.HUGGING_FACE_HUB_TOKEN;
			await assert.rejects(() => downloadModel(dir, 'nomic-embed-text'), /403.*set HF_TOKEN/s);
		} finally {
			if (savedToken !== undefined) process.env.HF_TOKEN = savedToken;
			if (savedHub !== undefined) process.env.HUGGING_FACE_HUB_TOKEN = savedHub;
			await rm(dir, { recursive: true });
		}
	});

	it('a waiting worker takes over when the downloader fails without producing a file', async () => {
		const dir = mkdtempSync(join(tmpdir(), 'hfe-test-'));
		try {
			const lockFile = join(dir, 'nomic-embed-text-v1.5.Q4_K_M.gguf.downloading');
			writeFileSync(lockFile, '');
			// Simulate the winning worker dying: lock vanishes, no final file appears
			setTimeout(() => unlinkSync(lockFile), 300);

			fakeFetch('recovered-model-bytes');
			const result = await downloadModel(dir, 'nomic-embed-text');
			assert.equal(readFileSync(result, 'utf8'), 'recovered-model-bytes');
		} finally {
			await rm(dir, { recursive: true });
		}
	});
});

// ─── Integration tests (need MODEL_PATH env var) ───────────────────────────

const MODEL_PATH = process.env.MODEL_PATH;
const ADDON_PATH = process.env.ADDON_PATH || undefined;

describe('embedding generation', { skip: !MODEL_PATH }, () => {
	before(async () => {
		await init({ modelPath: MODEL_PATH, addonPath: ADDON_PATH, threads: 4 });
	});

	after(async () => {
		await dispose();
	});

	it('returns the embedding dimensionality', () => {
		const dims = dimensions();
		assert.ok(dims > 0, `Expected positive dimensions, got ${dims}`);
	});

	it('generates an embedding vector', async () => {
		const vec = await embed('Hello world');
		assert.ok(Array.isArray(vec), 'Expected array');
		assert.ok(vec.length > 0, 'Expected non-empty vector');

		// Should be L2-normalized (magnitude ≈ 1.0)
		const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
		assert.ok(Math.abs(mag - 1.0) < 0.01, `Expected unit vector, magnitude = ${mag}`);
	});

	it('returns empty array for empty input', async () => {
		const vec = await embed('');
		assert.deepEqual(vec, []);
	});

	it('produces different vectors for different inputs', async () => {
		const v1 = await embed('Cats are great pets');
		const v2 = await embed('Quantum mechanics is complex');

		// Cosine similarity should be < 1.0 for unrelated texts
		let dot = 0;
		for (let i = 0; i < v1.length; i++) dot += v1[i] * v2[i];
		assert.ok(dot < 0.95, `Expected dissimilar vectors, cosine = ${dot}`);
	});

	it('produces similar vectors for similar inputs', async () => {
		const v1 = await embed('The cat sat on the mat');
		const v2 = await embed('A cat was sitting on a mat');

		let dot = 0;
		for (let i = 0; i < v1.length; i++) dot += v1[i] * v2[i];
		assert.ok(dot > 0.7, `Expected similar vectors, cosine = ${dot}`);
	});

	it('handles concurrent embed() calls without crashing', async () => {
		const texts = Array.from({ length: 15 }, (_, i) => `Concurrent test message ${i}`);
		const results = await Promise.all(texts.map((t) => embed(t)));

		assert.equal(results.length, 15);
		for (const vec of results) {
			assert.ok(vec.length > 0, 'Expected non-empty vector');
			const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
			assert.ok(Math.abs(mag - 1.0) < 0.01, `Expected unit vector, magnitude = ${mag}`);
		}
	});

	it('embedBatch returns vectors for all inputs', async () => {
		const texts = ['First text', 'Second text', 'Third text'];
		const results = await embedBatch(texts);

		assert.equal(results.length, 3);
		for (const vec of results) {
			assert.ok(vec.length > 0, 'Expected non-empty vector');
		}
	});

	it('embedBatch returns empty array for empty input', async () => {
		const results = await embedBatch([]);
		assert.deepEqual(results, []);
	});

	it('embeds inputs longer than the default 512 n_ubatch without crashing', async () => {
		// Natural English at ~0.3 tokens/char → ~3KB tokenizes well past the old
		// n_ubatch default of 512. Pre-fix this triggered GGML_ASSERT and killed
		// the host process; post-fix it embeds normally (batchSize defaults to
		// contextSize = 2048).
		const longText = 'The quick brown fox jumps over the lazy dog. '.repeat(70);
		const vec = await embed(longText);
		assert.ok(vec.length > 0, 'Expected non-empty vector for long input');
		const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
		assert.ok(Math.abs(mag - 1.0) < 0.01, `Expected unit vector, magnitude = ${mag}`);
	});
});

// Separate describe with its own init/dispose so we can exercise the
// explicit-small-batchSize truncation path without tearing down the shared
// context used by the tests above.
describe('embedding truncation with small batchSize', { skip: !MODEL_PATH }, () => {
	before(async () => {
		// First dispose any init from the prior describe block's before() —
		// node:test runs describes sequentially so the prior `after` will have
		// already fired, but we guard anyway.
		await dispose().catch(() => {});
		await init({ modelPath: MODEL_PATH, addonPath: ADDON_PATH, threads: 4, batchSize: 64, contextSize: 2048 });
	});

	after(async () => {
		await dispose();
	});

	it('truncates oversized input instead of aborting the process', async () => {
		// 70 repetitions ≈ 600+ tokens, well over batchSize=64. Pre-fix: crash.
		// Post-fix: truncate to 62 body tokens + BOS/EOS.
		const longText = 'The quick brown fox jumps over the lazy dog. '.repeat(70);
		const vec = await embed(longText);
		assert.ok(vec.length > 0, 'Expected non-empty vector after truncation');
	});
});

// The prefix assertions only hold for nomic models (others get no prefix, so
// document and query vectors are identical). CI's integration model is nomic.
const NOMIC_MODEL = MODEL_PATH ? /nomic-embed-text/i.test(basename(MODEL_PATH)) : false;

describe('register backend embed (integration)', { skip: !MODEL_PATH }, () => {
	const originalModels = globalThis.models;
	let spec;
	let engine;

	before(async () => {
		globalThis.models = {
			defineBackend(s) {
				spec = s;
				return s;
			},
			registerBackend() {},
		};
		engine = await register({
			logicalName: 'default',
			kind: 'embedding',
			config: { modelPath: MODEL_PATH, addonPath: ADDON_PATH, threads: 4 },
		});
	});

	after(async () => {
		if (originalModels === undefined) delete globalThis.models;
		else globalThis.models = originalModels;
		if (engine) await engine.dispose();
	});

	it('embeds through the models-backend contract', async () => {
		const result = await spec.embed('Hello world', { inputType: 'document' });
		assert.equal(result.status, 'completed');
		assert.equal(result.output.length, 1);
		assert.ok(result.output[0] instanceof Float32Array, 'Expected Float32Array output');
		assert.ok(result.output[0].length > 0, 'Expected non-empty vector');
		const mag = Math.sqrt(result.output[0].reduce((s, v) => s + v * v, 0));
		assert.ok(Math.abs(mag - 1.0) < 0.01, `Expected unit vector, magnitude = ${mag}`);
		assert.ok(result.usage.embeddingTokens > 0, 'Expected a token count');
		assert.ok(result.usage.latencyMs >= 0, 'Expected a latency measurement');
	});

	it('accepts string[] input and returns one vector per input', async () => {
		const result = await spec.embed(['first text', 'second text'], {});
		assert.equal(result.output.length, 2);
	});

	it('applies nomic task prefixes: document and query vectors differ', { skip: !NOMIC_MODEL }, async () => {
		const [doc] = (await spec.embed('sailing ships across the ocean', { inputType: 'document' })).output;
		const [query] = (await spec.embed('sailing ships across the ocean', { inputType: 'query' })).output;
		let dot = 0;
		for (let i = 0; i < doc.length; i++) dot += doc[i] * query[i];
		assert.ok(dot < 0.999, `Expected the task prefix to shift the vector, cosine = ${dot}`);
	});

	it('rejects a pre-aborted signal without decoding', async () => {
		const controller = new AbortController();
		controller.abort();
		await assert.rejects(() => spec.embed('never embedded', { signal: controller.signal }));
	});
});

describe('engine dispose during in-flight embed (integration)', { skip: !MODEL_PATH }, () => {
	it('dispose() drains the queue instead of freeing the native context under a decode', async () => {
		const engine = new EmbeddingEngine({ modelPath: MODEL_PATH, addonPath: ADDON_PATH, threads: 4 });
		// Warm up so the next embed goes straight to a native decode
		await engine.embedMany(['warmup']);
		const inflight = engine.embedMany(['some text to embed', 'and some more text']);
		const disposal = engine.dispose();
		const result = await inflight;
		assert.equal(result.vectors.length, 2);
		await disposal;
		await assert.rejects(() => engine.embedMany(['after dispose']), /disposed/);
	});
});
