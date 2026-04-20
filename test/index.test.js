/**
 * Basic tests for harper-fabric-embeddings.
 *
 * Run with a model file:
 *   MODEL_PATH=/path/to/model.gguf npm test
 *
 * Without MODEL_PATH, only unit tests (error handling, binary discovery) run.
 */

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { init, embed, embedBatch, dimensions, dispose, downloadModel, handleApplication } from '../dist/index.js';

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
