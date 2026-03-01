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
import { init, embed, dimensions, dispose } from '../dist/index.js';

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
});
