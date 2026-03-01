#!/usr/bin/env node

/**
 * Integration test runner.
 *
 * Downloads the platform-specific native addon (if needed) and a test model,
 * then runs the full test suite with MODEL_PATH set.
 *
 * Usage:
 *   node scripts/test-integration.js
 *   npm run test:integration
 */

import { existsSync, readdirSync } from 'node:fs';
import { execSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const PLATFORM_ADDONS = {
	'darwin-arm64': '@node-llama-cpp/mac-arm64-metal',
	'darwin-x64': '@node-llama-cpp/mac-x64',
	'linux-x64': '@node-llama-cpp/linux-x64',
	'linux-arm64': '@node-llama-cpp/linux-arm64',
};

const key = `${process.platform}-${process.arch}`;
const addonPkg = PLATFORM_ADDONS[key];

if (!addonPkg) {
	console.error(`No prebuilt addon available for ${key}`);
	process.exit(1);
}

// ── 1. Ensure platform addon is installed ───────────────────────────────────

const pkgDir = path.join(root, 'node_modules', addonPkg);

if (!existsSync(pkgDir)) {
	console.log(`Installing ${addonPkg}...`);
	execSync(`npm install --no-save ${addonPkg}`, { cwd: root, stdio: 'inherit' });
}

// ── 2. Find the addon binary ────────────────────────────────────────────────

const binsDir = path.join(pkgDir, 'bins');
let addonPath;

if (existsSync(binsDir)) {
	for (const entry of readdirSync(binsDir)) {
		const candidate = path.join(binsDir, entry, 'llama-addon.node');
		if (existsSync(candidate)) {
			addonPath = candidate;
			break;
		}
	}
}

if (!addonPath) {
	console.error(`Could not find llama-addon.node in ${addonPkg}`);
	process.exit(1);
}

// ── 3. Ensure test model is downloaded ──────────────────────────────────────

const modelsDir = path.join(root, '.models');
const modelFile = 'nomic-embed-text-v1.5.Q4_K_M.gguf';
const modelPath = path.join(modelsDir, modelFile);

if (!existsSync(modelPath)) {
	console.log(`Downloading test model to ${modelsDir}...`);
	const { downloadModel } = await import('../dist/index.js');
	await downloadModel(modelsDir, 'nomic-embed-text');
}

// ── 4. Run tests ────────────────────────────────────────────────────────────

console.log(`\nRunning integration tests...`);
console.log(`  MODEL_PATH=${modelPath}`);
console.log(`  ADDON_PATH=${addonPath}\n`);

execSync('node --test test/index.test.js', {
	cwd: root,
	stdio: 'inherit',
	env: {
		...process.env,
		MODEL_PATH: modelPath,
		ADDON_PATH: addonPath,
	},
});
