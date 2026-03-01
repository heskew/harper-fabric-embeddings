/**
 * Postinstall check: verify a compatible platform binary is available.
 * Does NOT download anything — just warns if missing.
 */

import { existsSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);

const PLATFORM_PACKAGES = {
	'linux-x64': '@node-llama-cpp/linux-x64',
	'linux-arm64': '@node-llama-cpp/linux-arm64',
	'darwin-arm64': '@node-llama-cpp/mac-arm64-metal',
	'darwin-x64': '@node-llama-cpp/mac-x64',
};

const key = `${process.platform}-${process.arch}`;
const pkg = PLATFORM_PACKAGES[key];

if (!pkg) {
	console.warn(
		`[harper-fabric-embeddings] No prebuilt binary available for ${key}. ` +
			`Embedding generation will not work on this platform.`
	);
	process.exit(0);
}

try {
	const pkgMain = require.resolve(pkg);
	const binsDir = path.join(path.dirname(pkgMain), '..', 'bins');

	let found = false;
	if (existsSync(binsDir)) {
		for (const entry of readdirSync(binsDir)) {
			if (existsSync(path.join(binsDir, entry, 'llama-addon.node'))) {
				found = true;
				break;
			}
		}
	}

	if (found) {
		console.log(`[harper-fabric-embeddings] Native binary found (${key}).`);
	} else {
		console.warn(
			`[harper-fabric-embeddings] Package ${pkg} installed but binary not found. ` +
				`Embedding generation may not work.`
		);
	}
} catch {
	console.warn(
		`[harper-fabric-embeddings] Platform package ${pkg} not installed. ` +
			`Install it to enable embedding generation: npm install ${pkg}`
	);
}
