# harper-fabric-embeddings

## What This Is

Minimal llama.cpp embedding wrapper for Harper Fabric. Talks directly to the `@node-llama-cpp` native N-API addon — no build tools, no CLI, no chat wrappers, no model downloaders beyond a simple HuggingFace fetch.

~19 MB installed (native binary only) vs ~250 MB+ for `node-llama-cpp`.

Published as `harper-fabric-embeddings` on npm. Used by `harper-kb` as an optional dependency for production embedding generation on Fabric (linux-x64).

## Naming

- The product is **Harper**, not "HarperDB". Use "Harper" in prose, docs, and UI text.
- Exceptions: npm scope (`@harperfast/`), API field names. (Older packages may still use `@harperdb/`.)
- Website: **https://harper.fast/**

## Project Structure

```
harper-fabric-embeddings
├── src/
│   └── index.ts   ← Single-file TypeScript module, entire public API
├── dist/          ← Compiled output (gitignored)
├── test/
│   └── index.test.js ← Node.js built-in test runner (plain JS, imports dist)
├── setup.js       ← Postinstall: checks for platform binary, warns if missing
├── tsconfig.json
└── package.json
```

TypeScript compiled with `tsc`. Tests run against the compiled `dist/` output.

## Public API

```javascript
import { init, embed, dimensions, dispose, downloadModel } from 'harper-fabric-embeddings';

// Initialize with a model directory (finds or downloads the model)
await init({ modelsDir: '/path/to/models', modelName: 'nomic-embed-text' });

// Or initialize with an explicit model file path
await init({ modelPath: '/path/to/model.gguf' });

// Generate an embedding (L2-normalized)
const vector = await embed('Hello world');

// Get vector dimensionality
const dims = dimensions();

// Clean up native resources
await dispose();
```

### init(options)

| Option        | Type   | Default              | Description                              |
| ------------- | ------ | -------------------- | ---------------------------------------- |
| `modelPath`   | string | —                    | Absolute path to a .gguf model file      |
| `modelsDir`   | string | —                    | Directory to search/download model files |
| `modelName`   | string | `"nomic-embed-text"` | Model name from the built-in registry    |
| `contextSize` | number | `2048`               | Token context window size                |
| `batchSize`   | number | `512`                | Batch processing size                    |
| `threads`     | number | `6`                  | CPU threads for inference                |
| `gpuLayers`   | number | `0`                  | Layers to offload to GPU (0 = CPU only)  |
| `addonPath`   | string | —                    | Override path to `llama-addon.node`      |

Either `modelPath` or `modelsDir` is required.

### Model Registry

Two models are built in:

- `nomic-embed-text` — nomic-ai/nomic-embed-text-v1.5 (Q4_K_M quantization)
- `nomic-embed-text-v2-moe` — nomic-ai/nomic-embed-text-v2-moe (Q4_K_M quantization)

Models are resolved in order: HuggingFace-prefixed filename, bare filename, stem match scan, then download from HuggingFace.

## Native Binary Resolution

The module finds the `llama-addon.node` binary from installed `@node-llama-cpp` platform packages. Candidates tried in order:

1. `@node-llama-cpp/linux-x64`
2. `@node-llama-cpp/mac-arm64-metal`
3. `@node-llama-cpp/mac-x64`
4. `@node-llama-cpp/linux-arm64`

The binary lives at `<package>/bins/<folder>/llama-addon.node`.

## Dependencies

- `@node-llama-cpp/linux-x64` — optional dependency (platform-specific native binary)
- No runtime npm dependencies

The `@node-llama-cpp/*` packages provide the prebuilt `llama-addon.node` native addon. Only the platform-specific package for the target architecture is needed.

## Development

```bash
npm install
npm run build    # Compile TypeScript
```

## Testing

```bash
# Unit tests (no model file needed)
npm test

# Integration tests (requires a model file)
MODEL_PATH=/path/to/model.gguf npm test

# With a custom addon path
MODEL_PATH=/path/to/model.gguf ADDON_PATH=/path/to/llama-addon.node npm test
```

Unit tests cover error handling and binary discovery. Integration tests (skipped without `MODEL_PATH`) cover embedding generation, L2 normalization, dimensionality, and cosine similarity comparisons.

## How harper-kb Uses This

`harper-kb`'s `src/core/embeddings.ts` dynamically imports `harper-fabric-embeddings` as the preferred backend:

```javascript
const fabricModule = await import('harper-fabric-embeddings');
await fabricModule.init({ modelsDir, modelName });
const vector = await fabricModule.embed(text);
```

If this package isn't available, `harper-kb` falls back to `node-llama-cpp` (heavier, but works for local dev).

## Versioning

Sub-1.0 — API is stabilizing but may change. The package is functional and deployed.

## CI/CD

- Node.js 22+
- npm OIDC publishing with provenance
- SHA-pinned GitHub Actions
