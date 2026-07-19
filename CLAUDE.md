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
│   ├── index.ts   ← Public API: raw init/embed, handleApplication, register factory
│   ├── engine.ts  ← EmbeddingEngine class, shared addon binding registry, model download
│   └── gguf.ts    ← Minimal GGUF header reader for pooling verification (issue #12)
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

| Option        | Type   | Default              | Description                                                                                                                                                                            |
| ------------- | ------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modelPath`   | string | —                    | Absolute path to a .gguf model file                                                                                                                                                    |
| `modelsDir`   | string | —                    | Directory to search/download model files                                                                                                                                               |
| `modelName`   | string | `"nomic-embed-text"` | Model name from the built-in registry                                                                                                                                                  |
| `contextSize` | number | `2048`               | Token context window size                                                                                                                                                              |
| `batchSize`   | number | `contextSize`        | Batch size (sets both `n_batch` and `n_ubatch`). Defaults to `contextSize` so the full context window is usable; inputs longer than `batchSize` are truncated with a one-time warning. |
| `threads`     | number | `6`                  | CPU threads for inference                                                                                                                                                              |
| `gpuLayers`   | number | `0`                  | Layers to offload to GPU (0 = CPU only)                                                                                                                                                |
| `addonPath`   | string | —                    | Override path to `llama-addon.node`                                                                                                                                                    |
| `pooling`     | string | —                    | Expected pooling (`none`/`mean`/`cls`/`last`/`rank`) — verified against GGUF `<arch>.pooling_type` at init; fails loudly if absent/different. The addon cannot override model pooling. |

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

## Harper Models Backend

The `register` export is a Harper models-backend factory: Harper's
`bootstrapModels` imports this package for `backend: harper-fabric-embeddings`
entries under `models.embedding.<name>` in `harperdb-config.yaml` and invokes
`register({ logicalName, kind, config })`. It registers the engine via the
global `models.registerBackend` / `models.defineBackend` API (no Harper
imports needed), wiring it into `models.embed()`, `@embed` directives, and
model-call analytics.

Key behaviors:

- **Fast boot** — registration starts the model load/download in the
  background; the first embed call awaits it. A failed attempt retries on the
  next call. Misconfiguration throws at registration so Harper's bootstrap
  logs and skips the entry at boot.
- **Per-registration engines** — each config entry gets its own
  `EmbeddingEngine` (own model, context, serial queue). The native addon
  binding is shared across engines via a refcounted registry in `engine.ts`.
- **Prompt templates as registry data (#4)** — `inputType: 'document' | 'query'`
  resolves the model entry's `templates` (built-in registry or `templates` in
  config), interpolating `{text}` / `{task}` / `defaults.*` single-pass with
  `{{`/`}}` escapes. Invalid templates fail at registration. Omitted
  `inputType` is ALWAYS passthrough (compat contract). Template-less models
  fall back to the legacy nomic name-regex prefix. Pooling for Qwen3-class
  models is out of scope here — tracked in #5.
- **Usage reporting** — backend calls return `embeddingTokens` and
  `latencyMs`, which Harper records in `hdb_model_calls`.
- **No disposal hook yet** — Harper's registry has no backend dispose
  lifecycle; engines registered this way live until process exit.

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
