# harper-fabric-embeddings

Minimal llama.cpp embedding wrapper for Harper Fabric. Talks directly to the `@node-llama-cpp` native N-API addon — no build tools, no CLI, no chat wrappers, no model downloaders beyond a simple HuggingFace fetch.

~19 MB installed (native binary only) vs ~250 MB+ for `node-llama-cpp`.

## Installation

```sh
npm install harper-fabric-embeddings
```

The package uses `@node-llama-cpp` platform-specific binaries. The `linux-x64` binary is included as an optional dependency. For other platforms, install the appropriate package:

```sh
npm install @node-llama-cpp/mac-arm64-metal  # macOS Apple Silicon
npm install @node-llama-cpp/mac-x64          # macOS Intel
npm install @node-llama-cpp/linux-arm64      # Linux ARM64
```

## Use as a Harper models backend

Harper's models bootstrap can load this package directly as an embedding
backend. Install it in the Harper instance root and name it in
`harperdb-config.yaml`:

```yaml
models:
  embedding:
    default:
      backend: harper-fabric-embeddings
      modelName: nomic-embed-text
      modelsDir: ./models
```

Everything that consumes Harper's models API then routes through the local
GGUF engine: `models.embed()`, `@embed` table directives, and model-call
analytics (`hdb_model_calls` gets `embeddingTokens` + `latencyMs` per call).

- `modelsDir` or `modelPath` is required. Relative paths resolve against
  Harper's working directory.
- `model` is accepted as an alias for `modelName` (the field Harper's
  built-in backends use). `contextSize`, `batchSize`, `threads`, `gpuLayers`,
  and `addonPath` pass through as with `init()`.
- Boot is not blocked on the model: registration kicks off the load/download
  in the background and the first embed call awaits it. Misconfiguration
  (wrong kind, missing model source, unknown model name) fails at boot, where
  Harper logs and skips the entry.
- `inputType` is honored: each model's **prompt templates** (see below) shape
  document vs query encodings — nomic models get their `search_document: ` /
  `search_query: ` task prefixes. Harper's `@embed` directive passes
  `inputType: 'document'`; when `inputType` is omitted (the default through
  `models.embed()`), **no template is applied, ever** — input handling
  identical to the raw API and to 0.2.x, so pre-existing vectors stay
  comparable. Corpora embedded template-less need a one-time re-embed to
  benefit from templated queries.
- Vector dimensionality: Harper's `models` facade has no model-metadata
  accessor yet, so read it from the first embed result
  (`(await models.embed('x'))[0].length` — 768 for both built-in nomic
  models), or use `dimensions()` on the raw API.
- Multiple entries work — each gets its own engine (own model + context),
  sharing one native addon binding:

```yaml
models:
  embedding:
    default:
      backend: harper-fabric-embeddings
      modelName: nomic-embed-text
      modelsDir: ./models
      fallback: [remote]
    moe:
      backend: harper-fabric-embeddings
      modelName: nomic-embed-text-v2-moe
      modelsDir: ./models
    remote:
      backend: openai
      model: text-embedding-3-small
      apiKey: ${OPENAI_API_KEY}
```

## Usage

```typescript
import { init, embed, dimensions, dispose } from 'harper-fabric-embeddings';

// Initialize with a models directory (finds or downloads the model)
await init({ modelsDir: '/path/to/models' });

// Generate an embedding (L2-normalized)
const vector = await embed('Hello world');

// Get vector dimensionality
const dims = dimensions();

// Clean up native resources
await dispose();
```

## API

### `init(options)`

Initialize the embedding engine. Call once before using `embed()`.

| Option        | Type   | Default              | Description                                                             |
| ------------- | ------ | -------------------- | ----------------------------------------------------------------------- |
| `modelPath`   | string | —                    | Absolute path to a .gguf model file                                     |
| `modelsDir`   | string | —                    | Directory to search/download model files                                |
| `modelName`   | string | `"nomic-embed-text"` | Model name from the built-in registry                                   |
| `contextSize` | number | `2048`               | Token context window size                                               |
| `batchSize`   | number | `contextSize`        | Batch size (`n_batch`/`n_ubatch`); longer inputs are truncated          |
| `threads`     | number | `6`                  | CPU threads for inference                                               |
| `gpuLayers`   | number | `0`                  | Layers to offload to GPU (0 = CPU only)                                 |
| `addonPath`   | string | —                    | Override path to `llama-addon.node`                                     |
| `templates`   | object | registry entry       | Per-inputType prompt templates (`document`/`query`/`defaults`)          |
| `pooling`     | string | —                    | Expected pooling declared by the model file — verified at init, not set |

`pooling` (`'none' | 'mean' | 'cls' | 'last' | 'rank'`) is verification, not
override: the native addon exposes no pooling option, so llama.cpp always uses
the model's own `<arch>.pooling_type` metadata. Declaring the expectation makes
init fail loudly when a GGUF omits or contradicts it — instead of a
metadata-less conversion silently mean-pooling a last-token model.

Either `modelPath` or `modelsDir` is required.

### `embed(text)`

Generate an L2-normalized embedding vector for the given text. Returns `number[]`.

### `dimensions()`

Returns the embedding vector dimensionality.

### `dispose()`

Clean up native resources (model, context, binding).

### `downloadModel(dir, modelName?)`

Download a model from HuggingFace. Called automatically by `init()` when using `modelsDir` and no local model is found.

## Prompt templates

Embedding models disagree about how document and query inputs should be
affixed — nomic wants fixed prefixes, instruct-style embedders (Qwen3-Embedding
and friends) want a free-text task instruction on the query side only. That
convention is **data on the model entry**, not code:

```yaml
models:
  embedding:
    default:
      backend: harper-fabric-embeddings
      modelPath: ./models/my-model.Q8_0.gguf
      templates:
        document: '{text}'
        query: "Instruct: {task}\nQuery: {text}"
        defaults:
          task: 'Given a search query, retrieve relevant passages that answer the query'
```

- `{text}` is the input text. `{task}` comes from the embed call's `task`
  option (`models.embed(q, { inputType: 'query', task: '...' })`), falling
  back to `defaults.task`. Any other placeholder must be covered by
  `defaults`. Escape literal braces as `{{` / `}}`.
- Interpolation is single-pass; invalid templates (unknown placeholders,
  unescaped braces) fail at registration — Harper logs and skips the entry at
  boot rather than surfacing at first embed.
- Omitted `inputType` is always passthrough, templates or not. A missing side
  (e.g. only `query` declared) falls back to the legacy nomic name-prefix
  heuristic, then passthrough.
- The built-in nomic entries declare their prefixes as templates; explicit
  `templates` in config override an entry's own.
- Note for typed Harper consumers: `task` passes through `models.embed()` at
  runtime, but isn't on core's `EmbedOpts` type yet — cast until core widens it.

## Models

Two models are built in:

| Name                      | Source                           | Quantization |
| ------------------------- | -------------------------------- | ------------ |
| `nomic-embed-text`        | nomic-ai/nomic-embed-text-v1.5   | Q4_K_M       |
| `nomic-embed-text-v2-moe` | nomic-ai/nomic-embed-text-v2-moe | Q4_K_M       |

Models are resolved in order: HuggingFace-prefixed filename, bare filename, stem match scan, then download from HuggingFace.

HuggingFace may reject anonymous large-file downloads (HTTP 403). Set
`HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) to a free account token and the
download sends it as a bearer; alternatively pre-seed `modelsDir` with the
model file and no download happens at all.

## Testing

```sh
# Unit tests (no model file needed)
npm test

# Integration tests (requires a model file)
MODEL_PATH=/path/to/model.gguf npm test
```

## Requirements

- Node.js 22+
- A `@node-llama-cpp` platform package for your architecture

## License

MIT
