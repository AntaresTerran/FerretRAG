# FerretRAG Privacy Checklist

FerretRAG release candidates must preserve the local-only promise.

- No cloud API calls in indexing, retrieval, or chat.
- No telemetry, analytics, or background network calls.
- No model files committed to git.
- Runtime data stays under ignored local paths such as `data/`.
- Config files must not contain secrets.
- Source snippets shown in the UI come only from indexed local files.
- GitHub Actions must not upload user data, local indexes, or GGUF files.
- Release docs must clearly state that users provide or download models locally.
