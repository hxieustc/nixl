# Task 1 Report: Add `nixlRespKvEngine` base

**Status:** DONE_WITH_CONCERNS  
**Working tree:** `redis5/nixl`  
**Commit:** none (author commits; git add/commit skipped per global constraints)

## Summary

Added the thin RESP-family KV base under `src/plugins/respkv/` only. Redis was not moved. `src/plugins/meson.build` still points at the old `redis/` plugin. The base target does not link hiredis or GLIDE.

## Files created

| File | Role |
|---|---|
| `src/plugins/respkv/respkv_engine.h` | `nixlRespKvEngine : public nixlBackendEngine` with verbatim brief API |
| `src/plugins/respkv/respkv_engine.cpp` | `loadLocalMD` + `validatePrepCommon` |
| `src/plugins/respkv/meson.build` | `respkv_engine` static lib + `respkv_engine_dep`; redis/valkey subdirs commented |

## Files intentionally not changed

- `src/plugins/meson.build` — still `subdir('redis')`; no `subdir('respkv')`
- `src/plugins/redis/**` — left in place for Task 2

## Implementation notes

### Header (`respkv_engine.h`)

Copied from the task brief (capabilities, no-ops, `getSupportedMems()` `{OBJ_SEG, DRAM_SEG}`, `loadLocalMD` declaration, protected `validatePrepCommon`). Class remains abstract: `registerMem` / `deregisterMem` / `prepXfer` / `postXfer` / `checkXfer` / `releaseReqH` stay pure virtual on `nixlBackendEngine`.

### `loadLocalMD`

Mirrors current REDIS identity mapping:

- `input == nullptr` → `output = nullptr`, `NIXL_ERR_INVALID_PARAM`
- else `output = input`, `NIXL_SUCCESS`

### `validatePrepCommon`

Mirrors `nixlRedisKVEngine`’s `isValidPrepXferParams` in `src/plugins/redis/redis_backend.cpp` **except** the remote DRAM/OBJ type check (subclasses add that in later tasks).

Checks and order (same as Redis, minus remote type):

1. `operation` must be `NIXL_READ` or `NIXL_WRITE` (`NIXL_ERROR` + false)
2. `local.descCount() > 0` (`NIXL_ERROR` + false)
3. `local.descCount() == remote.descCount()` (`NIXL_ERROR` + false)
4. if `remote_agent != localAgent`, `NIXL_WARN` only (does not fail)
5. `local.getType() == DRAM_SEG` (`NIXL_ERROR` + false)
6. **omitted:** Redis `remote.getType() == DRAM_SEG || OBJ_SEG`

Uses `nixlBackendEngine::localAgent` (protected member) instead of a fifth parameter, matching the brief signature. Logging includes match Redis today: `common/nixl_log.h` and `<absl/strings/str_format.h>`.

Brief bullet list mentioned the DRAM check before the agent warn; implementation follows Redis source order (warn, then local type) as required by “mirror redis `isValidPrepXferParams`”.

### Meson

Verbatim from the brief:

- `static_library('respkv_engine', …)` with `nixl_infra`, `nixl_common_dep`, `absl_strings_dep`
- `declare_dependency` does **not** re-export absl (header does not include it)
- No hiredis, libevent, or GLIDE
- `# subdir('redis')` / `# subdir('valkey')` left commented

Because `src/plugins/meson.build` is unchanged, this meson file is not yet part of the configure graph.

## Self-review

- [x] Header matches brief API and include (`backend/backend_engine.h`)
- [x] `getSupportedMems()` order `{OBJ_SEG, DRAM_SEG}`
- [x] Local-only, no notifs, connect/disconnect/unloadMD no-ops
- [x] Remote type **not** validated in the base
- [x] No REDIS/VALKEY plugin names or product clients in this task
- [x] Redis sources not moved; plugins meson still old path
- [x] Base does not link hiredis or GLIDE
- [x] No git commit / git add

## Tests / compile

- Redis unit tests **not run** (base not wired; Redis still the standalone plugin).
- Compile **not run**: Windows host has no `meson` / `g++` / `ninja` on PATH. Sources + meson were written to match the brief; a Linux meson configure after Task 2 wiring is required to compile-check.

## Suggested author commit

```text
respkv: add thin nixlRespKvEngine base

Introduce the RESP-family KV plugin base with shared capabilities and
prep helper. Product plugins are not wired yet.
```

Files:

- `src/plugins/respkv/respkv_engine.h`
- `src/plugins/respkv/respkv_engine.cpp`
- `src/plugins/respkv/meson.build`

## Concerns

1. **Compile unverified on this host** — no meson/g++ on Windows; `respkv/meson.build` is also not yet `subdir()`’d, so even a Linux tree would not build this target until a later task wires `src/plugins/meson.build`.
2. **No Redis UT this task** — expected; Task 2 moves Redis onto the base.
