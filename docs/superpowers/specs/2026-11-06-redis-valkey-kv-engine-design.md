# Design: Shared `nixlRespKvEngine` under `src/plugins/respkv/`

**Date:** 2026-11-06  
**Status:** Approved for planning (scheme 1; directory naming revised)  
**Working tree:** `redis5/nixl`  
**Related:** existing REDIS plugin; Valkey plugin from [ai-dynamo/nixl#2577](https://github.com/ai-dynamo/nixl/pull/2577); future Lavik Redis backend

## Summary

Introduce a **thin** base class `nixlRespKvEngine` for the **RESP-protocol KV family** of local NIXL storage backends (local-only, DRAM+OBJ, no notifications). Product plugins live as **subdirectories** under one parent folder and keep separate NIXL backend names, clients, configs, and transfer semantics:

| Plugin name | Source dir | Transport | Client |
|---|---|---|---|
| `REDIS` | `src/plugins/respkv/redis/` | TCP RESP `SET`/`GET`/`EXISTS` | hiredis + libevent (`iRedisClient`) |
| `VALKEY` | `src/plugins/respkv/valkey/` | Server-initiated RDMA via `BLOB.RDMA_*` | GLIDE FFI (`--features rdma`) |
| Future (e.g. Lavik) | `src/plugins/respkv/lavik/` | TBD (expected RESP-family) | Own client; subclass `nixlRespKvEngine` if the NIXL surface matches |

This mirrors the **shape** of GDS / GDS_MT (one family directory, multiple plugin entry points), not OBJ’s single-plugin registry. Shared code is deliberately small: Redis TCP value copies and Valkey GLIDE RDMA are not the same data path.

**Naming note:** `respkv` means “RESP-command KV backends.” For VALKEY, RESP carries control (`BLOB.*`); the payload moves by RDMA, not as RESP bulk strings. It does **not** mean every byte of every value travels over RESP.

## Decisions locked in

1. **Scheme:** thin base + separate subclasses/plugins (not a fat base, not one plugin with `transport=`).
2. **`getSupportedMems()` order:** `{OBJ_SEG, DRAM_SEG}` — match current REDIS engine/tests.
3. **Family directory:** `src/plugins/respkv/` with product subdirs `redis/`, `valkey/`, (later `lavik/`).
4. **Base class name:** `nixlRespKvEngine`.
5. **Public backend names unchanged:** `"REDIS"` and `"VALKEY"` (not a single `"RESPKV"` backend).
6. **No automatic git commits**; the author commits locally.

### Layout

```text
src/plugins/respkv/
  respkv_engine.h
  respkv_engine.cpp       # nixlRespKvEngine (no hiredis, no GLIDE)
  meson.build             # declare base dep; subdir('redis'); subdir('valkey')
  README.md               # family overview: when REDIS vs VALKEY
  redis/
    redis_backend.*
    redis_client.*
    redis_plugin.cpp      # registers "REDIS"
    meson.build
    README.md
  valkey/
    valkey_backend.*
    valkey_client.*
    valkey_plugin.cpp     # registers "VALKEY"
    meson.build
    README.md
  lavik/                  # future
```

Top-level `src/plugins/meson.build` gains `subdir('respkv')` instead of a top-level `subdir('redis')` once the move is done. Existing `src/plugins/redis/` is **moved** into `respkv/redis/` (not copied forever as a duplicate).

If a future backend is **not** RESP-family, do not force it under `respkv/`; give it its own plugin directory.

## Goals / non-goals

### Goals

- One thin C++ base for shared NIXL capabilities and no-ops.
- Group REDIS / VALKEY / future Lavik Redis under one source family.
- Keep REDIS behavior and unit tests stable after the move + inherit.
- Port VALKEY from #2577 with GLIDE RDMA as a hard dependency.

### Non-goals

- Shared `iRedisClient` / connection pool for Valkey.
- Unifying key derivation, error codes, or `queryMem` response shapes.
- RESP payload fallback for Valkey without fabric (follow-up).
- `VRAM_SEG` for Valkey (needs GLIDE device registration).
- nixlbench VALKEY support in the first implementation pass.
- Merging REDIS and VALKEY into a single plugin `.so` / backend name.

## Architecture

```text
nixlBackendEngine
└── nixlRespKvEngine                      # capabilities + no-ops + prep helper
    ├── nixlRedisKVEngine                 # REDIS
    │     └── iRedisClient / RedisConnectionPool
    ├── nixlValkeyEngine                  # VALKEY
    │     └── nixl::valkey::client (GLIDE) + workers
    └── (future) nixlLavikEngine          # example
```

```text
                    ┌──────────────────────────┐
                    │   nixlRespKvEngine       │
                    │ supports*, mems, no-ops  │
                    │ validatePrepCommon()     │
                    └────────────┬─────────────┘
               ┌─────────────────┼─────────────────┐
               ▼                 ▼                 ▼
        nixlRedisKVEngine  nixlValkeyEngine   (Lavik…)
        hiredis TCP        GLIDE BLOB.RDMA_*
```

### What the base owns

| API | Behavior |
|---|---|
| `supportsRemote()` | `false` |
| `supportsLocal()` | `true` |
| `supportsNotif()` | `false` |
| `getSupportedMems()` | `{OBJ_SEG, DRAM_SEG}` |
| `connect` / `disconnect` / `unloadMD` | `NIXL_SUCCESS` |
| `loadLocalMD` | null → `NIXL_ERR_INVALID_PARAM`; else identity |

**Protected helper (optional, non-virtual template):**

- `validatePrepCommon(operation, local, remote, remote_agent, local_agent)`  
  Checks: op is READ/WRITE; non-empty equal counts; local is `DRAM_SEG`; warn if `remote_agent != local_agent`.  
  Does **not** constrain remote segment type (REDIS allows DRAM|OBJ; VALKEY requires OBJ only).

### What subclasses own

| Concern | REDIS | VALKEY |
|---|---|---|
| Client | `iRedisClient`, async promises | Blocking GLIDE + RDMA regions |
| `registerMem` / `deregisterMem` | Logical key / addr map | Multi-worker `registerHost` |
| `queryMem` | `EXISTS` → empty params | `BLOB.INFO LEN` → `size` |
| `prepXfer` remote rule | DRAM or OBJ | OBJ only; whole-object later in post |
| `postXfer` / `checkXfer` / `releaseReqH` | Futures | Job queue + pending + watchdog |
| Config | `RedisConfig` + `REDIS_*` env | `nixlValkeyConfig` (no Redis env) |
| meson | hiredis, libevent | `glide_path`, protobuf, `--exclude-libs` |
| Init failure | Pool ctor may throw | `initErr` if `!rdmaUsable()` or probe fails |

**Do not** introduce a shared `iKvClient { putAsync, getAsync }` implemented by both. Valkey’s API is `registerHost` + blocking `transfer` + cancel; fitting it under Redis’s pointer+length SET/GET would hide RDMA registration and whole-object rules.

## Plugin wiring

- Two backend names: `"REDIS"` and `"VALKEY"`.
- Two `*_plugin.cpp` creators and option maps (do not merge options).
- `all_plugins` includes both; `glide_path` option for VALKEY.
- Static plugin hooks in `nixl_plugin_manager.cpp` for both.
- gtest: keep `test/gtest/unit/redis` (paths/includes updated); add `unit/valkey` from #2577.
- `respkv/meson.build` exposes a base dependency with **no** hiredis/GLIDE link; each child links its own libs.

Default all-plugin build: skip VALKEY with a warning if `glide_path` deps are missing; explicit `-Denable_plugins=VALKEY` must error (same as #2577).

## Implementation phases

| Phase | Work | Success |
|---|---|---|
| **P0** | Create `src/plugins/respkv/` + `nixlRespKvEngine`; **move** existing redis sources to `respkv/redis/`; inherit base; fix meson/includes/tests | Redis unit tests green; plugin name still `REDIS` |
| **P1** | Port valkey into `respkv/valkey/` + meson / plugin_manager / gtest wiring | VALKEY builds when `glide_path` set; skips otherwise |
| **P2** | `nixlValkeyEngine` inherits `nixlRespKvEngine` | Same Valkey behavior; capabilities from base |
| **P3** | Family README: REDIS vs VALKEY; note future Lavik under `respkv/` | Docs consistent |

P0 before large Valkey import keeps Redis regressions small. P1 may be a separate commit/PR from P0.

## Testing

- **REDIS:** existing gtest (`redisConfigTest`, `redisEngineTest`, event-loop wrap). After P0, capabilities still assert `{OBJ_SEG, DRAM_SEG}`.
- **VALKEY:** port #2577 unit tests (config, init failure without RDMA/server). No requirement to mock GLIDE in P1/P2.
- Do not unify error-code expectations across the two suites.

## Risks

| Risk | Mitigation |
|---|---|
| Fat base couples transports | Base limited to capabilities + no-ops + prep helper |
| `respkv` pulls in hiredis or GLIDE | Base target has zero product-library deps |
| Name implies all payloads are RESP | Document: RESP = command plane; VALKEY data plane is RDMA |
| Key/error unification pressure | Explicit non-goal; keep per-plugin semantics |
| Move breaks include paths / meson | P0 is move + inherit only; run full redis gtest |
| VALKEY draft / pre-1.0 GLIDE | Document; pin build to documented GLIDE rc until stable |

## Out of scope follow-ups

- Lavik Redis plugin design (separate spec when requirements exist).
- Valkey RESP fallback without fabric.
- Valkey `VRAM_SEG`.
- nixlbench VALKEY backend.
- Shared mock client interface (unlikely to be one interface).

## References

- Current REDIS plugin (to move): `src/plugins/redis/`
- VALKEY reference tree: `nixlvalkey/nixl/src/plugins/valkey/` and PR #2577
- Precedent: `src/plugins/cuda_gds/` (one directory, `gds_plugin` + `gds_mt_plugin`)
- Valkey Over RDMA (`valkey/src/rdma.c`) is **not** used by the VALKEY NIXL plugin; that plugin uses GLIDE + `BLOB.RDMA_*` only
