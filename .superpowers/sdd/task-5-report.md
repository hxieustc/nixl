# Task 5 Report: Family README

**Status:** DONE
**Working tree:** `redis5/nixl`
**Commit:** none (author commits; git add/commit skipped)

## Summary

Added `src/plugins/respkv/README.md` for the RESP-family plugins and pointed the REDIS and VALKEY READMEs at it. GLIDE build steps stay in the Valkey README. No source or meson edits. No commit.

## Files

| File | Change |
|---|---|
| `src/plugins/respkv/README.md` | Created. Family overview. |
| `src/plugins/respkv/redis/README.md` | One sentence after the intro pointing at `../README.md`. |
| `src/plugins/respkv/valkey/README.md` | Two sentences after the intro pointing at `../README.md`. Dependencies section unchanged. |

## What the family README covers

- `respkv` is the RESP command-plane family. Public names stay `REDIS` and `VALKEY`. For VALKEY, RESP is control (`BLOB.*`) and the data plane is RDMA.
- `nixlRespKvEngine` (`respkv_engine.h` / `.cpp`): local-only, `{OBJ_SEG, DRAM_SEG}`, connect/disconnect/unloadMD success, `loadLocalMD` identity, `validatePrepCommon` without a remote-segment rule. `respkv_engine_dep` does not link hiredis or GLIDE.
- Separate plugins: REDIS is hiredis TCP `SET`/`GET`/`EXISTS`; VALKEY is GLIDE `BLOB.RDMA_SET` / `BLOB.RDMA_GET`.
- Build: both names are in `all_plugins`. `src/plugins/meson.build` always `subdir('respkv')`. REDIS builds when hiredis, libevent, and libevent_pthreads are found; missing deps warn on a default build and error on explicit or static enable. VALKEY needs `-Dglide_path=<prefix>` (plus protoc and libprotobuf); otherwise the same warn-or-error gate.
- Pointers to `redis/README.md` and `valkey/README.md`.
- Future RESP-family plugins (Lavik) would be another subdirectory under `respkv/`.
- Link to `docs/superpowers/specs/2026-11-06-redis-valkey-kv-engine-design.md`.

Layout in the README matches the tree: base sources, `meson.build`, `redis/`, `valkey/`. No `lavik/` directory exists yet, so it is described as future only.

## Include paths

The task title says "include-path cleanup". The plan's file list for this task is the three READMEs. Tasks 2 and 3 already retargeted gtest and plugin includes to `src/plugins/respkv/...`. A search of `*.cpp`, `*.h`, and `meson.build` found no remaining `src/plugins/redis` or `src/plugins/valkey` include paths. Stale include guards noted in earlier reviews (`redis_client.h`, `valkey_backend.h`) were left as-is; they are outside this task's file list and do not affect the build.

## Suggested commit

```text
docs(respkv): add family README for REDIS and VALKEY plugins
```

Stage `src/plugins/respkv/README.md`, `src/plugins/respkv/redis/README.md`, and `src/plugins/respkv/valkey/README.md` with the earlier uncommitted respkv work, or as the docs commit the plan suggests. Do not commit `.superpowers/`.
