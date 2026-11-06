# RESP KV Family (REDIS + VALKEY) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move REDIS under `src/plugins/respkv/`, add thin `nixlRespKvEngine`, then port VALKEY from #2577 as a sibling subclass—without merging clients or backend names.

**Architecture:** One family directory `src/plugins/respkv/` holds `nixlRespKvEngine` (capabilities + no-ops + prep helper only). `redis/` and `valkey/` subdirs each build their own plugin (`REDIS` / `VALKEY`). No shared hiredis/GLIDE link in the base.

**Tech Stack:** C++17/20 NIXL plugins, meson, hiredis+libevent (REDIS), GLIDE FFI+protobuf+libfabric (VALKEY), gtest.

**Spec:** `docs/superpowers/specs/2026-11-06-redis-valkey-kv-engine-design.md`

## Global Constraints

- Plugin names remain `"REDIS"` and `"VALKEY"` (never a single `"RESPKV"` backend).
- `getSupportedMems()` returns `{OBJ_SEG, DRAM_SEG}` (Redis order).
- Base target links **neither** hiredis nor GLIDE.
- Do **not** unify key rules, error codes, or `queryMem` shapes.
- Do **not** invent a shared `iKvClient`.
- **Do not auto-commit.** After each task, stage suggested files and leave a commit message for the author; the author runs `git commit` themselves.
- Working tree: `redis5/nixl`. Valkey reference sources: `nixlvalkey/nixl/src/plugins/valkey/` and related meson/plugin_manager/gtest edits.

---

## File map

| Path | Role |
|---|---|
| `src/plugins/respkv/respkv_engine.h` | `nixlRespKvEngine` declaration |
| `src/plugins/respkv/respkv_engine.cpp` | capabilities, no-ops, `validatePrepCommon` |
| `src/plugins/respkv/meson.build` | base dep + `subdir('redis')` / `subdir('valkey')` |
| `src/plugins/respkv/README.md` | family overview |
| `src/plugins/respkv/redis/*` | moved from `src/plugins/redis/` |
| `src/plugins/respkv/valkey/*` | ported from nixlvalkey |
| `src/plugins/meson.build` | `subdir('respkv')` instead of top-level redis |
| `meson.build` / `meson_options.txt` | `VALKEY` in `all_plugins`; `glide_path` |
| `src/core/meson.build` / `nixl_plugin_manager.cpp` | static VALKEY |
| `test/gtest/unit/redis/*` | include path updates |
| `test/gtest/unit/valkey/*` | ported tests |
| `test/nixl/test_plugin.cpp` | add `"VALKEY"` to known plugins list |

---

### Task 1: Add `nixlRespKvEngine` base

**Files:**
- Create: `src/plugins/respkv/respkv_engine.h`
- Create: `src/plugins/respkv/respkv_engine.cpp`
- Create: `src/plugins/respkv/meson.build` (base only first; redis subdir in Task 2)

**Interfaces:**
- Produces: `class nixlRespKvEngine : public nixlBackendEngine` with the overrides below; `protected: bool validatePrepCommon(...) const;`
- Consumes: `backend/backend_engine.h`, `common/nixl_log.h`

- [ ] **Step 1: Create header**

Create `src/plugins/respkv/respkv_engine.h`:

```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_RESPKV_RESPKV_ENGINE_H
#define NIXL_SRC_PLUGINS_RESPKV_RESPKV_ENGINE_H

#include "backend/backend_engine.h"

#include <string>

/** Thin base for RESP-family local KV plugins (REDIS, VALKEY, …). */
class nixlRespKvEngine : public nixlBackendEngine {
public:
    explicit nixlRespKvEngine(const nixlBackendInitParams *init_params)
        : nixlBackendEngine(init_params) {}

    ~nixlRespKvEngine() override = default;

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG};
    }

    nixl_status_t
    connect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

protected:
    /** Shared prep checks. Does not validate remote segment type. */
    bool
    validatePrepCommon(const nixl_xfer_op_t &operation,
                       const nixl_meta_dlist_t &local,
                       const nixl_meta_dlist_t &remote,
                       const std::string &remote_agent) const;
};

#endif
```

- [ ] **Step 2: Create implementation**

Create `src/plugins/respkv/respkv_engine.cpp` implementing `loadLocalMD` (null → `NIXL_ERR_INVALID_PARAM`, else identity) and `validatePrepCommon` with the same rules as today’s Redis `isValidPrepXferParams` **except** remote type (skip DRAM/OBJ remote check—subclasses add that):

- op must be `NIXL_READ` or `NIXL_WRITE`
- `local.descCount() > 0` and equal to `remote.descCount()`
- `local.getType() == DRAM_SEG`
- if `remote_agent != localAgent`, `NIXL_WARN` only

Include `common/nixl_log.h` and `absl/strings/str_format.h` as Redis does today.

- [ ] **Step 3: Minimal meson for the base**

Create `src/plugins/respkv/meson.build`:

```meson
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

respkv_engine_lib = static_library(
    'respkv_engine',
    sources: ['respkv_engine.cpp', 'respkv_engine.h'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, include_directories('.')],
    dependencies: [nixl_infra, nixl_common_dep, absl_strings_dep],
    install: false,
)

respkv_engine_dep = declare_dependency(
    link_with: respkv_engine_lib,
    include_directories: include_directories('.'),
    dependencies: [nixl_infra, nixl_common_dep],
)

# Product plugins added in later tasks:
# subdir('redis')
# subdir('valkey')
```

Do **not** change `src/plugins/meson.build` yet (still points at old `redis/`).

- [ ] **Step 4: Author commit (optional checkpoint)**

Suggested message:

```text
respkv: add thin nixlRespKvEngine base

Introduce the RESP-family KV plugin base with shared capabilities and
prep helper. Product plugins are not wired yet.
```

Files: `src/plugins/respkv/respkv_engine.h`, `respkv_engine.cpp`, `meson.build`.

---

### Task 2: Move REDIS under `respkv/redis/` and inherit base

**Files:**
- Move: `src/plugins/redis/*` → `src/plugins/respkv/redis/`
- Modify: `src/plugins/respkv/redis/redis_backend.h` (inherit `nixlRespKvEngine`; remove duplicated overrides)
- Modify: `src/plugins/respkv/redis/redis_backend.cpp` (use `validatePrepCommon`; keep remote-type checks)
- Modify: `src/plugins/respkv/redis/meson.build` (depend on `respkv_engine_dep`; fix include paths if needed)
- Modify: `src/plugins/respkv/meson.build` (uncomment/add `subdir('redis')`)
- Modify: `src/plugins/meson.build` (`subdir('respkv')` instead of `subdir('redis')`)
- Modify: `test/gtest/unit/redis/meson.build` (include `../../../../src/plugins/respkv/redis` and link `respkv_engine_dep` if required via `redis_backend_interface`)

**Interfaces:**
- Consumes: `nixlRespKvEngine` from Task 1
- Produces: `class nixlRedisKVEngine : public nixlRespKvEngine` still exporting the same public transfer APIs; `redis_backend_interface` meson variable still exists for core/gtest

- [ ] **Step 1: Move the tree**

From `redis5/nixl`:

```bash
git mv src/plugins/redis src/plugins/respkv/redis
```

If `git mv` cannot move onto an existing `respkv`, move contents:

```bash
mkdir -p src/plugins/respkv/redis
git mv src/plugins/redis/* src/plugins/respkv/redis/
rmdir src/plugins/redis   # if empty
```

- [ ] **Step 2: Change the engine base class**

In `redis_backend.h`:

- `#include "respkv_engine.h"`
- `class nixlRedisKVEngine : public nixlRespKvEngine`
- **Delete** overrides now provided by the base: `supportsRemote/Local/Notif`, `getSupportedMems`, `connect`, `disconnect`, `unloadMD`, `loadLocalMD`
- Keep ctors; call `nixlRespKvEngine(init_params)` in initializer lists (both constructors)

Example ctor:

```cpp
explicit nixlRedisKVEngine(const nixlBackendInitParams *init_params);
// impl:
nixlRedisKVEngine::nixlRedisKVEngine(const nixlBackendInitParams *init_params)
    : nixlRespKvEngine(init_params),
      redisClient_(std::make_shared<RedisConnectionPool>(
          RedisConfig::fromBackendParams(getInitCustomParams(init_params)))) {}
```

(Adapt to the file’s actual helper names—do not invent new config parsing.)

- [ ] **Step 3: Use `validatePrepCommon` in prep**

In `redis_backend.cpp`, replace the duplicated common checks inside `isValidPrepXferParams` (or the `prepXfer` call path) with:

```cpp
if (!validatePrepCommon(operation, local, remote, remote_agent)) {
    return NIXL_ERR_INVALID_PARAM; // or false, matching current helper shape
}
// Keep Redis-only remote type check:
if (remote.getType() != DRAM_SEG && remote.getType() != OBJ_SEG) {
    ...
}
```

Do not change Redis remote DRAM|OBJ policy.

- [ ] **Step 4: Wire meson**

`src/plugins/meson.build`: replace

```meson
if enabled_plugins.get('REDIS')
    subdir('redis')
endif
```

with:

```meson
subdir('respkv')
```

`src/plugins/respkv/meson.build`: after defining `respkv_engine_dep`, add:

```meson
if enabled_plugins.get('REDIS')
    subdir('redis')
endif
```

`src/plugins/respkv/redis/meson.build`: add `respkv_engine_dep` to `redis_plugin_deps` / `declare_dependency`, and set include dirs so `#include "respkv_engine.h"` resolves (parent `respkv` include or `include_directories('..')`).

Update `test/gtest/unit/redis/meson.build` include path from `../../../../src/plugins/redis` to `../../../../src/plugins/respkv/redis` (and add `../../../../src/plugins/respkv` if headers need it).

- [ ] **Step 5: Build and run Redis unit tests**

```bash
# Linux build dir example
ninja -C build-ut test/gtest/unit/unit
./build-ut/test/gtest/unit/unit --gtest_filter='redisConfigTest.*:redisEngineTest.*'
```

Expected: all Redis tests PASS, including `ReportsRedisBackendCapabilities` asserting `{OBJ_SEG, DRAM_SEG}`.

- [ ] **Step 6: Author commit**

Suggested message:

```text
redis: move under respkv and inherit nixlRespKvEngine

Place the REDIS plugin in the RESP-family tree and drop capability
overrides now provided by the shared base.
```

---

### Task 3: Port VALKEY sources into `respkv/valkey/`

**Files:**
- Create: `src/plugins/respkv/valkey/` (copy from `nixlvalkey/nixl/src/plugins/valkey/`)
- Create/adjust: `src/plugins/respkv/valkey/meson.build` (same as #2577, plus `respkv_engine_dep` in Task 4)
- Modify: `meson.build` (`VALKEY` in `all_plugins`)
- Modify: `meson_options.txt` (add `glide_path`)
- Modify: `src/plugins/respkv/meson.build` (`subdir('valkey')` when enabled)
- Modify: `src/core/meson.build` (static VALKEY deps)
- Modify: `src/core/nixl_plugin_manager.cpp` (`STATIC_PLUGIN_VALKEY`)
- Modify: `test/nixl/test_plugin.cpp` (add `"VALKEY"`)
- Create: `test/gtest/unit/valkey/` from nixlvalkey
- Modify: `test/gtest/unit/meson.build`

**Interfaces:**
- Produces: `valkey_backend_interface` meson dependency; plugin name `"VALKEY"`
- Consumes: unchanged GLIDE FFI from #2577 (still inherits `nixlBackendEngine` until Task 4)

- [ ] **Step 1: Copy Valkey plugin sources**

Copy these files from `nixlvalkey/nixl/src/plugins/valkey/` into `redis5/nixl/src/plugins/respkv/valkey/`:

- `valkey_backend.h`, `valkey_backend.cpp`
- `valkey_client.h`, `valkey_client.cpp`
- `valkey_plugin.cpp`
- `meson.build`
- `README.md`

Keep behavior identical for this task (still `: public nixlBackendEngine`).

- [ ] **Step 2: Root meson / options**

In root `meson.build`, extend `all_plugins` to include `'VALKEY'` next to `'REDIS'` (match nixlvalkey order).

In `meson_options.txt` add:

```meson
option('glide_path', type: 'string', value: '', description: 'valkey-glide prefix for the VALKEY plugin: lib/libglide_ffi.a, include/glide_ffi.h, proto/connection_request.proto')
```

- [ ] **Step 3: Plugin manager + core link**

In `src/core/nixl_plugin_manager.cpp`, after REDIS static registration, add:

```cpp
#ifdef STATIC_PLUGIN_VALKEY
    NIXL_REGISTER_STATIC_PLUGIN(Backend, VALKEY)
#endif
```

In `src/core/meson.build`, after REDIS static block:

```meson
if 'VALKEY' in static_plugins
    nixl_lib_deps += [ valkey_backend_interface ]
endif
```

(If nixlvalkey passes `valkey_link_args` via the interface, keep that exact pattern so `--exclude-libs,libglide_ffi.a` stays on the VALKEY target.)

- [ ] **Step 4: Enable subdir + gtest + test_plugin list**

`src/plugins/respkv/meson.build`:

```meson
if enabled_plugins.get('VALKEY')
    subdir('valkey')
endif
```

Copy `nixlvalkey/nixl/test/gtest/unit/valkey/` → `redis5/nixl/test/gtest/unit/valkey/`.  
Update includes to `../../../../src/plugins/respkv/valkey`.

`test/gtest/unit/meson.build`:

```meson
if is_variable('valkey_backend_interface')
    subdir('valkey')
    unit_test_deps += [valkey_unit_test_dep]
endif
```

`test/nixl/test_plugin.cpp`: add `"VALKEY"` next to `"REDIS"` in the known-plugins list.

- [ ] **Step 5: Configure build**

Without `glide_path`, reconfigure and confirm meson warns and skips VALKEY (default all-plugins).  
With `-Dglide_path=...` (GLIDE built `--features rdma`), confirm VALKEY builds.

```bash
meson setup build-ut --buildtype=debug   # or reconfigure
# expect warning about VALKEY deps if glide_path empty
```

- [ ] **Step 6: Run unit tests that do not need a live RDMA server**

```bash
./build-ut/test/gtest/unit/unit --gtest_filter='valkey*'
```

Expected: config / init-failure style tests PASS when built; if VALKEY was skipped, this filter runs nothing—document that in the PR notes.

- [ ] **Step 7: Author commit**

Suggested message:

```text
valkey: add VALKEY plugin under respkv from GLIDE RDMA port

Port the Valkey storage backend (BLOB.RDMA_*) into the RESP-family
tree with glide_path wiring. Inheritance onto nixlRespKvEngine follows.
```

---

### Task 4: Make `nixlValkeyEngine` inherit `nixlRespKvEngine`

**Files:**
- Modify: `src/plugins/respkv/valkey/valkey_backend.h`
- Modify: `src/plugins/respkv/valkey/valkey_backend.cpp` (optional: use `validatePrepCommon`)
- Modify: `src/plugins/respkv/valkey/meson.build` (add `respkv_engine_dep`)

**Interfaces:**
- Consumes: `nixlRespKvEngine`
- Produces: `class nixlValkeyEngine : public nixlRespKvEngine`

- [ ] **Step 1: Change inheritance**

In `valkey_backend.h`:

- `#include "respkv_engine.h"`
- `class nixlValkeyEngine : public nixlRespKvEngine`
- Remove duplicated `supports*`, `getSupportedMems` / `supportedMems()` if redundant—**keep** `supportedMems()` only if `valkey_plugin.cpp` still calls it; if so, implement as:

```cpp
static nixl_mem_list_t supportedMems() { return {OBJ_SEG, DRAM_SEG}; }
```

matching Redis order (update plugin.cpp if it listed `{DRAM, OBJ}`).

- Remove `connect` / `disconnect` / `unloadMD` / `loadLocalMD` overrides if identical to base (prefer base `loadLocalMD` with null check).

Ctor initializer: `: nixlRespKvEngine(init_params), config_(...)`.

- [ ] **Step 2: Prep path**

In `prepXfer`, call `validatePrepCommon` then enforce Valkey-only `remote.getType() == OBJ_SEG` (and any existing empty-list policy Valkey had—do not silently adopt Redis empty-list rejection if tests depend on old Valkey behavior; prefer keeping Valkey semantics unless a test requires otherwise). Prefer **documenting** any intentional difference in a one-line comment.

- [ ] **Step 3: meson dep**

Add `respkv_engine_dep` to `valkey_plugin_deps`.

- [ ] **Step 4: Rebuild + filter tests**

```bash
ninja -C build-ut
./build-ut/test/gtest/unit/unit --gtest_filter='redis*:valkey*'
```

Expected: Redis still green; Valkey tests green when plugin built.

- [ ] **Step 5: Author commit**

Suggested message:

```text
valkey: inherit nixlRespKvEngine

Share RESP-family capabilities with REDIS while keeping GLIDE RDMA
transfer and registration logic in the Valkey subclass.
```

---

### Task 5: Family README and include-path cleanup

**Files:**
- Create: `src/plugins/respkv/README.md`
- Modify: `src/plugins/respkv/redis/README.md` (point up to family README)
- Modify: `src/plugins/respkv/valkey/README.md` (same; keep GLIDE build instructions)

- [ ] **Step 1: Write family README**

Cover:

- What `respkv` means (RESP command plane; VALKEY data plane is RDMA)
- When to use `REDIS` vs `VALKEY`
- Directory layout
- Link to design spec under `docs/superpowers/specs/`

- [ ] **Step 2: Author commit**

Suggested message:

```text
docs(respkv): add family README for REDIS and VALKEY plugins
```

---

## Spec coverage checklist

| Spec item | Task |
|---|---|
| `nixlRespKvEngine` thin base | Task 1 |
| `{OBJ_SEG, DRAM_SEG}` order | Tasks 1–2, 4 |
| Nest under `src/plugins/respkv/{redis,valkey}` | Tasks 2–3 |
| REDIS inherit + keep client | Task 2 |
| VALKEY port + glide_path | Task 3 |
| VALKEY inherit | Task 4 |
| No shared client / fat base | All tasks |
| Family docs | Task 5 |
| Author commits only | Global + each task commit step |

## Placeholder / consistency review

- Base name consistently `nixlRespKvEngine` / files `respkv_engine.*` / dir `respkv/`.
- Meson variable `respkv_engine_dep` used by both children.
- Plugin strings `"REDIS"` / `"VALKEY"` unchanged.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-11-06-respkv-redis-valkey.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — run tasks in this session with checkpoints  

Which approach? (Remember: I will **not** run `git commit` for you; I will only prepare the changes and suggested messages.)
