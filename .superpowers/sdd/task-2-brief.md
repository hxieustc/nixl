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
