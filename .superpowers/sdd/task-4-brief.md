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
