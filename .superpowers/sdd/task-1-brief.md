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
