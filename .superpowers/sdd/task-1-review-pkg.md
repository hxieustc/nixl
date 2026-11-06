# Task 1 review package (uncommitted)\nCommits: none\n## Stat\n src/plugins/respkv/respkv_engine.h  (1706 bytes)\n src/plugins/respkv/respkv_engine.cpp  (1854 bytes)\n src/plugins/respkv/meson.build  (698 bytes)\n\n## Full files as added\n\n### src/plugins/respkv/respkv_engine.h\n`\n/*
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
\n`\n\n### src/plugins/respkv/respkv_engine.cpp\n`\n/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "respkv_engine.h"

#include "common/nixl_log.h"

#include <absl/strings/str_format.h>

nixl_status_t
nixlRespKvEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    if (!input) {
        output = nullptr;
        return NIXL_ERR_INVALID_PARAM;
    }
    output = input;
    return NIXL_SUCCESS;
}

bool
nixlRespKvEngine::validatePrepCommon(const nixl_xfer_op_t &operation,
                                     const nixl_meta_dlist_t &local,
                                     const nixl_meta_dlist_t &remote,
                                     const std::string &remote_agent) const {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (local.descCount() == 0) {
        NIXL_ERROR << "Error: Transfer descriptor lists must not be empty";
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local and remote descriptor counts must match (%d != %d)",
            local.descCount(),
            remote.descCount());
        return false;
    }

    if (remote_agent != localAgent) {
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            localAgent,
            remote_agent);
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    return true;
}
\n`\n\n### src/plugins/respkv/meson.build\n`\n# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
\n`\n