/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file inmemkv_engine_impl.cpp
 * @brief nixlInMemKVEngineImpl - INMEMKV-specific nixlKVEngineImpl logic.
 */

#include "inmemkv_engine_impl.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <memory>

namespace {

/** Placeholder request handle for synchronous INMEMKV operations. */
class nixlInMemKVBackendReqH : public nixlBackendReqH {
public:
    nixlInMemKVBackendReqH() = default;
    ~nixlInMemKVBackendReqH() = default;
};

/** Backend metadata: maps NIXL devId to KV key string. */
class nixlInMemKVMetadata : public nixlBackendMD {
public:
    nixlInMemKVMetadata(uint64_t dev_id, std::string key)
        : nixlBackendMD(true),
          devId(dev_id),
          key(std::move(key)) {}

    ~nixlInMemKVMetadata() = default;

    uint64_t devId;
    std::string key;
};

bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    (void)remote_agent;
    (void)local_agent;
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    if (remote.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be DRAM_SEG, got %d",
                                      remote.getType());
        return false;
    }

    return true;
}

} // namespace

nixlInMemKVEngineImpl::nixlInMemKVEngineImpl(const nixlBackendInitParams *init_params)
    : nixlInMemKVEngineImpl(init_params, std::make_unique<InMemKVStore>()) {}

nixlInMemKVEngineImpl::nixlInMemKVEngineImpl(const nixlBackendInitParams *init_params,
                                               std::unique_ptr<iKVStore> store)
    : store_(std::move(store)),
      localAgent_(init_params->localAgent) {
    NIXL_INFO << "INMEMKV backend initialized (in-memory only)";
    NIXL_INFO << "INMEMKV: Local agent = " << localAgent_;
}

nixl_status_t
nixlInMemKVEngineImpl::registerMem(const nixlBlobDesc &mem,
                                   const nixl_mem_t &nixl_mem,
                                   nixlBackendMD *&out) {
    NIXL_INFO << "registerMem: type=" << nixl_mem << ", devId=" << mem.devId
              << ", metaInfo=" << (mem.metaInfo.empty() ? "<empty>" : mem.metaInfo);

    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        NIXL_ERROR << "registerMem: unsupported memory type " << nixl_mem;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    std::string key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
    NIXL_DEBUG << "registerMem: using key=" << key;

    auto inmemkv_md = std::make_unique<nixlInMemKVMetadata>(mem.devId, key);
    devIdToKey_[mem.devId] = key;
    NIXL_INFO << "registerMem: registered devId=" << mem.devId << " -> key=" << key;

    out = inmemkv_md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlInMemKVEngineImpl::deregisterMem(nixlBackendMD *meta) {
    auto *inmemkv_md = static_cast<nixlInMemKVMetadata *>(meta);
    if (inmemkv_md) {
        NIXL_INFO << "deregisterMem: removing devId=" << inmemkv_md->devId
                  << ", key=" << inmemkv_md->key;
        devIdToKey_.erase(inmemkv_md->devId);
        std::unique_ptr<nixlInMemKVMetadata> ptr(inmemkv_md);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlInMemKVEngineImpl::queryMem(const nixl_reg_dlist_t &descs,
                                std::vector<nixl_query_resp_t> &resp) const {
    resp.reserve(descs.descCount());

    for (auto &desc : descs) {
        std::string key = desc.metaInfo.empty() ? std::to_string(desc.devId) : desc.metaInfo;
        const bool exists = store_->exists(key);

        NIXL_DEBUG << "queryMem: key=" << key << ", exists=" << exists;
        resp.emplace_back(exists ? nixl_query_resp_t{nixl_b_params_t{}} : std::nullopt);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlInMemKVEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                const std::string &local_agent,
                                nixlBackendReqH *&handle,
                                const nixl_opt_b_args_t *opt_args) const {
    (void)opt_args;
    NIXL_INFO << "prepXfer: operation=" << (operation == NIXL_WRITE ? "WRITE" : "READ")
              << ", local_count=" << local.descCount()
              << ", remote_count=" << remote.descCount();

    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        NIXL_ERROR << "prepXfer: parameter validation failed";
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlInMemKVBackendReqH>();
    handle = req_h.release();
    NIXL_DEBUG << "prepXfer: request handle created, ready for postXfer";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlInMemKVEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH *&handle,
                                const nixl_opt_b_args_t *opt_args) const {
    NIXL_INFO << "postXfer: ========================================";
    NIXL_INFO << "postXfer: Starting " << (operation == NIXL_WRITE ? "WRITE (PUT)" : "READ (GET)")
              << " operation with " << local.descCount() << " descriptor(s)";
    (void)remote_agent;
    (void)handle;
    (void)opt_args;

    NIXL_INFO << "postXfer: Local operation - using key-value store interface";

    for (int i = 0; i < local.descCount(); ++i) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];

        NIXL_INFO << "postXfer: [Descriptor " << i << "/" << local.descCount() << "]";
        NIXL_INFO << "postXfer:   Local: addr=0x" << std::hex << local_desc.addr << std::dec
                  << ", len=" << local_desc.len;
        NIXL_INFO << "postXfer:   Remote: devId=" << remote_desc.devId;

        std::string key;
        if (remote_desc.metadataP) {
            auto *inmemkv_md = dynamic_cast<nixlInMemKVMetadata *>(remote_desc.metadataP);
            if (inmemkv_md) {
                key = inmemkv_md->key;
                NIXL_INFO << "postXfer:   Found key in metadata: " << key;
            } else {
                auto key_search = devIdToKey_.find(remote_desc.devId);
                if (key_search == devIdToKey_.end()) {
                    NIXL_ERROR << "postXfer: Key for devId " << remote_desc.devId << " not found";
                    return NIXL_ERR_INVALID_PARAM;
                }
                key = key_search->second;
                NIXL_DEBUG << "postXfer:   Found key in mapping: " << key;
            }
        } else {
            auto key_search = devIdToKey_.find(remote_desc.devId);
            if (key_search == devIdToKey_.end()) {
                NIXL_ERROR << "postXfer: Key for devId " << remote_desc.devId << " not found";
                return NIXL_ERR_INVALID_PARAM;
            }
            key = key_search->second;
            NIXL_DEBUG << "postXfer:   Found key in mapping (no metadata): " << key;
        }

        uintptr_t data_ptr = local_desc.addr;
        size_t data_len = local_desc.len;

        if (operation == NIXL_WRITE) {
            NIXL_INFO << "postXfer:   WRITE: Copying " << data_len
                      << " bytes from buffer to key=" << key;

            const auto write_status =
                store_->put(key, reinterpret_cast<const uint8_t *>(data_ptr), data_len);
            if (write_status != NIXL_SUCCESS) {
                NIXL_ERROR << "postXfer:   WRITE: Failed to store key=" << key
                           << ", status=" << write_status;
                return write_status;
            }

            NIXL_INFO << "postXfer:   WRITE: Successfully stored " << data_len
                      << " bytes in key=" << key;
        } else {
            NIXL_INFO << "postXfer:   READ: Retrieving data from key=" << key;

            size_t bytes_read = 0;
            const auto read_status =
                store_->get(key, reinterpret_cast<uint8_t *>(data_ptr), data_len, bytes_read);
            if (read_status == NIXL_ERR_BACKEND) {
                NIXL_ERROR << "postXfer:   READ: Key not found: " << key;
                return NIXL_ERR_BACKEND;
            }
            if (read_status != NIXL_SUCCESS) {
                NIXL_ERROR << "postXfer:   READ: Failed to read key=" << key
                           << ", status=" << read_status;
                return read_status;
            }

            if (bytes_read < data_len) {
                NIXL_WARN << "postXfer:   READ: Stored data size (" << bytes_read
                          << ") < requested size (" << data_len << ")";
                data_len = bytes_read;
            }

            NIXL_INFO << "postXfer:   READ: Successfully retrieved " << data_len
                      << " bytes from key=" << key;
        }
    }

    NIXL_INFO << "postXfer: ========================================";
    NIXL_INFO << "postXfer: All " << local.descCount()
              << " operations completed successfully";
    NIXL_INFO << "postXfer: Returning NIXL_SUCCESS (synchronous operations)";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlInMemKVEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    (void)handle;
    NIXL_DEBUG << "checkXfer: Operations are synchronous, always return SUCCESS";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlInMemKVEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    NIXL_INFO << "releaseReqH: Releasing request handle";
    delete handle;
    return NIXL_SUCCESS;
}
