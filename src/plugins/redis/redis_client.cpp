/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "redis_client.h"
#include "common/backend.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>

RedisConfig
RedisConfig::fromBackendParams(const nixl_b_params_t *custom_params) {
    using nixl::getBackendParamOptional;
    using nixl::config::getValueOptional;

    auto resolve = [&]<typename T>(const char *param, const char *env) -> std::optional<T> {
        if (auto v = getBackendParamOptional<T>(custom_params, param)) {
            return v;
        }
        return getValueOptional<T>(env);
    };

    RedisConfig config;
    config.host = resolve.operator()<std::string>("host", "REDIS_HOST").value_or("localhost");
    config.username = resolve.operator()<std::string>("username", "REDIS_USERNAME").value_or("");
    config.password = resolve.operator()<std::string>("password", "REDIS_PASSWORD").value_or("");

    if (const auto port = resolve.operator()<int>("port", "REDIS_PORT")) {
        if (*port <= 0 || *port > 65535) {
            throw std::invalid_argument("Redis port out of range [1,65535]: " +
                                        std::to_string(*port));
        }
        config.port = *port;
    }
    if (const auto db = resolve.operator()<int>("db", "REDIS_DB")) {
        if (*db < 0) {
            throw std::invalid_argument("Redis db must be >= 0, got: " + std::to_string(*db));
        }
        config.db = *db;
    }
    if (const auto pool = resolve.operator()<int>("pool_size", "REDIS_POOL_SIZE")) {
        if (*pool <= 0) {
            throw std::invalid_argument("Redis pool_size must be > 0, got: " +
                                        std::to_string(*pool));
        }
        config.pool_size = *pool;
    }

    if (!config.username.empty() && config.password.empty()) {
        throw std::invalid_argument("Redis username requires a password");
    }

    return config;
}

#ifdef HAVE_HIREDIS_ASYNC

namespace {

using redis_event_task_t = std::function<void()>;

void
runEventTask(evutil_socket_t, short, void *arg) {
    std::unique_ptr<redis_event_task_t> task(static_cast<redis_event_task_t *>(arg));
    try {
        (*task)();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Redis: event task threw: " << e.what();
    }
    catch (...) {
        NIXL_ERROR << "Redis: event task threw unknown exception";
    }
}

bool
checkRedisReplyOk(redisReply *reply, const char *command) {
    if (!reply) {
        NIXL_ERROR << absl::StrFormat("Redis %s: no reply", command);
        return false;
    }
    if (reply->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis %s error: %s", command, reply->str);
        return false;
    }
    if (reply->type == REDIS_REPLY_STATUS && strcmp(reply->str, "OK") != 0) {
        NIXL_ERROR << absl::StrFormat("Redis %s unexpected status: %s", command, reply->str);
        return false;
    }
    return true;
}

std::once_flag evthreadOnce;

struct CallbackContext {
    uintptr_t data_ptr;
    size_t data_len;
    std::shared_ptr<std::promise<nixl_status_t>> promise_ptr;
    std::atomic<int> *inFlight;
};

} // namespace

struct RedisConnectionPool::SubConn {
    RedisConnectionPool::Slot *slot = nullptr;
    int idx = 0;
    redisAsyncContext *asyncCtx = nullptr;
    std::atomic<bool> connected{false};
};

// Slot: one libevent thread serving kConns pipelined async TCP connections.
struct RedisConnectionPool::Slot {
    static constexpr int kConns = 2;

    RedisConnectionPool *pool = nullptr;
    RedisConnectionPool::SubConn subconns[kConns];
    std::atomic<int> initCount{0};
    std::atomic<int> initOkCount{0};
    std::atomic<bool> initDone{false};
    std::atomic<bool> initSucceeded{false};
    std::atomic<int> inFlight{0};
    std::atomic<int> nextConn{0};
    event_base *eventBase = nullptr;
    std::thread eventLoopThread;
};

RedisConnectionPool::RedisConnectionPool(RedisConfig config) : config_(std::move(config)) {
    std::call_once(evthreadOnce, [] {
        if (evthread_use_pthreads() != 0) {
            throw std::runtime_error("evthread_use_pthreads() failed");
        }
    });

    const int N = config_.pool_size;
    slots_.reserve(N);
    for (int i = 0; i < N; ++i) {
        slots_.push_back(std::make_unique<Slot>());
        slots_.back()->pool = this;
    }

    // Each slot gets its own event base and event loop thread so callbacks run in parallel.
    // On error, stopEventLoop() cleans up whatever was successfully started.
    try {
        for (auto &s : slots_) {
            initSlotAsyncCtx(*s);
        }
    }
    catch (...) {
        stopEventLoop();
        throw;
    }

    // Wait for every slot to complete initialization (parallel: ~1 RTT regardless of N).
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    for (auto &s : slots_) {
        while (!s->initDone.load()) {
            if (std::chrono::steady_clock::now() > deadline) {
                stopEventLoop();
                throw std::runtime_error("Redis connection pool initialization timed out");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    const bool allOk =
        std::all_of(slots_.begin(), slots_.end(), [](const std::unique_ptr<Slot> &s) {
            return s->initSucceeded.load();
        });
    if (!allOk) {
        stopEventLoop();
        throw std::runtime_error("Failed to initialize one or more Redis connections in pool");
    }

    // Shared sync connection for EXISTS (queryMem is single-threaded, one connection suffices).
    connectSyncContext();

    // Worker threads perform the GET reply memcpy off the event loop thread so that
    // large transfers do not stall other pending callbacks.
    workers_.reserve(N);
    for (int i = 0; i < N; ++i) {
        workers_.emplace_back([this]() { workerLoop(); });
    }

    NIXL_INFO << absl::StrFormat(
        "Redis connection pool ready: %d connections at %s:%d (db=%d, slots=%d, conns_per_slot=%d, workers=%d)",
        N * Slot::kConns,
        config_.host,
        config_.port,
        config_.db,
        N,
        Slot::kConns,
        N);
}

RedisConnectionPool::~RedisConnectionPool() {
    stopEventLoop(); // no new callbacks after this; no new work enters the queue
    {
        std::lock_guard<std::mutex> lock(workMutex_);
        stopWorkers_.store(true);
    }
    workCv_.notify_all();
    for (auto &w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    if (syncCtx_) {
        redisFree(syncCtx_);
        syncCtx_ = nullptr;
    }
}

void
RedisConnectionPool::initSlotAsyncCtx(Slot &slot) {
    slot.eventBase = event_base_new();
    if (!slot.eventBase) {
        throw std::runtime_error("Failed to create event base for slot");
    }

    for (int i = 0; i < Slot::kConns; ++i) {
        SubConn &sc = slot.subconns[i];
        sc.slot = &slot;
        sc.idx = i;

        sc.asyncCtx = redisAsyncConnect(config_.host.c_str(), config_.port);
        if (!sc.asyncCtx || sc.asyncCtx->err) {
            std::string msg = sc.asyncCtx ?
                absl::StrFormat("Failed to connect to Redis: %s", sc.asyncCtx->errstr) :
                "Failed to allocate Redis async context";
            if (sc.asyncCtx) {
                redisAsyncFree(sc.asyncCtx);
                sc.asyncCtx = nullptr;
            }
            for (int j = 0; j < i; ++j) {
                redisAsyncFree(slot.subconns[j].asyncCtx);
                slot.subconns[j].asyncCtx = nullptr;
            }
            event_base_free(slot.eventBase);
            slot.eventBase = nullptr;
            throw std::runtime_error(msg);
        }

        sc.asyncCtx->data = &sc;

        if (redisLibeventAttach(sc.asyncCtx, slot.eventBase) != REDIS_OK) {
            std::string msg =
                absl::StrFormat("Failed to attach Redis to event base: %s", sc.asyncCtx->errstr);
            redisAsyncFree(sc.asyncCtx);
            sc.asyncCtx = nullptr;
            for (int j = 0; j < i; ++j) {
                redisAsyncFree(slot.subconns[j].asyncCtx);
                slot.subconns[j].asyncCtx = nullptr;
            }
            event_base_free(slot.eventBase);
            slot.eventBase = nullptr;
            throw std::runtime_error(msg);
        }

        redisAsyncSetConnectCallback(sc.asyncCtx, connectCallback);
        redisAsyncSetDisconnectCallback(sc.asyncCtx, disconnectCallback);
    }

    slot.eventLoopThread = std::thread([&slot]() { event_base_dispatch(slot.eventBase); });
}

void
RedisConnectionPool::connectSyncContext() {
    // 1 s: short enough not to stall queryMem callers on reconnect while
    // still giving a slow server a reasonable chance to accept.
    struct timeval timeout = {1, 0};
    syncCtx_ = redisConnectWithTimeout(config_.host.c_str(), config_.port, timeout);
    if (!syncCtx_ || syncCtx_->err) {
        std::string err_msg = syncCtx_ ? syncCtx_->errstr : "allocation failed";
        if (syncCtx_) {
            redisFree(syncCtx_);
            syncCtx_ = nullptr;
        }
        NIXL_WARN << absl::StrFormat(
            "Sync Redis connection failed (%s:%d): %s; queryMem will return errors",
            config_.host,
            config_.port,
            err_msg);
        return;
    }

    if (!config_.password.empty()) {
        redisReply *reply = config_.username.empty() ?
            static_cast<redisReply *>(redisCommand(syncCtx_, "AUTH %s", config_.password.c_str())) :
            static_cast<redisReply *>(redisCommand(
                syncCtx_, "AUTH %s %s", config_.username.c_str(), config_.password.c_str()));
        if (!checkRedisReplyOk(reply, "AUTH")) {
            freeReplyObject(reply);
            redisFree(syncCtx_);
            syncCtx_ = nullptr;
            NIXL_WARN << "Sync Redis AUTH failed; queryMem will return errors";
            return;
        }
        freeReplyObject(reply);
    }

    if (config_.db != 0) {
        redisReply *reply =
            static_cast<redisReply *>(redisCommand(syncCtx_, "SELECT %d", config_.db));
        if (!checkRedisReplyOk(reply, "SELECT")) {
            freeReplyObject(reply);
            redisFree(syncCtx_);
            syncCtx_ = nullptr;
            NIXL_WARN << "Sync Redis SELECT failed; queryMem will return errors";
            return;
        }
        freeReplyObject(reply);
    }
}

void
RedisConnectionPool::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(workMutex_);
            workCv_.wait(lock, [this] { return stopWorkers_.load() || !workQueue_.empty(); });
            if (stopWorkers_.load() && workQueue_.empty()) {
                return;
            }
            task = std::move(workQueue_.front());
            workQueue_.pop();
        }
        try {
            task();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Redis: worker task threw: " << e.what();
        }
        catch (...) {
            NIXL_ERROR << "Redis: worker task threw unknown exception";
        }
    }
}

void
RedisConnectionPool::postToWorker(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(workMutex_);
        workQueue_.push(std::move(task));
    }
    workCv_.notify_one();
}

bool
RedisConnectionPool::scheduleOnSlot(Slot &slot, std::function<void()> task) {
    if (!slot.eventBase) {
        return false;
    }
    auto *owned_task = new redis_event_task_t(std::move(task));
    timeval immediate = {0, 0};
    if (event_base_once(slot.eventBase, -1, EV_TIMEOUT, runEventTask, owned_task, &immediate) != 0) {
        delete owned_task;
        return false;
    }
    return true;
}

void
RedisConnectionPool::stopEventLoop() {
    for (auto &s : slots_) {
        if (!s->eventBase) {
            continue;
        }
        if (s->eventLoopThread.joinable()) {
            Slot *sp = s.get();
            bool scheduled = scheduleOnSlot(*s, [sp]() {
                for (int i = 0; i < Slot::kConns; ++i) {
                    SubConn &sc = sp->subconns[i];
                    if (sc.asyncCtx) {
                        sc.connected.store(false);
                        redisAsyncFree(sc.asyncCtx);
                        sc.asyncCtx = nullptr;
                    }
                }
                event_base_loopbreak(sp->eventBase);
            });
            if (!scheduled) {
                event_base_loopbreak(s->eventBase);
            }
            s->eventLoopThread.join();
        }
        event_base_free(s->eventBase);
        s->eventBase = nullptr;
    }
}

void
RedisConnectionPool::completeSubConnInit(SubConn &sc, bool success) {
    if (success) {
        sc.slot->initOkCount.fetch_add(1, std::memory_order_relaxed);
    }
    sc.connected.store(success);
    if (sc.slot->initCount.fetch_add(1, std::memory_order_acq_rel) + 1 == Slot::kConns) {
        bool allOk = (sc.slot->initOkCount.load() == Slot::kConns);
        sc.slot->initSucceeded.store(allOk);
        sc.slot->initDone.store(true, std::memory_order_release);
    }
}

void
RedisConnectionPool::freeSubConnAsyncCtx(SubConn &sc) {
    if (sc.asyncCtx) {
        redisAsyncFree(sc.asyncCtx);
        sc.asyncCtx = nullptr;
    }
}

void
RedisConnectionPool::startSubConnSelect(SubConn &sc) {
    if (config_.db == 0) {
        completeSubConnInit(sc, true);
        return;
    }
    int ret = redisAsyncCommand(sc.asyncCtx, selectCallback, &sc, "SELECT %d", config_.db);
    if (ret != REDIS_OK) {
        NIXL_ERROR << "Failed to queue Redis SELECT command";
        completeSubConnInit(sc, false);
        freeSubConnAsyncCtx(sc);
    }
}

void
RedisConnectionPool::startSubConnAuth(SubConn &sc) {
    if (config_.password.empty()) {
        startSubConnSelect(sc);
        return;
    }
    const int ret = config_.username.empty() ?
        redisAsyncCommand(sc.asyncCtx, authCallback, &sc, "AUTH %s", config_.password.c_str()) :
        redisAsyncCommand(sc.asyncCtx,
                          authCallback,
                          &sc,
                          "AUTH %s %s",
                          config_.username.c_str(),
                          config_.password.c_str());
    if (ret != REDIS_OK) {
        NIXL_ERROR << "Failed to queue Redis AUTH command";
        completeSubConnInit(sc, false);
        freeSubConnAsyncCtx(sc);
    }
}

void
RedisConnectionPool::connectCallback(const redisAsyncContext *c, int status) {
    auto *sc = static_cast<SubConn *>(c->data);
    if (status != REDIS_OK) {
        NIXL_ERROR << absl::StrFormat("Redis connection error: %s", c->errstr);
        // hiredis frees c after this callback returns (REDIS_CONNECTED was never set,
        // so disconnectCallback is NOT called). Null asyncCtx now so stopEventLoop
        // does not call redisAsyncFree on already-freed memory.
        sc->asyncCtx = nullptr;
        sc->slot->pool->completeSubConnInit(*sc, false);
    } else {
        sc->slot->pool->startSubConnAuth(*sc);
    }
}

void
RedisConnectionPool::disconnectCallback(const redisAsyncContext *c, int status) {
    auto *sc = static_cast<SubConn *>(c->data);
    if (status != REDIS_OK) {
        NIXL_WARN << absl::StrFormat("Redis disconnected with error: %s", c->errstr);
    }
    // hiredis frees c after this callback returns. Null asyncCtx now so stopEventLoop
    // does not call redisAsyncFree on already-freed memory (double-free).
    sc->asyncCtx = nullptr;
    sc->connected.store(false);
}

void
RedisConnectionPool::authCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *sc = static_cast<SubConn *>(privdata);
    auto *r = static_cast<redisReply *>(reply);
    if (!checkRedisReplyOk(r, "AUTH")) {
        sc->slot->pool->completeSubConnInit(*sc, false);
        if (r == nullptr) {
            // Connection dropped: hiredis fires disconnectCallback and frees c itself.
            // Null asyncCtx now so disconnectCallback and stopEventLoop don't double-free.
            sc->asyncCtx = nullptr;
        } else {
            // Server rejected AUTH while connection is still live; close it explicitly.
            sc->slot->pool->freeSubConnAsyncCtx(*sc);
        }
        return;
    }
    sc->slot->pool->startSubConnSelect(*sc);
}

void
RedisConnectionPool::selectCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *sc = static_cast<SubConn *>(privdata);
    auto *r = static_cast<redisReply *>(reply);
    if (!checkRedisReplyOk(r, "SELECT")) {
        sc->slot->pool->completeSubConnInit(*sc, false);
        if (r == nullptr) {
            sc->asyncCtx = nullptr;
        } else {
            sc->slot->pool->freeSubConnAsyncCtx(*sc);
        }
        return;
    }
    sc->slot->pool->completeSubConnInit(*sc, true);
}

void
RedisConnectionPool::setCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *ctx = static_cast<CallbackContext *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    bool success = false;
    if (!r) {
        NIXL_ERROR << "Redis SET: connection lost (no reply)";
    } else if (r->type == REDIS_REPLY_STATUS) {
        success = (strcmp(r->str, "OK") == 0);
    } else if (r->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis SET error: %s", r->str);
    }

    auto promise_ptr = ctx->promise_ptr;
    auto *inFlight = ctx->inFlight;
    delete ctx;
    if (inFlight) {
        inFlight->fetch_sub(1, std::memory_order_relaxed);
    }
    if (promise_ptr) {
        promise_ptr->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
    }
}

void
RedisConnectionPool::getCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *ctx = static_cast<CallbackContext *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    if (r && r->type == REDIS_REPLY_STRING) {
        const size_t reply_len = static_cast<size_t>(r->len);
        const bool size_ok = (reply_len == ctx->data_len);

        if (size_ok && ctx->data_len > 0 && ctx->data_ptr) {
            uintptr_t dst = ctx->data_ptr;
            size_t len = ctx->data_len;
            auto promise = ctx->promise_ptr;
            auto *inFlight = ctx->inFlight;
            delete ctx;

            // For small payloads the worker-dispatch overhead (mutex + futex per callback)
            // exceeds the memcpy cost, so copy inline on the event loop thread.
            // 512 KB at ~20 GB/s ≈ 25 µs — acceptable blocking time for one callback.
            static constexpr size_t kInlineThreshold = 512 * 1024;
            if (len <= kInlineThreshold) {
                std::memcpy(reinterpret_cast<void *>(dst), r->str, len);
                if (inFlight) {
                    inFlight->fetch_sub(1, std::memory_order_relaxed);
                }
                if (promise) {
                    promise->set_value(NIXL_SUCCESS);
                }
            } else {
                // Steal the reply buffer so the large memcpy runs on a worker thread
                // without blocking the event loop. hiredis skips hi_free() for null str;
                // the worker uses hi_free() to match the hiredis allocator.
                auto *sc = static_cast<SubConn *>(c->data);
                char *str = r->str;
                r->str = nullptr;
                sc->slot->pool->postToWorker([str, dst, len, promise, inFlight]() {
                    std::memcpy(reinterpret_cast<void *>(dst), str, len);
                    hi_free(str);
                    if (inFlight) {
                        inFlight->fetch_sub(1, std::memory_order_relaxed);
                    }
                    if (promise) {
                        promise->set_value(NIXL_SUCCESS);
                    }
                });
            }
            return;
        }

        const bool null_dst = (ctx->data_len > 0 && !ctx->data_ptr);
        if (!size_ok) {
            NIXL_ERROR << absl::StrFormat(
                "Redis GET size mismatch: expected %zu bytes, got %zu bytes",
                ctx->data_len,
                reply_len);
        } else if (null_dst) {
            NIXL_ERROR << "Redis GET: data_ptr is null with non-zero data_len";
        }
        // Resolve on event loop: size mismatch, null dst, or zero-length success (no copy needed).
        auto promise_ptr = ctx->promise_ptr;
        auto *inFlight = ctx->inFlight;
        delete ctx;
        if (inFlight) {
            inFlight->fetch_sub(1, std::memory_order_relaxed);
        }
        if (promise_ptr) {
            promise_ptr->set_value((size_ok && !null_dst) ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
        }
        return;
    }

    if (r && r->type == REDIS_REPLY_NIL) {
        NIXL_WARN << "Redis GET: key not found";
    } else if (r && r->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis GET error: %s", r->str);
    }

    auto promise_ptr = ctx->promise_ptr;
    auto *inFlight = ctx->inFlight;
    delete ctx;
    if (inFlight) {
        inFlight->fetch_sub(1, std::memory_order_relaxed);
    }
    if (promise_ptr) {
        promise_ptr->set_value(NIXL_ERR_BACKEND);
    }
}

RedisConnectionPool::Slot *
RedisConnectionPool::leastLoadedHealthySlot() {
    Slot *best = nullptr;
    for (auto &s : slots_) {
        bool any = false;
        for (int i = 0; i < Slot::kConns; ++i) {
            if (s->subconns[i].connected.load()) {
                any = true;
                break;
            }
        }
        if (!any) {
            continue;
        }
        if (!best ||
            s->inFlight.load(std::memory_order_relaxed) <
                best->inFlight.load(std::memory_order_relaxed)) {
            best = s.get();
        }
    }
    return best;
}

void
RedisConnectionPool::putKeyAsync(std::string_view key,
                                 uintptr_t data_ptr,
                                 size_t data_len,
                                 std::shared_ptr<std::promise<nixl_status_t>> promise) {
    auto *slot = leastLoadedHealthySlot();
    if (!slot) {
        if (promise) {
            promise->set_value(NIXL_ERR_BACKEND);
        }
        return;
    }

    slot->inFlight.fetch_add(1, std::memory_order_relaxed);

    std::string key_copy(key);
    const bool scheduled = scheduleOnSlot(
        *slot,
        [slot, key = std::move(key_copy), data_ptr, data_len, promise]() mutable {
            int next =
                slot->nextConn.fetch_add(1, std::memory_order_relaxed) % Slot::kConns;
            SubConn *sc = &slot->subconns[next];
            if (!sc->connected.load() || !sc->asyncCtx) {
                sc = &slot->subconns[(next + 1) % Slot::kConns];
            }
            if (!sc->connected.load() || !sc->asyncCtx) {
                slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
                if (promise) {
                    promise->set_value(NIXL_ERR_BACKEND);
                }
                return;
            }

            auto *ctx = new CallbackContext;
            ctx->data_ptr = 0;
            ctx->data_len = 0;
            ctx->promise_ptr = promise;
            ctx->inFlight = &slot->inFlight;

            const int ret = redisAsyncCommand(sc->asyncCtx,
                                              setCallback,
                                              ctx,
                                              "SET %b %b",
                                              key.data(),
                                              key.size(),
                                              reinterpret_cast<const char *>(data_ptr),
                                              data_len);
            if (ret != REDIS_OK) {
                slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
                auto p = ctx->promise_ptr;
                delete ctx;
                if (p) {
                    p->set_value(NIXL_ERR_BACKEND);
                }
            }
        });

    if (!scheduled) {
        slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
        if (promise) {
            promise->set_value(NIXL_ERR_BACKEND);
        }
    }
}

void
RedisConnectionPool::getKeyAsync(std::string_view key,
                                 uintptr_t data_ptr,
                                 size_t data_len,
                                 std::shared_ptr<std::promise<nixl_status_t>> promise) {
    auto *slot = leastLoadedHealthySlot();
    if (!slot) {
        if (promise) {
            promise->set_value(NIXL_ERR_BACKEND);
        }
        return;
    }

    slot->inFlight.fetch_add(1, std::memory_order_relaxed);

    std::string key_copy(key);
    const bool scheduled = scheduleOnSlot(
        *slot,
        [slot, key = std::move(key_copy), data_ptr, data_len, promise]() mutable {
            int next =
                slot->nextConn.fetch_add(1, std::memory_order_relaxed) % Slot::kConns;
            SubConn *sc = &slot->subconns[next];
            if (!sc->connected.load() || !sc->asyncCtx) {
                sc = &slot->subconns[(next + 1) % Slot::kConns];
            }
            if (!sc->connected.load() || !sc->asyncCtx) {
                slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
                if (promise) {
                    promise->set_value(NIXL_ERR_BACKEND);
                }
                return;
            }

            auto *ctx = new CallbackContext;
            ctx->data_ptr = data_ptr;
            ctx->data_len = data_len;
            ctx->promise_ptr = promise;
            ctx->inFlight = &slot->inFlight;

            const int ret =
                redisAsyncCommand(sc->asyncCtx, getCallback, ctx, "GET %b", key.data(), key.size());
            if (ret != REDIS_OK) {
                slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
                auto p = ctx->promise_ptr;
                delete ctx;
                if (p) {
                    p->set_value(NIXL_ERR_BACKEND);
                }
            }
        });

    if (!scheduled) {
        slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
        if (promise) {
            promise->set_value(NIXL_ERR_BACKEND);
        }
    }
}

std::optional<bool>
RedisConnectionPool::checkKeyExistsSync(std::string_view key) {
    std::lock_guard<std::mutex> lock(syncMutex_);
    if (!syncCtx_ || syncCtx_->err) {
        if (syncCtx_) {
            redisFree(syncCtx_);
            syncCtx_ = nullptr;
        }
        connectSyncContext();
    }
    if (!syncCtx_) {
        NIXL_ERROR << "Sync Redis connection unavailable for EXISTS";
        return std::nullopt;
    }

    redisReply *reply =
        static_cast<redisReply *>(redisCommand(syncCtx_, "EXISTS %b", key.data(), key.size()));

    if (!reply) {
        NIXL_ERROR << "Redis EXISTS: no reply";
        redisFree(syncCtx_);
        syncCtx_ = nullptr;
        return std::nullopt;
    }
    if (reply->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis EXISTS error: %s", reply->str);
        freeReplyObject(reply);
        return std::nullopt;
    }
    if (reply->type != REDIS_REPLY_INTEGER) {
        NIXL_ERROR << absl::StrFormat("Redis EXISTS unexpected reply type: %d", reply->type);
        freeReplyObject(reply);
        return std::nullopt;
    }

    const bool exists = (reply->integer == 1);
    freeReplyObject(reply);
    return exists;
}

#endif // HAVE_HIREDIS_ASYNC
