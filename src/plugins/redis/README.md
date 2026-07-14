<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Redis Plugin

The `REDIS` plugin exposes Redis as a local NIXL storage backend. It supports

- asynchronous `NIXL_WRITE`/`NIXL_READ` transfers and
- synchronous `queryMem()` checks.

## Architecture

The plugin follows the direct-backend layout similar to existing backends:

```text
src/plugins/redis/
  redis_backend.h/.cpp  nixlRedisKVEngine and NIXL transfer logic
  redis_client.h/.cpp   hiredis client and its unit-test interface
  redis_plugin.cpp      NIXL plugin registration
  meson.build           REDIS plugin target
  README.md
```

`nixlRedisKVEngine` derives directly from `nixlBackendEngine`. The internal `iRedisClient`
interface isolates hiredis/libevent and permits Redis-free unit tests; it is not a generic KV
extension API.

The production client uses a `RedisConnectionPool` (default: pool_size=8).
Each *slot* owns two pipelined hiredis/libevent async connections sharing one dedicated
libevent event loop thread — so a pool of size N creates 2N async TCP connections and N
event loop threads. Commands are distributed round-robin across the two connections within
the chosen slot to halve per-connection queue depth. A single shared blocking hiredis
connection handles `EXISTS` for all slots; `queryMem` calls EXISTS serially, so one
connection is sufficient.

Resource cost: **(2N + 1) OS threads** + 2N async TCP connections + 1 sync TCP connection
for a pool of size N. Increasing pool size helps when the Redis server runs with
`--io-threads` and multiple connections saturate more server threads. Decrease pool size to
1 for memory-constrained environments.

Operations are dispatched to the healthy slot with the fewest in-flight async commands
(least-connection routing). Commands fail immediately if all slots are disconnected.

GET reply buffers for payloads > 512 KB are handed off to a worker pool (N threads) so
that large memcpy calls do not block the event loop threads. For payloads ≤ 512 KB the
copy is performed inline on the event loop thread because the worker dispatch overhead
(mutex + condition variable) exceeds the copy cost.

## Performance

### Benchmark setup

Measurements were taken on a dual-socket Intel Xeon Gold 6438Y+ (32 cores / 2 threads per
core, 4 GHz) with both the Redis server (Docker, host networking, 16 io-threads) and the
NIXL benchmark process pinned to the same NUMA node. All numbers are loopback TCP;
no NIC is involved. Total buffer: 64 MiB. Warmup: 32 iterations; measured: 208 iterations.

Notation: **bs** = batch size (descriptors per `postXfer` call), **T** = thread count,
**pool** = `pool_size`.

### Optimization history

Three successive optimizations were applied to the original single-loop, single-connection
implementation. Each is described below with its measured impact.

---

#### Fix 1 — Inline memcpy for GET replies ≤ 512 KB

**What changed.** Previously every GET reply was handed to the N-thread worker pool for
the `memcpy` regardless of payload size. The mutex + condition-variable round-trip cost
(~7 µs each way) dominated small transfers. Fix 1 copies payloads ≤ 512 KB directly on
the event loop thread and only dispatches to the worker pool for larger blocks.

**Threshold rationale.** 512 KB at ~20 GB/s DRAM bandwidth costs ≈ 25 µs — acceptable
blocking time for one callback cycle. Larger blocks would stall the event loop long enough
to delay other pending callbacks.

**Measured impact at 128 KB READ, bs=16, 4T, pool=8.**

| Before Fix 1 | After Fix 1 |
|:---:|:---:|
| 1.35 GB/s | **2.69 GB/s** (+99%) |

Fix 1 also eliminates the per-callback mutex overhead for all block sizes ≤ 512 KB on
the READ path. The WRITE path is unaffected (SET callbacks copy no data).

---

#### Option A — One dedicated event loop thread per slot

**What changed.** The original pool used a single `event_base` shared across all N async
connections, so every `getCallback` and `setCallback` serialized on one thread. Option A
gives each slot its own `event_base` and a dedicated OS thread (`event_base_dispatch`).
Callbacks for different slots now run truly in parallel.

With N = 8 and bs = 16, 4T issues 64 concurrent GETs spread evenly across 8 slots. Each
event loop thread handles 8 callbacks independently instead of all 64 serializing.

**Measured impact at 128 KB READ, bs=16, 4T, pool=8.**

| Fix 1 baseline | + Option A |
|:---:|:---:|
| 2.69 GB/s | **3.48 GB/s** (+29%) |

Throughput at bs=8 and bs=16 improves most because those configurations create the most
contention on the shared event loop.

---

#### Option B — Two async connections per slot

**What changed.** Each slot now owns two pipelined async TCP connections instead of one
(`kConns = 2`). Both connections share the slot's event loop thread. Dispatch round-robins
between them, halving per-connection queue depth from `(T × bs) / N` to `(T × bs) / 2N`.
This matches the connection count of the original "2N" design.

**Measured impact at 128 KB READ, bs=16, 4T, pool=8.**

| Fix 1 baseline | + Option A | + Option A + B |
|:---:|:---:|:---:|
| 2.69 GB/s | 3.48 GB/s | **3.71 GB/s** (+38% vs baseline) |

Option B contributes a modest additional +7% on top of Option A because the dominant
bottleneck at 128 KB has shifted from connection queue depth to per-callback inline memcpy
serialization within each event loop thread.

---

### Representative results (Option A + B)

All numbers from the final implementation with default `pool_size=8`.

#### WRITE throughput (GB/s)

| Block (KB) | bs=1, 4T | bs=4, 4T | bs=8, 4T | bs=16, 4T | bs=8, 8T | bs=16, 8T |
|:----------:|:--------:|:--------:|:--------:|:---------:|:--------:|:---------:|
| 128 | 1.50 | 2.15 | 3.02 | 3.62 | 3.68 | 4.08 |
| 256 | 1.70 | 3.35 | 5.44 | — | 7.12 | — |
| 512 | 1.94 | 5.72 | 5.93 | 7.02 | 7.43 | 4.77 |
| 1024 | 2.72 | 4.23 | 4.27 | 5.93 | 5.34 | 4.61 |

WRITE throughput exceeds 7 GB/s for 256 KB and 512 KB at bs ≥ 8 with 8 threads.
At 512 KB bs=16 the pool is back-pressured by the Redis server buffer limit, causing
a throughput dip; reducing bs or pool_size recovers bandwidth.

#### READ throughput (GB/s)

| Block (KB) | bs=1, 4T | bs=4, 4T | bs=8, 4T | bs=16, 4T | bs=8, 8T | bs=16, 8T |
|:----------:|:--------:|:--------:|:--------:|:---------:|:--------:|:---------:|
| 128 | 2.25 | 1.92 | 2.48 | 3.71 | 2.46 | 3.35 |
| 256 | 2.33 | 2.73 | 2.92 | 2.74 | 2.67 | 2.55 |
| 512 | 2.94 | 3.12 | 2.78 | 2.59 | 2.49 | 2.43 |
| 1024 | 2.67 | 2.47 | 3.16 | 2.41 | 2.75 | 2.65 |

READ throughput is lower than WRITE because `getCallback` performs an inline memcpy for
every reply ≤ 512 KB, serializing within each event loop thread. At 128 KB the memcpy
cost per callback (≈ 6–7 µs) is small; higher batch sizes amortize scheduling overhead
and reach 3.7 GB/s. At 256 KB and above the memcpy cost grows proportionally and limits
per-slot throughput to ≈ 2.5–3 GB/s regardless of pool configuration.

---

## Dependencies

- `hiredis` with async API support
- `libevent`
- `libevent_pthreads`

If REDIS is explicitly enabled or selected as a static plugin, missing hiredis/libevent is a
configuration error. During a default all-plugin build, missing dependencies cause REDIS to be
skipped with a warning.

## Configuration

`RedisConfig` is the Redis client's resolved, internal configuration value. The backend calls
`RedisConfig::fromBackendParams()` once during construction and passes the result to
`RedisConnectionPool`; the client does not repeatedly read backend parameters or environment
variables.

```cpp
struct RedisConfig {
    std::string host = "localhost";
    int port = 6379;
    std::string username;
    std::string password;
    int db = 0;
    int pool_size = 8;
};
```

Each setting is resolved in this precedence order:

1. A valid value in the NIXL backend parameter map.
2. The corresponding `REDIS_*` environment variable, when one exists.
3. The built-in default.

An explicitly provided string value, including an empty username or password, takes precedence
over its environment fallback. Port values that fail validation fall through to `REDIS_PORT` and
then to `6379`; database parse failures fall back to `REDIS_DB` and then to `0`; trailing
garbage (e.g. `"2x"`) and negative values are rejected at each stage. Invalid `pool_size`
values fall back to `8`.

| Parameter | Environment fallback | Default | Description |
|-----------|----------------------|---------|-------------|
| `host` | `REDIS_HOST` | `localhost` | Redis hostname or IP address |
| `port` | `REDIS_PORT` | `6379` | Redis TCP port |
| `username` | `REDIS_USERNAME` | empty | Redis ACL username |
| `password` | `REDIS_PASSWORD` | empty | Redis AUTH password |
| `db` | `REDIS_DB` | `0` | Redis logical database (must be ≥ 0) |
| `pool_size` | `REDIS_POOL_SIZE` | `8` | Number of concurrent Redis connections |

Authentication behavior is determined by the resolved credentials:

- When `password` is empty, the client does not send `AUTH`. This is correct for a Redis server
  that allows unauthenticated access. A password-protected server will reject subsequent commands.
- When only `password` is set, the client sends legacy/default-user `AUTH password`.
- When both `username` and `password` are set, the client sends ACL-style
  `AUTH username password`. A username without a password is rejected during backend creation.

```cpp
nixl_b_params_t params = {
    {"host", "127.0.0.1"},
    {"port", "6379"},
    {"username", "nixl"},
    {"password", "example-password"},
    {"db", "2"},
    {"pool_size", "8"},
};
agent.createBackend("REDIS", params);
```

## Transfer behavior

| NIXL operation | Redis operation | Result |
|----------------|-----------------|--------|
| `NIXL_WRITE` | `SET key bytes` | Stores local DRAM bytes |
| `NIXL_READ` | `GET key` | Copies the exact Redis value into local DRAM |
| `queryMem` | `EXISTS key` | Reports whether the key exists |

Local descriptors must be `DRAM_SEG`; remote descriptors may be `DRAM_SEG` or `OBJ_SEG`. A Redis
key comes from descriptor `metaInfo` when present, otherwise from the decimal `addr`. OBJ_SEG
descriptors must always supply a non-empty `metaInfo`. `postXfer()`
resolves every remote key before dispatching commands, so an invalid descriptor cannot produce a
partially submitted transfer.

`postXfer()` returns `NIXL_IN_PROG`; `checkXfer()` polls the request futures and returns success or
the first backend error once all completed work has been observed.

## Build and test

Build the plugin:

```bash
meson setup build -Denable_plugins=REDIS
meson compile -C build REDIS
```

NIXL does not add tests to a `release` build, even when `build_tests` is enabled. To build and run
the Redis unit tests, install GoogleTest and GoogleMock and configure a separate non-release build:

```bash
# Debian/Ubuntu
sudo apt-get install libgtest-dev libgmock-dev

meson setup build-redis-tests \
  --buildtype=debug \
  -Dbuild_tests=true \
  -Denable_plugins=REDIS
meson compile -C build-redis-tests unit
meson devenv -C build-redis-tests \
  ./test/gtest/unit/unit \
  '--gtest_filter=redisConfigTest.*:redisEngineTest.*'
```

The filter runs only the Redis configuration and backend suites. A successful run currently
reports 21 tests from 2 test suites. These tests inject `mockRedisClient` and do not require a
running Redis server; use the live smoke test below to exercise hiredis and a real server.

The dynamic plugin is produced at:

```text
build/src/plugins/redis/libplugin_REDIS.so
```

For a static build:

```bash
meson setup build -Denable_plugins=REDIS -Dstatic_plugins=REDIS
meson compile -C build
```

## nixlbench benchmark test

The unit tests use an injected Redis client and do not connect to a server. Use this smoke test to
exercise plugin creation, hiredis/libevent, and an actual Redis write/read data path. The example
uses a static REDIS plugin so the standalone nixlbench binary does not depend on dynamic plugin
discovery.

### Redis server

Start Redis with host networking so the container shares the host network stack
directly, eliminating the Docker bridge veth overhead. 

To make Redis server as performant as possible, 

1. Pass `--io-threads 4` to enable Redis 6+ threaded socket I/O, which parallelises 
reads and writes across connections.
2. use host mode (`--network=host`) to remove docker veth limit

```bash
docker run --detach --rm --name nixl-redis-smoke \
  --network=host redis:7-alpine \
  --io-threads 4 --io-threads-do-reads yes
docker exec nixl-redis-smoke redis-cli PING
```

The readiness command must print `PONG`. With `--network=host` the container
binds directly to the host's `127.0.0.1:6379`; no `-p` port mapping is needed.

### benchmark test

Install a static REDIS-enabled NIXL build into a temporary prefix, then build nixlbench against
that installation:

```bash
export NIXL_PREFIX=/tmp/nixl-redis-smoke-install

meson setup build-redis \
  --prefix="$NIXL_PREFIX" \
  -Denable_plugins=REDIS \
  -Dstatic_plugins=REDIS
meson compile -C build-redis
meson install -C build-redis

meson setup build-nixlbench benchmark/nixlbench \
  -Dnixl_path="$NIXL_PREFIX"
meson compile -C build-nixlbench
build-nixlbench/nixlbench --help | grep REDIS
```

The NIXL build requires the plugin dependencies listed above. The standalone nixlbench build also
requires the hiredis development package `libhiredis-dev` and `redis-tools` because it seeds Redis
before a READ and independently checks transferred data when consistency checking is enabled.

Run fixed-size 4 KiB WRITE and READ benchmarks. These commands use one thread, one descriptor per
batch, one in-flight request, 32 warm-up iterations, and 208 measured iterations:

```bash
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_POOL_SIZE=8
export LD_LIBRARY_PATH="$NIXL_PREFIX/lib/$(uname -m)-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build-nixlbench/nixlbench \
  --backend REDIS \
  --op_type WRITE \
  --start_block_size 131072 \
  --max_block_size 131072   \
  --start_batch_size=1 \
  --max_batch_size=64 \
  --num_iter 1000 \
  --warmup_iter 10 \
  --num_threads 4
  

build-nixlbench/nixlbench \
  --backend REDIS \
  --op_type READ \
  --start_block_size 131072 \
  --max_block_size 131072   \
  --start_batch_size=1 \
  --max_batch_size=64 \
  --num_iter 1000 \
  --warmup_iter 10 \
  --num_threads 4
```

Each command must exit successfully and print a result row for block size `4096`. The READ run
must also print `Seeded Redis key for READ`; with `--check_consistency=true`, nixlbench reports an
error and exits unsuccessfully if the transferred data differs. The reported bandwidth and latency
are loopback smoke-test measurements, not production performance results.

The example uses Redis database 0 without authentication. `REDIS_PASSWORD` may be set for a
password-protected default user. nixlbench's direct seed/consistency helper does not currently
support `REDIS_USERNAME` or selecting a nonzero `db`, even though the plugin itself supports both.

Inspect the created keys if desired, then stop the server:

```bash
docker exec nixl-redis-smoke redis-cli DBSIZE
docker stop nixl-redis-smoke
```
