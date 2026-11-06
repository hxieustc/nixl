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
