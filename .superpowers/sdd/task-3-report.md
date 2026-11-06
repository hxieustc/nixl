# Task 3 Report: Port VALKEY into `respkv/valkey/`

**Status:** DONE_WITH_CONCERNS  
**Working tree:** `redis5/nixl`  
**Commit:** none (explicitly skipped)

## Summary

VALKEY is a separate plugin at `src/plugins/respkv/valkey/`. `nixlValkeyEngine` still subclasses `nixlBackendEngine`. Plugin name is `"VALKEY"`. GLIDE stays on the VALKEY target only (`valkey_glide_dep` plus `-Wl,--exclude-libs,libglide_ffi.a` on the shared library and on `valkey_backend_interface`). It is not in `respkv_engine_dep`. The Valkey client is not merged with the REDIS client. `respkv_engine_dep` is not added to VALKEY yet (Task 4).

## Sources

Copied from `nixlvalkey/nixl/src/plugins/valkey/` (C++ and headers hash-identical):

- `valkey_backend.h`, `valkey_backend.cpp`
- `valkey_client.h`, `valkey_client.cpp`
- `valkey_plugin.cpp`
- `meson.build`, `README.md`

`class nixlValkeyEngine : public nixlBackendEngine` and the constructor `: nixlBackendEngine(init_params)` are unchanged. `VALKEY_PLUGIN_NAME` is `"VALKEY"`.

Gtest copied from `nixlvalkey/nixl/test/gtest/unit/valkey/` (`valkey.cpp` hash-identical). Include dir is now `../../../../src/plugins/respkv/valkey`.

Two path strings were updated for the new tree:

- meson skip warning points at `src/plugins/respkv/valkey/README.md`
- README compile comment points at `build/src/plugins/respkv/valkey/libplugin_VALKEY.so`

## Wiring

- Root `meson.build`: `all_plugins` is `..., 'REDIS', 'VALKEY', 'GDS', ...` (nixlvalkey order).
- `meson_options.txt`: `glide_path` string option, default empty, placed before `infinia_path`.
- `src/plugins/respkv/meson.build`: `subdir('valkey')` when `enabled_plugins` has `VALKEY`. The old commented `subdir('valkey')` is gone. `respkv_engine` deps are unchanged (no GLIDE).
- `src/plugins/respkv/valkey/meson.build`: same dependency gate as the reference. Missing `libglide_ffi.a`, `glide_ffi.h`, `connection_request.proto`, `protoc`, or libprotobuf warns and `subdir_done()` on a default all-plugin build, and errors when VALKEY is explicitly enabled or static. `valkey_link_args` is on the shared library and on `valkey_backend_interface`.
- `src/core/nixl_plugin_manager.cpp`: `STATIC_PLUGIN_VALKEY` registers `Backend, VALKEY` immediately after REDIS.
- `src/core/meson.build`: static VALKEY adds `valkey_backend_interface` (carries the exclude-libs link args) after the REDIS block. Plugins are configured before core, so the variable exists when the static plugin is actually built.
- `test/gtest/unit/meson.build`: `subdir('valkey')` and `valkey_unit_test_dep` only when `valkey_backend_interface` is defined.
- `test/nixl/test_plugin.cpp`: `"VALKEY"` is next to `"REDIS"` in the known-plugins set.

## Self-review

- [x] Plugin name `"VALKEY"`
- [x] Still inherits `nixlBackendEngine` (Task 4 switches the base)
- [x] GLIDE is not linked into the respkv base
- [x] Not merged with the REDIS client
- [x] `glide_path` option present; skip-vs-error gate preserved
- [x] `--exclude-libs,libglide_ffi.a` stays on the VALKEY target and the interface
- [x] `subdir('valkey')` is enabled from `respkv/meson.build` only (not a second `src/plugins/valkey`)
- [x] Gtest include path is `respkv/valkey`
- [x] No `respkv_engine_dep` on VALKEY
- [x] No git commit

## Tests / configure

Not run. This Windows host has no `meson`, `ninja`, `g++`, `clang++`, `cl`, or `protoc` on PATH. `py -3` has no `mesonbuild` module. `wsl` has no installed distribution. `libglide_ffi.a` is not present under `nixlvalkey/valkey-glide/ffi/target/`.

Expected on Linux:

```bash
meson setup build-ut --buildtype=debug
# empty glide_path: warning, VALKEY skipped, valkey_backend_interface unset
# ./build-ut/test/gtest/unit/unit --gtest_filter='valkey*' then matches nothing

meson setup build-ut --buildtype=debug -Dglide_path=<prefix> --reconfigure
# prefix from valkey-glide ffi built with --features rdma
# VALKEY builds; valkey* config and init-failure tests should pass without a live RDMA server

meson setup build-valkey --buildtype=debug -Denable_plugins=VALKEY
# must error: VALKEY requested and glide deps missing
```
