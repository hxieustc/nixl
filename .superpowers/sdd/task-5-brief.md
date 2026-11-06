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
