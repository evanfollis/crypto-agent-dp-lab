# Archive architecture and lineage

## Classification

- shape: `library`
- lifecycle: `archived`
- agentic risk: `none`
- successor: [Atlas](https://github.com/evanfollis/atlas)

The preserved package is a JAX/Equinox research prototype with Φ-layer,
portfolio, graph, RL, data, and gradient-monitoring modules. It is not a
running agent or service. The original Phase 2 comparative study has no
committed outcomes, so the central performance thesis remains unvalidated.

## Evidence boundary

Source code, experiment plans, and implementation checks demonstrate that a
prototype was built. They do not demonstrate statistical or economic
performance. The original README and research charter are historical artifacts
under `docs/` and must not be interpreted as current portfolio claims.

## Successor boundary

Atlas did not absorb this code. Atlas uses a smaller CPU-oriented stack and a
falsification workflow with pre-registration, walk-forward OOS evidence, and a
promotion gate. Only the historical relationship and general tooling lessons
carry forward.

## Operations

There is no deployment or mutable runtime state. `make check` verifies the
archive contract without attempting to recreate the heavy historical
environment. GitHub is the preservation surface.
