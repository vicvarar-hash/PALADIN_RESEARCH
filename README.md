# PALADIN — SACMAT 2027 reviewer artifact

This snapshot contains the policy evaluation implementation, policies, cached model outputs, ASTRA v0.3 task data, MCP tool schemas, and deterministic replay scripts. It excludes manuscripts, internal notes, exploratory reports, cloud deployment settings, and Git history.

## Reproduce

Use Python 3.12 on Linux. From this directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-replay.txt
python reproduce/run.py
```

Alternatively: `docker build -t paladin-review .` then `docker run --rm paladin-review`.

Replay requires no API keys, model calls, active MCP servers, or paid services. Installation requires package downloads. Results are written under `outputs/`. The runner regenerates the canonical validation and selection decision rows, verifies exact output bytes against `reproduce/expected_sha256.json`, and prints validation ablations, corroborated coverage, selection, and paired retrieval comparisons. After replay, tables can be recomputed using only the Python standard library with `python reproduce/canonical_tables.py`.

## Scope and interpretation

- `app/`: loaders, models, and policy/replay services; upstream runtime code is preserved.
- `policies/`: active configurations, including legacy compatibility files.
- `datasets/`: task data, fixed split files, and cached inference outputs.
- `mcp_servers/`: schema snapshots; these are data, never executed by replay.
- `reproduce/expected_sha256.json`: exact output hashes from the tested replay.
- `reproduce/`: replay and table generation entrypoints, adapted only for output paths.

Security-view metrics treat illegitimate requests as positive. SecFail is the fraction of illegitimate requests admitted; admission is the fraction of legitimate requests admitted. Cached outputs reproduce policy decisions, not fresh model behavior. Some historical logs omit issue codes: replay supplies an empty list, as documented in `app/services/replay_service.py`. Identity/transport deployment and OPA parity are outside this offline replay; Docker is a replay runner, not a live enforcement deployment. Additional analyses in the manuscript are not all covered by these canonical scripts.

Third-party schema and dataset provenance remains in the data where present. No new license is asserted for upstream materials. Model names are retained as recorded in cached logs, without independently validating provider provenance.

## Reference regeneration

The expected output hashes were generated with the supplied implementation. Earlier stored rows were stale after normalized tool-name aliases were added to BM25 lookup; only the earlier GPT-4o validation references had been refreshed upstream. The artifact applies the implementation consistently to every replay. This changes some historical row decisions and aggregate metrics; it does not modify the original study repository's results or manuscript claims. `python reproduce/assemble_crossvendor.py` adds the six-model cross-vendor panel after canonical replay. Treat any remaining manuscript-to-artifact numeric differences as limitations requiring reconciliation, rather than exact reproduction of every manuscript claim.
