# Contributing

## Setup and Verification

```bash
cargo fmt --check
cargo clippy --workspace
cargo test --workspace
cargo build --workspace --release
python examples/test_all_features.py
python examples/test_enterprise_features.py
```

## Guidelines

- Keep changes minimal and focused on a single topic.
- Unsafe blocks must have comments explaining their safety invariants.
- If a change modifies kernel generation or runtime dispatch, include throughput and latency numbers in the PR description.

## Licensing

Tessera is dual-licensed under GNU AGPLv3 and Commercial Proprietary terms. Contributions are accepted under the same dual terms, with commercial licensing rights retained by the copyright holder (Rahad Bhuiya / Exploidus).
