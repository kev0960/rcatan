# Engine

## Commands

All commands run from the repository root.

Running the tests.

```sh
cargo test -p rcatan
```

Running the profiler. Build the bench first; the `Executable` line it prints
is the binary to record. The hash in the name changes whenever the code or the
dependencies change, so build again after an edit.

```sh
cargo bench -p rcatan --bench engine --no-run
#   Executable benches/engine.rs (target/release/deps/engine-<hash>)

samply record target/release/deps/engine-<hash> --bench --profile-time 10 random_play/1000
```
