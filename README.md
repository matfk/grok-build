<div align="center">

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://media.x.ai/v1/website/spacexai-symbol-white-transparent-0c31957f.png">
    <source media="(prefers-color-scheme: light)" srcset="https://media.x.ai/v1/website/spacexai-symbol-black-transparent-6435cf42.png">
    <img alt="SpaceXAI logo" src="https://media.x.ai/v1/website/spacexai-symbol-black-transparent-6435cf42.png" width="96">
  </picture>
  <br>
  Grok Build (<code>grok</code>)
</h1>

</div>

> **This fork** adds **Cursor subscription** support to Grok Build.
> Route inference through the Cursor Agent CLI (`agent`) while Grok still
> owns tools (edit, shell, search). Models show up as `cursor/<id>`
> (for example `cursor/auto`, `cursor/composer-2.5`). Billing follows your
> Cursor plan, not xAI.

**Upstream Grok Build** is SpaceXAI's terminal AI coding agent. It runs as a
full-screen TUI that understands your codebase, edits files, executes shell
commands, searches the web, and manages long-running tasks — interactively,
headlessly for scripting/CI, or embedded in editors via the Agent Client
Protocol (ACP).

[Using Cursor subscription](#using-cursor-subscription) ·
[Install this fork](#install-this-fork) ·
[Upstream binary](#upstream-released-binary) ·
[Documentation](#documentation) ·
[Repository layout](#repository-layout) ·
[Development](#development) ·
[License](#license)

![Grok Build TUI](https://media.x.ai/v1/website/universe-tui-screenshot-6f7a0837.png)

**Learn more about upstream Grok Build at [x.ai/cli](https://x.ai/cli)**

This tree holds the Rust source for the `grok` CLI/TUI and agent runtime,
synced from the SpaceXAI monorepo, plus local changes for the Cursor bridge.
A small `SOURCE_REV` file at the root records the monorepo commit SHA for the
synced baseline.

---

## Using Cursor subscription

Grok spawns `agent --print --mode ask` for each model turn. Cursor stays in
ask mode so it does not edit your tree. Grok still runs tools from fenced
`grok-tool-calls` blocks in the model reply.

### 1. Install the Cursor Agent CLI

```sh
curl https://cursor.com/install -fsS | bash
agent --version
agent --list-models
```

Put `agent` on your `PATH`, or set `CURSOR_AGENT_PATH` / `AGENT_PATH`.

### 2. Sign in with Cursor

Use one of:

```sh
agent login
# or: export CURSOR_API_KEY=...
```

Inside the TUI you can also run `/login cursor`. Bare `/login` on an active
`cursor/…` model also starts Cursor login.

You do **not** need `grok login` or `XAI_API_KEY` for Cursor-billed models.

### 3. Pick a Cursor model

When `agent` is available, Grok auto-discovers models as `cursor/<id>`.
Force the feature on or off:

```toml
# ~/.grok/config.toml
[features]
cursor_cli = true   # or false to disable
```

```sh
export GROK_CURSOR_CLI=1   # or 0
```

Precedence: `GROK_CURSOR_CLI` → `[features].cursor_cli` → auto-detect `agent`.

Select a model in the picker (`Ctrl+M`), with `/model`, or on the CLI:

```sh
grok -m cursor/composer-2.5 -p "Explain this repo"
```

```toml
[models]
default = "cursor/auto"
```

While a Cursor model is active, Grok hides SuperGrok access-gate upsells for
that route. Installing `agent` alone does not hide upsells on xAI free-tier
models. Details and limits:
[`11-custom-models.md`](crates/codegen/xai-grok-pager/docs/user-guide/11-custom-models.md#cursor-cli-models-cursor-subscription).

---

## Install this fork

The official install script from [x.ai/cli](https://x.ai/cli) ships **upstream**
Grok Build. It does **not** include this fork's Cursor bridge. Build from this
repository instead.

### Requirements

- **Rust** — pinned by [`rust-toolchain.toml`](rust-toolchain.toml);
  `rustup` installs it on first build.
- **[DotSlash](https://dotslash-cli.com)** — hermetic tools under [`bin/`](bin/)
  (notably [`bin/protoc`](bin/protoc)) need `dotslash` on your `PATH` before
  you build:

  ```sh
  cargo install dotslash
  # or: https://dotslash-cli.com/docs/installation/
  /usr/bin/env dotslash --help
  ```

- **protoc** — via DotSlash `bin/protoc`, or `protoc` / `$PROTOC` on `PATH`.
- **Cursor Agent CLI** — required for Cursor models (see above).
- macOS and Linux are supported build hosts. Windows builds are best-effort.

### Build and run

```sh
# From a clone of this fork:
cargo run -p xai-grok-pager-bin              # build + launch the TUI
cargo build -p xai-grok-pager-bin --release  # target/release/xai-grok-pager
cargo check -p xai-grok-pager-bin            # fast validation
```

The binary artifact is named `xai-grok-pager`. Upstream packages rename it to
`grok`. You can symlink or alias it yourself:

```sh
ln -sf "$(pwd)/target/release/xai-grok-pager" ~/.local/bin/grok
```

On first launch with an xAI model, the TUI may open a browser for grok.com
auth. For Cursor-only use, pick a `cursor/…` model and run `/login cursor`
(or `agent login`). See the
[authentication guide](crates/codegen/xai-grok-pager/docs/user-guide/02-authentication.md).

---

## Upstream released binary

Prebuilt **upstream** binaries (no Cursor fork changes) for macOS, Linux, and
Windows:

```sh
curl -fsSL https://x.ai/cli/install.sh | bash   # macOS / Linux / Git Bash
irm https://x.ai/cli/install.ps1 | iex          # Windows PowerShell
grok --version
```

See the [changelog](https://x.ai/build/changelog) for upstream releases.

---

## Documentation

Full online documentation (upstream):
[docs.x.ai/build/overview](https://docs.x.ai/build/overview).

The user guide ships with the pager crate:
[`crates/codegen/xai-grok-pager/docs/user-guide/`](crates/codegen/xai-grok-pager/docs/user-guide/)
— getting started, keyboard shortcuts, slash commands, configuration, theming,
MCP servers, skills, plugins, hooks, headless mode, sandboxing, and more.

Cursor-specific pages in this fork:

- [Authentication — Cursor subscription](crates/codegen/xai-grok-pager/docs/user-guide/02-authentication.md#cursor-subscription-cursor-cli)
- [Custom models — Cursor CLI](crates/codegen/xai-grok-pager/docs/user-guide/11-custom-models.md#cursor-cli-models-cursor-subscription)

---

## Repository layout

| Path | Contents |
|------|----------|
| `crates/codegen/xai-grok-pager-bin` | Composition-root package; builds the `xai-grok-pager` binary |
| `crates/codegen/xai-grok-pager` | The TUI: scrollback, prompt, modals, rendering |
| `crates/codegen/xai-grok-shell` | Agent runtime + leader/stdio/headless entry points |
| `crates/codegen/xai-grok-sampler` | Model backends (includes `cursor_cli`) |
| `crates/codegen/xai-grok-tools` | Tool implementations (terminal, file edit, search, ...) |
| `crates/codegen/xai-grok-workspace` | Host filesystem, VCS, execution, checkpoints |
| `crates/codegen/...` | The rest of the CLI crate closure (config, MCP, markdown, sandbox, ...) |
| `crates/common/`, `crates/build/`, `prod/mc/` | Small shared leaf crates pulled in by the closure |
| `third_party/` | Vendored upstream source (Mermaid diagram stack) — see below |

> [!IMPORTANT]
> The root `Cargo.toml` (workspace members, dependency versions, lints,
> profiles) is **generated** — treat it as read-only. Prefer editing per-crate
> `Cargo.toml` files.

---

## Development

```sh
cargo check -p <crate>        # always target specific crates; full-workspace builds are slow
cargo test -p xai-grok-config # per-crate tests
cargo clippy -p <crate>       # lint config: clippy.toml at the repo root
cargo fmt --all               # rustfmt.toml at the repo root
```

---

## Contributing

> [!NOTE]
> External contributions are not accepted upstream. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

First-party code in this repository is licensed under the **Apache License,
Version 2.0** — see [`LICENSE`](LICENSE).

Third-party and vendored code remains under its original licenses. See:

- [`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES) — crates.io / git dependencies,
  bundled UI themes, and **in-tree source ports** (including openai/codex and
  sst/opencode tool implementations)
- [`crates/codegen/xai-grok-tools/THIRD_PARTY_NOTICES.md`](crates/codegen/xai-grok-tools/THIRD_PARTY_NOTICES.md)
  — crate-local notice for the codex and opencode ports (license texts +
  Apache §4(b) change notice)
- [`third_party/NOTICE`](third_party/NOTICE) — vendored Mermaid-stack index
