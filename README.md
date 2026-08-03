# Grok Build (Cursor subscription fork)

> **Temporary fork.** This repo exists only to add Cursor subscription support. It is not a long-term maintained fork of Grok Build, but it will sync with upstream.

Fork of [xai-org/grok-build](https://github.com/xai-org/grok-build) with Cursor subscription support.

Full Grok Build docs: [upstream README](https://github.com/xai-org/grok-build/blob/main/README.md).

## Setup

### 1. Install and sign in to the Cursor Agent CLI

```sh
curl https://cursor.com/install -fsS | bash
```

That puts `agent` in `~/.local/bin`. Check it:

```sh
agent --version
```

If you get “command not found”, tell your shell to look in that folder, then try again:

```sh
# bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Sign in:

```sh
agent login
```

### 2. Install build tools

Install Rust (if you do not have it):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Restart your terminal (or run the command the installer prints), then install DotSlash (needed to fetch `protoc` during the build):

```sh
cargo install dotslash
```

### 3. Build and install `grok`

```sh
git clone https://github.com/matfk/grok-build.git
cd grok-build
cargo build -p xai-grok-pager-bin --release
mkdir -p ~/.local/bin
cp target/release/xai-grok-pager ~/.local/bin/grok
```

Check it:

```sh
grok --version
```

If that fails with “command not found”, use the same PATH steps from step 1.

After you rebuild later, run the `cp` line again so `~/.local/bin/grok` is updated.

### 4. Try it out

```sh
grok
```

In the TUI, run `/model` and pick a Cursor model (or a DeepSeek / OpenRouter model if you set `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY`).

## Staying up to date

This fork may show a tip when a newer Grok Build release exists:

`Update: upstream vX available — git pull & rebuild (ctrl+u dismisses)`

That tip is **notice-only**. Do **not** run the stock updater (`grok update`). It installs the official binary and would drop Cursor support. `ctrl+u` on the tip only dismisses it.

Upstream is already merged into this fork when available. To update:

```sh
cd /path/to/grok-build
git pull
cargo build -p xai-grok-pager-bin --release
cp target/release/xai-grok-pager ~/.local/bin/grok
```
