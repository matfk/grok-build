//! Cursor Agent CLI (`agent`) sampling backend.
//!
//! Spawns `agent --print --mode ask` (prompt on stdin) so inference uses the caller's Cursor
//! subscription while Grok retains client-side tool execution.
//! Tools are described in the prompt; the model returns a `grok-tool-calls`
//! fence when it needs Grok to run a tool. Cursor Agent/Plan CLI modes are
//! never used — Shift+Tab session modes stay Grok-local.

mod process;
mod prompt;
mod stream;

pub use process::{
    CURSOR_CLI_API_KEY, CURSOR_CLI_BASE_URL, CursorAccountInfo, CursorCliError, CursorCliMode,
    DiscoveredCursorModel, agent_available, agent_bin, fetch_about, list_models, run_agent_login,
};
pub use prompt::{
    TOOL_CALLS_FENCE, build_prompt, looks_like_ask_mode_refusal, looks_like_tool_intent,
    parse_assistant_output,
};
pub use stream::{CursorCliStreamOpts, stream_cursor_cli};
