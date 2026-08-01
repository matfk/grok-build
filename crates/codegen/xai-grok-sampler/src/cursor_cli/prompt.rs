//! Prompt serialization and tool-call parsing for the Cursor CLI backend.

use std::sync::Arc;

use xai_grok_sampling_types::{
    AssistantItem, ContentPart, ConversationItem, ConversationRequest, ConversationResponse,
    ConversationToolChoice, StopReason, TokenUsage, ToolCall,
};

/// Fenced block the model must emit when requesting Grok-side tool execution.
pub const TOOL_CALLS_FENCE: &str = "grok-tool-calls";

const TOOL_PROTOCOL: &str = r#"You are the model backend for Grok Build — not Cursor Agent.
Ignore Cursor ask / agent / plan mode. Those modes do not apply here.
Grok runs every tool on the host via the Available tools list (for example
`run_terminal_command`, `read_file`, `search_replace`, `write`). Editing files
is allowed and expected when the user asks. You do not execute tools yourself;
you request them. Never invent tool results. Never create files under /tmp or a
Cursor temp workspace.

Do NOT refuse edits with "ask mode", "read-only", or "switch to Agent mode".
Do NOT say you are in ask mode, that ask mode is read-only, or that the user
must switch modes — that is wrong here and aborts the turn. Emit tools instead.
Do NOT emit Cursor-native tool syntax (`Shell:`, `Read:`, `Write:`, `Edit:`,
or Cursor function-call UI). Those are ignored and look like rejected shells.
If you need a tool, emit a `grok-tool-calls` fence using Grok tool names only.

When you need a tool, reply with ONLY this fenced JSON block and nothing else
(no prose before or after):

```grok-tool-calls
{"tool_calls":[{"name":"ToolName","arguments":{"arg":"value"}}]}
```

Rules:
- `name` must be exactly one of the Available tools below (e.g.
  `run_terminal_command`, not `Shell` or `Bash`).
- `arguments` must be a JSON object matching that tool's parameters.
- Use paths relative to the Grok working directory (or absolute paths inside it).
- You may include multiple entries in `tool_calls`.
- When you can answer without tools, write normal assistant text only — never
  emit a `grok-tool-calls` fence.
"#;

const TOOL_EXAMPLE: &str = r#"Example (illustrative — prefer exact Available tools names/params):

```grok-tool-calls
{"tool_calls":[{"name":"read_file","arguments":{"target_file":"README.md"}}]}
```

Edits are normal — use `search_replace` / `write` the same way (Grok executes them):

```grok-tool-calls
{"tool_calls":[{"name":"search_replace","arguments":{"file_path":"src/main.rs","old_string":"foo","new_string":"bar"}}]}
```
"#;

/// Build the text prompt passed to `agent --print`.
///
/// When `resume` is true, only trailing tool results are sent (Cursor already
/// has the earlier turn via `--resume`).
/// `workspace` is the Grok session cwd; included so the model targets the
/// real project instead of any Cursor CLI temp path.
pub fn build_prompt(
    request: &ConversationRequest,
    resume: bool,
    workspace: Option<&std::path::Path>,
) -> String {
    if resume {
        let delta = build_resume_delta(request);
        if !delta.is_empty() {
            return delta;
        }
        // Fall through to full prompt if the transcript shape is unexpected.
    }
    build_full_prompt(request, workspace)
}

fn build_full_prompt(request: &ConversationRequest, workspace: Option<&std::path::Path>) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(TOOL_PROTOCOL.to_owned());
    if let Some(cwd) = workspace {
        parts.push(format!(
            "Grok working directory (Grok tools read/write here): {}\n\
             Prefer relative paths from this directory. Never use /tmp Cursor \
             temp workspaces for project edits. File edits go through \
             `grok-tool-calls`, not Cursor Agent.",
            cwd.display()
        ));
    }

    if !request.tools.is_empty() {
        parts.push(TOOL_EXAMPLE.to_owned());
        parts.push("Available tools:".to_owned());
        for tool in &request.tools {
            let params = serde_json::to_string_pretty(&tool.parameters).unwrap_or_else(|_| "{}".into());
            let desc = tool.description.as_deref().unwrap_or("");
            parts.push(format!(
                "Function: {}\nDescription: {}\nParameters:\n{}",
                tool.name, desc, params
            ));
        }
        match &request.tool_choice {
            Some(ConversationToolChoice::None) => {
                parts.push("Tool choice: do not call tools; answer in plain text.".into());
            }
            Some(ConversationToolChoice::Required) => {
                parts.push(
                    "Tool choice: required — you MUST emit a `grok-tool-calls` fence \
                     with at least one tool call; do not answer in plain text."
                        .into(),
                );
            }
            Some(ConversationToolChoice::Function(name)) => {
                parts.push(format!(
                    "Tool choice: you MUST call the `{name}` tool via a `grok-tool-calls` fence."
                ));
            }
            Some(ConversationToolChoice::Auto) | None => {
                parts.push("Tool choice: auto — call a tool only when needed.".into());
            }
        }
    }

    let mut convo: Vec<String> = Vec::new();
    for item in &request.items {
        match item {
            ConversationItem::System(s) => {
                parts.push(format!("System:\n{}", s.content));
            }
            ConversationItem::User(u) => {
                let text = content_parts_to_text(&u.content);
                if !text.is_empty() {
                    convo.push(format!("User: {text}"));
                }
            }
            ConversationItem::Assistant(a) => {
                if !a.tool_calls.is_empty() {
                    let encoded = encode_tool_calls_fence(&a.tool_calls);
                    let body = if a.content.is_empty() {
                        encoded
                    } else {
                        format!("{}\n\n{encoded}", a.content)
                    };
                    convo.push(format!("Assistant: {body}"));
                } else if !a.content.is_empty() {
                    convo.push(format!("Assistant: {}", a.content));
                }
            }
            ConversationItem::ToolResult(t) => {
                convo.push(format!(
                    "Tool result (call_id={}): {}",
                    t.tool_call_id, t.content
                ));
            }
            ConversationItem::Reasoning(r) => {
                let text = xai_grok_sampling_types::reasoning_item_text(r);
                if !text.is_empty() {
                    convo.push(format!("(reasoning): {text}"));
                }
            }
            ConversationItem::BackendToolCall(b) => {
                convo.push(format!("(backend tool): {}", b.text_summary()));
            }
        }
    }

    if !convo.is_empty() {
        parts.push(convo.join("\n\n"));
    }
    if !request.tools.is_empty() {
        parts.push(
            "Reminder: you are Grok Build's model, not Cursor Agent. Emit \
             `grok-tool-calls` with Available tool names (`run_terminal_command`, \
             `read_file`, …) — never Cursor `Shell:` / `Read:` syntax. \
             Editing via Grok tools is allowed and expected."
                .into(),
        );
    }
    parts.push("Assistant:".to_owned());
    parts.join("\n\n")
}

/// Prompt fragment for `--resume`: trailing tool results only.
fn build_resume_delta(request: &ConversationRequest) -> String {
    let mut results: Vec<String> = Vec::new();
    for item in request.items.iter().rev() {
        match item {
            ConversationItem::ToolResult(t) => {
                results.push(format!(
                    "Tool result (call_id={}): {}",
                    t.tool_call_id, t.content
                ));
            }
            _ => break,
        }
    }
    if results.is_empty() {
        return String::new();
    }
    results.reverse();
    let mut parts = results;
    parts.push(
        "Continue. If you need another tool (including `write` / \
         `search_replace` / `run_terminal_command`), emit ONLY a \
         `grok-tool-calls` fence with Grok tool names — never Cursor \
         `Shell:` / `Read:` syntax. Editing via those tools is allowed. \
         Otherwise answer in plain text."
            .into(),
    );
    parts.push("Assistant:".into());
    parts.join("\n\n")
}

fn content_parts_to_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text } => Some(text.as_ref()),
            ContentPart::Image { url } => Some(url.as_ref()),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn encode_tool_calls_fence(tool_calls: &[ToolCall]) -> String {
    let payload = serde_json::json!({
        "tool_calls": tool_calls.iter().map(|t| {
            let args: serde_json::Value = serde_json::from_str(t.arguments.as_ref())
                .unwrap_or_else(|_| serde_json::Value::String(t.arguments.as_ref().to_owned()));
            serde_json::json!({
                "id": t.id.as_ref(),
                "name": t.name,
                "arguments": args,
            })
        }).collect::<Vec<_>>(),
    });
    format!(
        "```{TOOL_CALLS_FENCE}\n{}\n```",
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".into())
    )
}

/// Parsed assistant output after stripping / interpreting tool-call fences.
#[derive(Debug, Clone)]
pub struct ParsedAssistantOutput {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Extract optional `grok-tool-calls` fence (or trailing JSON) from assistant text.
pub fn parse_assistant_output(raw: &str) -> ParsedAssistantOutput {
    if let Some(parsed) = extract_fenced_tool_calls(raw, TOOL_CALLS_FENCE) {
        return parsed;
    }
    // Tolerate models that use a plain `json` fence with the same envelope.
    if let Some(parsed) = extract_fenced_tool_calls(raw, "json")
        && !parsed.tool_calls.is_empty()
    {
        return parsed;
    }
    if let Some(parsed) = extract_trailing_json_tool_calls(raw) {
        return parsed;
    }
    // Last resort: Cursor-native `Shell:` / `Read:` lines never execute on the
    // Grok bridge. Promote them to real tool calls instead of retrying forever.
    if let Some(parsed) = extract_cursor_native_tool_ui(raw) {
        return parsed;
    }
    ParsedAssistantOutput {
        content: raw.trim().to_owned(),
        tool_calls: Vec::new(),
    }
}

/// True when the raw text looks like a (possibly broken) tool-call payload.
/// Used to suppress UI streaming and to trigger retries instead of finishing
/// the turn with raw JSON as the assistant message.
///
/// High-precision: do NOT match prose that merely mentions the fence name
/// (e.g. docs / protocol coaching). Bare `grok-tool-calls` without a markdown
/// fence opener previously false-positived and exhausted retries under Cursor
/// billing as "tool-call-shaped response that could not be parsed".
pub fn looks_like_tool_intent(raw: &str) -> bool {
    let t = raw.trim_start();
    if t.is_empty() {
        return false;
    }
    // Real or truncated fence openers — not the bare fence language tag in prose.
    t.contains("```grok-tool")
        || t.contains("\"tool_calls\"")
        // ` ```json ` alone is common in normal answers; require tool_calls too.
        || (t.starts_with("```json") && t.contains("tool_calls"))
        || (t.starts_with('{') && t.contains("tool_calls"))
        // Cursor-native tool UI mixed into the reply — retry so the model can
        // emit a real Grok fence instead of finishing with a no-op.
        || looks_like_cursor_native_tool_ui(t)
}

fn looks_like_cursor_native_tool_ui(t: &str) -> bool {
    // Lines like `Shell: …` / `Read: …` that Cursor Agent emits; on the Grok
    // bridge these never execute.
    for line in t.lines() {
        let line = line.trim_start();
        if let Some((head, _)) = line.split_once(':') {
            let head = head.trim();
            if matches!(
                head,
                "Shell" | "Bash" | "Read" | "Write" | "Edit" | "Glob" | "Grep" | "TodoWrite"
            ) {
                return true;
            }
        }
    }
    false
}

/// Promote Cursor-native `Shell:` / `Read:` (etc.) lines into Grok tool calls.
///
/// Returns `None` when no recoverable tool lines are present. Prose lines are
/// kept as `content`; tool lines are removed from content.
fn extract_cursor_native_tool_ui(raw: &str) -> Option<ParsedAssistantOutput> {
    let mut tool_calls = Vec::new();
    let mut content_lines = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim_start();
        let Some((head, rest)) = trimmed.split_once(':') else {
            content_lines.push(line);
            continue;
        };
        let head = head.trim();
        let rest = rest.trim();
        if rest.is_empty() {
            content_lines.push(line);
            continue;
        }
        // Only promote one-line tools with enough args. Write/Edit/TodoWrite
        // need multi-line payloads — leave those for retry via tool intent.
        let (name, arguments) = match head {
            "Shell" | "Bash" => (
                "run_terminal_command".to_owned(),
                serde_json::json!({ "command": rest, "description": rest }),
            ),
            "Read" => (
                "read_file".to_owned(),
                serde_json::json!({ "target_file": rest }),
            ),
            "Glob" => ("glob".to_owned(), serde_json::json!({ "pattern": rest })),
            "Grep" => ("grep".to_owned(), serde_json::json!({ "pattern": rest })),
            "LS" => (
                "list_dir".to_owned(),
                serde_json::json!({ "target_directory": rest }),
            ),
            _ => {
                content_lines.push(line);
                continue;
            }
        };
        let (name, arguments) = canonicalize_cursor_bridge_tool(name, arguments);
        let arguments = match arguments {
            serde_json::Value::String(s) => Arc::<str>::from(s),
            other => Arc::<str>::from(other.to_string()),
        };
        let id = format!("cursor_cli_call_{}", tool_calls.len());
        tool_calls.push(ToolCall {
            id: Arc::<str>::from(id),
            name,
            arguments,
        });
    }
    if tool_calls.is_empty() {
        return None;
    }
    Some(ParsedAssistantOutput {
        content: content_lines.join("\n").trim().to_owned(),
        tool_calls,
    })
}

/// True when the assistant text is a Cursor ask-mode / read-only refusal.
///
/// High-precision only: ordinary answers that mention ask mode, say "I can't"
/// about something unrelated, or discuss switching modes must not be retried.
/// Used with tools-offered turns to resample instead of finishing with a no-op.
pub fn looks_like_ask_mode_refusal(raw: &str) -> bool {
    let lower = raw.to_ascii_lowercase();
    if lower.trim().is_empty() {
        return false;
    }

    // Tight whole-phrase hits — each ties refusal to ask/agent mode explicitly.
    // Do NOT list mode-less phrases like "unable to make changes" or
    // "i can only provide guidance": those are common in normal answers and
    // caused retry loops under Cursor billing.
    const PHRASES: &[&str] = &[
        "i'm in ask mode",
        "i am in ask mode",
        "currently in ask mode",
        "we're in ask mode",
        "we are in ask mode",
        "still in ask mode",
        "ask mode is still active",
        "ask-mode is still active",
        "switch out of ask",
        "can't make changes in ask",
        "cannot make changes in ask",
        "can't edit in ask",
        "cannot edit in ask",
        "ask mode is read-only",
        "ask-mode is read-only",
        "in ask mode, i can't",
        "in ask mode i can't",
        "in ask mode, i cannot",
        "in ask mode i cannot",
        "in ask mode and cannot",
        "in ask mode and can't",
        "please switch to agent mode",
        "switch to agent mode to",
        "switch to agent mode if",
        "you'll need to switch to agent",
        "you need to switch to agent",
        "you will need to switch to agent",
    ];
    if PHRASES.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // Combo: ask/read-only mode cue + an *edit/change* refusal (not bare
    // can't/cannot — those pair with identifiers like `ask-mode` in docs).
    let ask_mode_cue = lower.contains("ask mode")
        || lower.contains("ask-mode")
        || lower.contains("read-only mode")
        || lower.contains("readonly mode");
    if !ask_mode_cue {
        return false;
    }
    lower.contains("can't edit")
        || lower.contains("cannot edit")
        || lower.contains("can't make changes")
        || lower.contains("cannot make changes")
        || lower.contains("can't make edits")
        || lower.contains("cannot make edits")
        || lower.contains("unable to edit")
        || lower.contains("unable to make changes")
        || lower.contains("won't edit")
        || lower.contains("will not edit")
        || lower.contains("only answer questions")
        || lower.contains("only provide guidance")
        || lower.contains("only provide information")
        || lower.contains("switch to agent")
}

/// True when streaming text should be held back from the UI (likely a tool payload).
pub fn should_suppress_assistant_stream(accumulated: &str) -> bool {
    let t = accumulated.trim_start();
    if t.is_empty() {
        return false;
    }
    // Hold `{...` and fenced blocks until final parse — avoids flashing raw
    // tool JSON into the transcript when the model is mid-payload.
    t.starts_with('{')
        || t.starts_with("```")
        || t.contains(TOOL_CALLS_FENCE)
        || t.contains("```grok-tool")
        || (t.len() >= 12 && t.contains("\"tool_calls\""))
}

fn extract_fenced_tool_calls(raw: &str, lang: &str) -> Option<ParsedAssistantOutput> {
    let open = format!("```{lang}");
    let start = raw.find(&open)?;
    let after_open = start + open.len();
    let body_start = if raw[after_open..].starts_with('\n') {
        after_open + 1
    } else {
        after_open
    };
    // Prefer a closing fence; if the model omitted it, take the rest.
    let (json_body, after_fence) = if let Some(close_rel) = raw[body_start..].find("```") {
        (
            raw[body_start..body_start + close_rel].trim(),
            raw[body_start + close_rel + 3..].trim(),
        )
    } else {
        (raw[body_start..].trim(), "")
    };
    let tool_calls = parse_tool_calls_json(json_body)?;
    let mut content = String::new();
    content.push_str(raw[..start].trim());
    if !after_fence.is_empty() {
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(after_fence);
    }
    Some(ParsedAssistantOutput {
        content: content.trim().to_owned(),
        tool_calls,
    })
}

fn extract_trailing_json_tool_calls(raw: &str) -> Option<ParsedAssistantOutput> {
    let trimmed = raw.trim();
    if !(trimmed.contains("\"tool_calls\"") || trimmed.contains("tool_calls")) {
        return None;
    }
    // Prefer a JSON object that contains tool_calls (may be preceded by prose).
    let json_slice = if trimmed.starts_with('{') {
        trimmed
    } else {
        let start = trimmed.find('{')?;
        &trimmed[start..]
    };
    let tool_calls = parse_tool_calls_json(json_slice)?;
    let prose = if trimmed.starts_with('{') {
        String::new()
    } else {
        trimmed[..trimmed.find('{').unwrap_or(0)].trim().to_owned()
    };
    Some(ParsedAssistantOutput {
        content: prose,
        tool_calls,
    })
}

#[derive(serde::Deserialize)]
struct WireToolCall {
    #[serde(default)]
    id: Option<String>,
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

fn wire_to_tool_calls(calls: Vec<WireToolCall>) -> Vec<ToolCall> {
    calls
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            let id = t
                .id
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("cursor_cli_call_{i}"));
            let (name, arguments) = canonicalize_cursor_bridge_tool(t.name, t.arguments);
            let arguments = match arguments {
                serde_json::Value::String(s) => Arc::<str>::from(s),
                other => Arc::<str>::from(other.to_string()),
            };
            ToolCall {
                id: Arc::<str>::from(id),
                name,
                arguments,
            }
        })
        .collect()
}

/// Map Cursor/Claude-shaped tool names + args onto Grok's model-facing tools.
///
/// Cursor-bridged models often slip into `Shell` / `Read` / `path` even when the
/// Available tools list uses `run_terminal_command` / `read_file` / `target_file`.
/// Without this remap, those calls fail as unknown tools and look like shell
/// rejections under Cursor billing.
fn canonicalize_cursor_bridge_tool(
    name: String,
    mut arguments: serde_json::Value,
) -> (String, serde_json::Value) {
    let canonical = match name.as_str() {
        "Shell" | "shell" | "Bash" | "bash" | "run_terminal_cmd" => "run_terminal_command",
        "Read" | "read" => "read_file",
        "Write" | "write_file" => "write",
        // MultiEdit uses an edits[] payload that does not match search_replace;
        // leave it unmapped so unknown-tool recovery can steer toward edits.
        "Edit" | "StrReplace" | "str_replace" => "search_replace",
        "Glob" => "glob",
        "LS" | "ls" => "list_dir",
        "Grep" => "grep",
        "WebSearch" => "web_search",
        "WebFetch" => "web_fetch",
        "TodoWrite" => "todo_write",
        "AskUserQuestion" => "ask_user_question",
        other => other,
    };

    // Some models stringify the arguments object; parse so key remaps apply.
    if let serde_json::Value::String(s) = &arguments
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
    {
        arguments = parsed;
    }

    if let serde_json::Value::Object(map) = &mut arguments {
        match canonical {
            "read_file" => {
                rename_json_key(map, "path", "target_file");
                rename_json_key(map, "file_path", "target_file");
            }
            "write" | "search_replace" => {
                rename_json_key(map, "path", "file_path");
                rename_json_key(map, "target_file", "file_path");
                if canonical == "write" {
                    rename_json_key(map, "contents", "content");
                }
            }
            "list_dir" => {
                rename_json_key(map, "path", "target_directory");
                rename_json_key(map, "target_file", "target_directory");
            }
            "glob" => {
                rename_json_key(map, "glob_pattern", "pattern");
                // Cursor sometimes sends `target_directory`; Grok glob uses `path`.
                rename_json_key(map, "target_directory", "path");
            }
            "run_terminal_command" => {
                rename_json_key(map, "cmd", "command");
                rename_json_key(map, "full_command", "command");
            }
            _ => {}
        }
    }

    (canonical.to_owned(), arguments)
}

fn rename_json_key(
    map: &mut serde_json::Map<String, serde_json::Value>,
    from: &str,
    to: &str,
) {
    if map.contains_key(to) {
        return;
    }
    if let Some(v) = map.remove(from) {
        map.insert(to.to_owned(), v);
    }
}

fn parse_tool_calls_json(body: &str) -> Option<Vec<ToolCall>> {
    #[derive(serde::Deserialize)]
    struct Envelope {
        tool_calls: Vec<WireToolCall>,
    }

    let body = body.trim();
    if body.is_empty() {
        return None;
    }

    if let Ok(env) = serde_json::from_str::<Envelope>(body)
        && !env.tool_calls.is_empty()
    {
        return Some(wire_to_tool_calls(env.tool_calls));
    }

    // Models often emit almost-valid envelopes (missing `}` between array
    // elements). Recover by scanning for `"name"` / `"arguments"` pairs.
    let recovered = recover_tool_calls_lenient(body);
    if recovered.is_empty() {
        None
    } else {
        Some(wire_to_tool_calls(recovered))
    }
}

/// Pull individual tool calls out of a broken `tool_calls` JSON payload.
fn recover_tool_calls_lenient(body: &str) -> Vec<WireToolCall> {
    let mut out = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Find `"name"`
        let Some(name_rel) = body[i..].find("\"name\"") else {
            break;
        };
        let name_key = i + name_rel;
        let after_name = name_key + "\"name\"".len();
        let Some(colon) = body[after_name..].find(':') else {
            break;
        };
        let mut p = after_name + colon + 1;
        while p < bytes.len() && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        if p >= bytes.len() || bytes[p] != b'"' {
            i = after_name;
            continue;
        }
        p += 1;
        let name_start = p;
        while p < bytes.len() && bytes[p] != b'"' {
            if bytes[p] == b'\\' {
                p += 1;
            }
            p += 1;
        }
        if p >= bytes.len() {
            break;
        }
        let name = body[name_start..p].to_owned();
        p += 1;

        // Find `"arguments"` after the name.
        let Some(args_rel) = body[p..].find("\"arguments\"") else {
            i = p;
            continue;
        };
        let args_key = p + args_rel;
        let after_args = args_key + "\"arguments\"".len();
        let Some(colon) = body[after_args..].find(':') else {
            i = after_args;
            continue;
        };
        let mut a = after_args + colon + 1;
        while a < bytes.len() && bytes[a].is_ascii_whitespace() {
            a += 1;
        }
        let Some((args_val, consumed)) = parse_json_value_at(body, a) else {
            i = a;
            continue;
        };
        out.push(WireToolCall {
            id: None,
            name,
            arguments: args_val,
        });
        i = a + consumed;
    }
    out
}

/// Parse one JSON value starting at `start`; returns (value, byte length).
fn parse_json_value_at(s: &str, start: usize) -> Option<(serde_json::Value, usize)> {
    let slice = &s[start..];
    let end = end_of_json_value(slice)?;
    let value: serde_json::Value = serde_json::from_str(&slice[..end]).ok()?;
    Some((value, end))
}

/// Byte length of the first complete JSON value in `s`.
fn end_of_json_value(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    match bytes[i] {
        b'"' => {
            i += 1;
            while i < bytes.len() {
                match bytes[i] {
                    b'\\' => i = i.saturating_add(2),
                    b'"' => return Some(i + 1),
                    _ => i += 1,
                }
            }
            None
        }
        b'{' | b'[' => {
            let mut depth = 0i32;
            let mut in_string = false;
            let mut escape = false;
            while i < bytes.len() {
                let c = bytes[i];
                if in_string {
                    if escape {
                        escape = false;
                    } else if c == b'\\' {
                        escape = true;
                    } else if c == b'"' {
                        in_string = false;
                    }
                    i += 1;
                    continue;
                }
                match c {
                    b'"' => in_string = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        b't' if s[i..].starts_with("true") => Some(i + 4),
        b'f' if s[i..].starts_with("false") => Some(i + 5),
        b'n' if s[i..].starts_with("null") => Some(i + 4),
        b'-' | b'0'..=b'9' => {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                i += 1;
            }
            if i == start {
                None
            } else {
                Some(i)
            }
        }
        _ => None,
    }
}

/// True when the request requires at least one tool call.
pub fn tool_calls_required(request: &ConversationRequest) -> bool {
    if request.tools.is_empty() {
        return false;
    }
    matches!(
        request.tool_choice,
        Some(ConversationToolChoice::Required) | Some(ConversationToolChoice::Function(_))
    )
}

/// Build a [`ConversationResponse`] from streamed assistant text + optional usage.
pub fn build_response(
    raw_text: &str,
    model: &str,
    usage: Option<TokenUsage>,
    message_chunks_emitted: u64,
    cursor_session_id: Option<String>,
) -> ConversationResponse {
    let parsed = parse_assistant_output(raw_text);
    let stop_reason = if parsed.tool_calls.is_empty() {
        StopReason::Stop
    } else {
        StopReason::ToolCalls
    };
    let assistant = AssistantItem {
        content: Arc::<str>::from(parsed.content),
        tool_calls: parsed.tool_calls,
        model_id: Some(model.to_owned()),
        model_fingerprint: None,
        reasoning_effort: None,
    };
    ConversationResponse {
        items: vec![ConversationItem::Assistant(assistant)],
        stop_reason: Some(stop_reason),
        usage,
        cost_usd_ticks: None,
        message_chunks_emitted,
        doom_loop_signals: Vec::new(),
        stop_message: None,
        // Reuse message_id to carry the Cursor CLI session id for `--resume`.
        message_id: cursor_session_id,
        raw_stop_reason: None,
        stop_sequence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_tool_calls() {
        let raw = r#"I'll look that up.

```grok-tool-calls
{"tool_calls":[{"name":"read_file","arguments":{"target_file":"Cargo.toml"}}]}
```
"#;
        let parsed = parse_assistant_output(raw);
        assert_eq!(parsed.content, "I'll look that up.");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read_file");
        assert!(parsed.tool_calls[0].arguments.contains("Cargo.toml"));
    }

    #[test]
    fn parses_json_fence_envelope() {
        let raw = r#"```json
{"tool_calls":[{"name":"run_terminal_command","arguments":{"command":"ls"}}]}
```"#;
        let parsed = parse_assistant_output(raw);
        assert!(parsed.content.is_empty());
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "run_terminal_command");
    }

    #[test]
    fn remaps_cursor_shell_and_read_aliases() {
        let raw = r#"```grok-tool-calls
{"tool_calls":[
  {"name":"Shell","arguments":{"command":"git status"}},
  {"name":"Read","arguments":{"path":"README.md"}}
]}
```"#;
        let parsed = parse_assistant_output(raw);
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].name, "run_terminal_command");
        assert!(parsed.tool_calls[0].arguments.contains("git status"));
        assert_eq!(parsed.tool_calls[1].name, "read_file");
        assert!(parsed.tool_calls[1].arguments.contains("target_file"));
        assert!(parsed.tool_calls[1].arguments.contains("README.md"));
        assert!(!parsed.tool_calls[1].arguments.contains("\"path\""));
    }

    #[test]
    fn remaps_cursor_glob_to_glob_not_list_dir() {
        let raw = r#"```grok-tool-calls
{"tool_calls":[
  {"name":"Glob","arguments":{"glob_pattern":"**/*.rs","target_directory":"src"}},
  {"name":"LS","arguments":{"path":"crates"}}
]}
```"#;
        let parsed = parse_assistant_output(raw);
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].name, "glob");
        assert!(parsed.tool_calls[0].arguments.contains("\"pattern\""));
        assert!(parsed.tool_calls[0].arguments.contains("**/*.rs"));
        assert!(parsed.tool_calls[0].arguments.contains("\"path\""));
        assert!(!parsed.tool_calls[0].arguments.contains("glob_pattern"));
        assert_eq!(parsed.tool_calls[1].name, "list_dir");
        assert!(parsed.tool_calls[1].arguments.contains("target_directory"));
    }

    #[test]
    fn does_not_remap_multiedit_to_search_replace() {
        let raw = r#"```grok-tool-calls
{"tool_calls":[{"name":"MultiEdit","arguments":{"file_path":"a.rs","edits":[{"old_string":"a","new_string":"b"}]}}]}
```"#;
        let parsed = parse_assistant_output(raw);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "MultiEdit");
        assert!(parsed.tool_calls[0].arguments.contains("edits"));
    }

    #[test]
    fn plain_text_passthrough() {
        let parsed = parse_assistant_output("hello");
        assert_eq!(parsed.content, "hello");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn resume_delta_uses_trailing_tool_results() {
        let request = ConversationRequest::from_items(vec![
            ConversationItem::user("hi"),
            ConversationItem::assistant_tool_calls(vec![ToolCall {
                id: "c1".into(),
                name: "read_file".into(),
                arguments: r#"{"target_file":"a"}"#.into(),
            }]),
            ConversationItem::tool_result("c1", "file contents"),
        ]);
        let prompt = build_prompt(&request, true, None);
        assert!(prompt.contains("Tool result (call_id=c1)"));
        assert!(!prompt.contains("User: hi"));
        assert!(prompt.contains("Assistant:"));
        assert!(prompt.contains("run_terminal_command"));
        assert!(!prompt.contains("Write/Edit/Shell"));
    }

    #[test]
    fn recovers_malformed_multi_tool_payload() {
        // Missing `}` after the first tool's arguments (common model slip).
        let raw = r#"{"tool_calls":[{"name":"todo_write","arguments":{"merge":false,"todos":[{"id":"1","content":"a","status":"in_progress"},{"id":"2","content":"b","status":"pending"}]},{"name":"grep","arguments":{"pattern":"Upgrade","glob":"*.rs"}},{"name":"grep","arguments":{"pattern":"cursor","glob":"*.ts"}}]}"#;
        assert!(serde_json::from_str::<serde_json::Value>(raw).is_err());
        let parsed = parse_assistant_output(raw);
        assert_eq!(parsed.tool_calls.len(), 3);
        assert_eq!(parsed.tool_calls[0].name, "todo_write");
        assert_eq!(parsed.tool_calls[1].name, "grep");
        assert_eq!(parsed.tool_calls[2].name, "grep");
        assert!(parsed.content.is_empty());
    }

    #[test]
    fn looks_like_tool_intent_detects_raw_json() {
        assert!(looks_like_tool_intent(
            r#"{"tool_calls":[{"name":"read_file","arguments":{}}]}"#
        ));
        assert!(!looks_like_tool_intent("hello world"));
    }

    #[test]
    fn looks_like_tool_intent_detects_cursor_shell_ui_and_truncated_fence() {
        assert!(looks_like_tool_intent(
            "Shell: I'll stage everything and amend with a proper message."
        ));
        assert!(looks_like_tool_intent(
            "Checking status.```grok-tool-Amending the last commit"
        ));
    }

    #[test]
    fn looks_like_tool_intent_ignores_prose_fence_mention_and_plain_json_fence() {
        // Protocol coaching / docs that name the fence must not exhaust retries.
        assert!(!looks_like_tool_intent(
            "When you need a tool, emit a grok-tool-calls fence with Grok names."
        ));
        assert!(!looks_like_tool_intent(
            "```json\n{\"foo\": 1}\n```"
        ));
        assert!(looks_like_tool_intent(
            "```json\n{\"tool_calls\":[]}\n```"
        ));
    }

    #[test]
    fn parses_cursor_native_shell_and_read_lines() {
        let raw = "Checking.\nShell: git status\nRead: README.md\n";
        let parsed = parse_assistant_output(raw);
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].name, "run_terminal_command");
        assert!(parsed.tool_calls[0].arguments.contains("git status"));
        assert_eq!(parsed.tool_calls[1].name, "read_file");
        assert!(parsed.tool_calls[1].arguments.contains("README.md"));
        assert!(parsed.content.contains("Checking"));
        assert!(!parsed.content.contains("Shell:"));
    }

    #[test]
    fn suppress_holds_partial_tool_json_before_full_fence() {
        // Progressive deltas must not leak `{` / `"tool_calls"` into the UI.
        assert!(should_suppress_assistant_stream("{"));
        assert!(should_suppress_assistant_stream(r#"{"tool_"#));
        assert!(should_suppress_assistant_stream(
            r#"{"tool_calls":[{"name":"read_file""#
        ));
        assert!(should_suppress_assistant_stream("```grok-tool-calls"));
        assert!(should_suppress_assistant_stream(
            "Sure.\n\n```grok-tool-calls\n{\"tool_calls\":[]}"
        ));
        assert!(!should_suppress_assistant_stream("Sure, I can help with that."));
    }

    #[test]
    fn protocol_overrides_cursor_ask_mode_refusal() {
        assert!(TOOL_PROTOCOL.contains("Ignore Cursor ask"));
        assert!(TOOL_PROTOCOL.contains("Editing files\nis allowed"));
        assert!(TOOL_PROTOCOL.contains("Do NOT refuse edits"));
        assert!(TOOL_PROTOCOL.contains("Do NOT say you are in ask mode"));
        assert!(TOOL_PROTOCOL.contains("run_terminal_command"));
        assert!(TOOL_PROTOCOL.contains("Do NOT emit Cursor-native tool syntax"));
        assert!(!TOOL_PROTOCOL.contains("ask-mode is\nread-only"));
        // Must not coach Cursor-native Shell/Read as Grok tool names.
        assert!(!TOOL_PROTOCOL.contains("(Read, Write, Edit, Shell"));
        assert!(!TOOL_EXAMPLE.contains("\"name\":\"Read\""));
        assert!(TOOL_EXAMPLE.contains("read_file"));
        // Edit example steers away from ask-mode "I can't write" refusals.
        assert!(TOOL_EXAMPLE.contains("search_replace"));
        assert!(TOOL_EXAMPLE.contains("Edits are normal"));
    }

    #[test]
    fn looks_like_ask_mode_refusal_detects_common_phrases() {
        assert!(looks_like_ask_mode_refusal(
            "I'm in ask mode, so I can't edit files. Switch to agent mode."
        ));
        assert!(looks_like_ask_mode_refusal(
            "I am currently in ask mode and cannot make changes."
        ));
        assert!(looks_like_ask_mode_refusal(
            "Ask mode is read-only — please switch to Agent mode to apply edits."
        ));
        assert!(looks_like_ask_mode_refusal(
            "I can only provide guidance while in ask-mode."
        ));
        assert!(looks_like_ask_mode_refusal(
            "Ask mode is still active. I can't make changes to the repo."
        ));
        assert!(looks_like_ask_mode_refusal(
            "Please switch to Agent mode to apply these edits."
        ));
    }

    #[test]
    fn looks_like_ask_mode_refusal_ignores_benign_mentions() {
        // Docs / explanations that mention the words without refusing.
        assert!(!looks_like_ask_mode_refusal(
            "Grok Shift+Tab ask mode is separate from Cursor CLI --mode ask."
        ));
        assert!(!looks_like_ask_mode_refusal(
            "The bridge always uses ask mode; Grok still runs write tools."
        ));
        assert!(!looks_like_ask_mode_refusal("I'll update the file now."));
        assert!(!looks_like_ask_mode_refusal("hello world"));
        // Mode-less phrases that previously false-positived and exhausted retries.
        assert!(!looks_like_ask_mode_refusal(
            "I can only provide guidance on the API design for now."
        ));
        assert!(!looks_like_ask_mode_refusal(
            "Unable to make changes to prod; I'll edit the migration instead."
        ));
        assert!(!looks_like_ask_mode_refusal(
            "You can switch to agent if you want Cursor-native tools."
        ));
        // ask-mode in an identifier / topic + unrelated can't/cannot.
        assert!(!looks_like_ask_mode_refusal(
            "The ask-mode refusal detector cannot match identifiers like looks_like_ask_mode_refusal."
        ));
        assert!(!looks_like_ask_mode_refusal(
            "The bridge always uses ask mode so Cursor cannot execute tools; Grok runs them."
        ));
        assert!(!looks_like_ask_mode_refusal(
            "I can't fix that without reading the ask-mode code path first."
        ));
    }
}
