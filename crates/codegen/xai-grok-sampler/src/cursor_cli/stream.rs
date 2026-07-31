//! Layer-2 style stream: ConversationRequest → SamplingEvents via Cursor CLI.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::Stream;
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use xai_grok_sampling_types::{ConversationRequest, TokenUsage};

use crate::events::{SamplingChannel, SamplingErrorInfo, SamplingErrorKind, SamplingEvent};
use crate::metrics::InferenceLatencyStats;
use crate::types::RequestId;

use super::process::{
    CursorCliError, CursorNdjsonEvent, parse_ndjson_line, read_stdout_lines, spawn_ask_stream,
};
use super::prompt::{
    build_prompt, build_response, looks_like_tool_intent, parse_assistant_output,
    should_suppress_assistant_stream, tool_calls_required,
};

/// Options for one Cursor CLI sampling turn.
#[derive(Debug, Clone, Default)]
pub struct CursorCliStreamOpts {
    /// Resume a prior Cursor CLI session (`--resume`).
    pub resume: Option<String>,
    /// When set, the stream writes the latest Cursor `session_id` here on success
    /// so the shell can resume on the next tool-result turn.
    pub session_slot: Option<Arc<Mutex<Option<String>>>>,
    /// Real project cwd for `--workspace` (ask-mode is read-only).
    pub workspace: Option<std::path::PathBuf>,
}

/// Drive one Cursor CLI turn into [`SamplingEvent`]s.
///
/// Always uses `--mode ask`. Grok owns tools via the `grok-tool-calls` fence.
pub fn stream_cursor_cli(
    request: ConversationRequest,
    model: String,
    request_id: RequestId,
    idle_timeout: Duration,
    cancel_token: CancellationToken,
    opts: CursorCliStreamOpts,
) -> Pin<Box<dyn Stream<Item = SamplingEvent> + Send>> {
    Box::pin(async_stream::stream! {
        let stream_start = Instant::now();
        yield SamplingEvent::StreamStarted {
            request_id: request_id.clone(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        };

        let resume = opts.resume.filter(|s| !s.is_empty());
        let is_resume = resume.is_some();
        let workspace = opts.workspace.clone();
        let prompt = build_prompt(&request, is_resume, workspace.as_deref());
        let tools_required = tool_calls_required(&request);

        let spawned = match spawn_ask_stream(
            &model,
            &prompt,
            resume.as_deref(),
            workspace.as_deref(),
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                yield failed_event(&request_id, e);
                return;
            }
        };
        let mut child = spawned.child;
        let _temp_workspace = spawned.temp_workspace;

        let mut lines = std::pin::pin!(read_stdout_lines(spawned.stdout));
        let mut assistant_text = String::new();
        let mut first_token = false;
        let mut chunk_index = 0u64;
        let mut message_chunks = 0u64;
        let mut chunk_timestamps: Vec<Instant> = Vec::new();
        let mut usage: Option<TokenUsage> = None;
        let mut fatal: Option<CursorCliError> = None;
        let mut cursor_session_id: Option<String> = None;
        let mut native_tool_warned = false;

        loop {
            let next = tokio::select! {
                biased;
                _ = cancel_token.cancelled() => {
                    let _ = child.kill().await;
                    // Drop without a terminal event; drive_l2 treats cancel
                    // via its own select and returns AttemptOutcome::Cancelled.
                    return;
                }
                item = tokio::time::timeout(idle_timeout, lines.next()) => item,
            };

            match next {
                Err(_) => {
                    let _ = child.kill().await;
                    fatal = Some(CursorCliError::Timeout);
                    break;
                }
                Ok(None) => break,
                Ok(Some(Err(e))) => {
                    let _ = child.kill().await;
                    fatal = Some(CursorCliError::Io(e));
                    break;
                }
                Ok(Some(Ok(line))) => {
                    let Some(event) = parse_ndjson_line(&line) else {
                        continue;
                    };
                    match event {
                        CursorNdjsonEvent::ThinkingDelta(text) => {
                            if !first_token {
                                first_token = true;
                                yield SamplingEvent::FirstToken {
                                    request_id: request_id.clone(),
                                };
                            }
                            chunk_timestamps.push(Instant::now());
                            let idx = chunk_index;
                            chunk_index += 1;
                            yield SamplingEvent::ChannelToken {
                                request_id: request_id.clone(),
                                channel: SamplingChannel::Reasoning,
                                text,
                                chunk_index: idx,
                            };
                        }
                        CursorNdjsonEvent::AssistantText(text) => {
                            let Some(delta) = merge_assistant_delta(&mut assistant_text, &text) else {
                                continue;
                            };
                            // Hold back tool-call JSON/fences so they never flash
                            // into the transcript as a finished assistant message.
                            if should_suppress_assistant_stream(&assistant_text) {
                                if !first_token {
                                    first_token = true;
                                    yield SamplingEvent::FirstToken {
                                        request_id: request_id.clone(),
                                    };
                                }
                                chunk_timestamps.push(Instant::now());
                                continue;
                            }
                            if !first_token {
                                first_token = true;
                                yield SamplingEvent::FirstToken {
                                    request_id: request_id.clone(),
                                };
                            }
                            chunk_timestamps.push(Instant::now());
                            message_chunks += 1;
                            let idx = chunk_index;
                            chunk_index += 1;
                            yield SamplingEvent::ChannelToken {
                                request_id: request_id.clone(),
                                channel: SamplingChannel::Text,
                                text: delta,
                                chunk_index: idx,
                            };
                        }
                        CursorNdjsonEvent::NativeToolCall => {
                            if !native_tool_warned {
                                native_tool_warned = true;
                                warn!(
                                    "Cursor CLI emitted a native tool_call in ask mode; \
                                     ignoring (Grok owns tools)"
                                );
                            }
                        }
                        CursorNdjsonEvent::Result {
                            text,
                            usage: u,
                            is_error,
                            error_message,
                            session_id,
                        } => {
                            if let Some(sid) = session_id {
                                cursor_session_id = Some(sid);
                            }
                            if is_error {
                                fatal = Some(CursorCliError::Exit {
                                    code: Some(1),
                                    stderr: error_message.unwrap_or_else(|| {
                                        text.unwrap_or_else(|| "Cursor Agent CLI error".into())
                                    }),
                                });
                                break;
                            }
                            if let Some(t) = text {
                                let _ = merge_assistant_delta(&mut assistant_text, &t);
                            }
                            if let Some(u) = u {
                                usage = Some(TokenUsage {
                                    prompt_tokens: u.input_tokens,
                                    completion_tokens: u.output_tokens,
                                    total_tokens: u.input_tokens + u.output_tokens,
                                    reasoning_tokens: 0,
                                    cached_prompt_tokens: u.cache_read_tokens,
                                    cache_creation_prompt_tokens: u.cache_write_tokens,
                                });
                            }
                        }
                        CursorNdjsonEvent::SessionId(session_id) => {
                            debug!(%session_id, "Cursor Agent CLI session started");
                            cursor_session_id = Some(session_id);
                        }
                        CursorNdjsonEvent::Other => {}
                    }
                }
            }
        }

        let stderr = drain_stderr(&mut child).await;
        let wait_status = child.wait().await;

        match wait_status {
            Ok(status) if status.success() || (fatal.is_none() && !assistant_text.is_empty()) => {
                // Auth failures often exit 0 with an empty stream and a stderr
                // message — treat those as fatal instead of empty-response retries.
                if assistant_text.is_empty()
                    && fatal.is_none()
                    && let Some(err) = classify_cli_stderr(&stderr, status.code())
                {
                    fatal = Some(err);
                }
            }
            Ok(status) if !status.success() && fatal.is_none() => {
                fatal = Some(classify_cli_stderr(&stderr, status.code()).unwrap_or_else(|| {
                    CursorCliError::Exit {
                        code: status.code(),
                        stderr: stderr.clone(),
                    }
                }));
            }
            Ok(_) => {}
            Err(e) if fatal.is_none() => fatal = Some(CursorCliError::Io(e)),
            Err(_) => {}
        }

        if let Some(err) = fatal {
            yield failed_event(&request_id, err);
            return;
        }

        if assistant_text.is_empty() {
            let message = if stderr.trim().is_empty() {
                "Cursor Agent CLI returned no assistant text".into()
            } else {
                format!("Cursor Agent CLI returned no assistant text: {}", stderr.trim())
            };
            let auth = is_auth_stderr(&stderr);
            yield SamplingEvent::Failed {
                request_id: request_id.clone(),
                error: SamplingErrorInfo {
                    kind: if auth {
                        SamplingErrorKind::Auth
                    } else {
                        SamplingErrorKind::EmptyResponse
                    },
                    status_code: None,
                    message,
                    // Auth / misconfig must not spin the retry loop forever.
                    is_retryable: !auth,
                    retry_after_secs: None,
                    model_metadata: None,
                    empty_response_context: None,
                    doom_loop_triggers: None,
                    doom_loop_aborted_at_chunk: None,
                },
            };
            return;
        }

        let parsed = parse_assistant_output(&assistant_text);
        let tools_offered = !request.tools.is_empty();
        if parsed.tool_calls.is_empty()
            && tools_offered
            && (tools_required || looks_like_tool_intent(&assistant_text))
        {
            yield SamplingEvent::Failed {
                request_id: request_id.clone(),
                error: SamplingErrorInfo {
                    kind: SamplingErrorKind::EmptyResponse,
                    status_code: None,
                    message: "Cursor CLI emitted a tool-call-shaped response that \
                         could not be parsed; retrying"
                        .into(),
                    is_retryable: true,
                    retry_after_secs: None,
                    model_metadata: None,
                    empty_response_context: None,
                    doom_loop_triggers: None,
                    doom_loop_aborted_at_chunk: None,
                },
            };
            return;
        }

        // Tool payloads were suppressed from the text channel; keep
        // message_chunks at whatever plain-text count we emitted.
        let message_chunks = if !parsed.tool_calls.is_empty() {
            0
        } else {
            message_chunks
        };

        if let Some(slot) = &opts.session_slot
            && let Some(sid) = cursor_session_id.clone()
            && let Ok(mut guard) = slot.lock()
        {
            *guard = Some(sid);
        }

        let stream_end = Instant::now();
        let metrics =
            InferenceLatencyStats::from_timestamps(stream_start, &chunk_timestamps, stream_end);
        let response = build_response(
            &assistant_text,
            &model,
            usage,
            message_chunks,
            cursor_session_id,
        );
        yield SamplingEvent::Completed {
            request_id: request_id.clone(),
            response: Box::new(response),
            metrics,
        };
    })
}

/// Merge a new assistant fragment into `acc`. Returns the delta to emit, if any.
fn merge_assistant_delta(acc: &mut String, text: &str) -> Option<String> {
    if text.is_empty() || text == acc.as_str() {
        return None;
    }
    if text.starts_with(acc.as_str()) {
        let delta = text[acc.len()..].to_owned();
        *acc = text.to_owned();
        if delta.is_empty() {
            None
        } else {
            Some(delta)
        }
    } else {
        acc.push_str(text);
        Some(text.to_owned())
    }
}

async fn drain_stderr(child: &mut tokio::process::Child) -> String {
    let Some(mut err) = child.stderr.take() else {
        return String::new();
    };
    use tokio::io::AsyncReadExt;
    let mut buf = String::new();
    let _ = err.read_to_string(&mut buf).await;
    buf
}

fn is_auth_stderr(stderr: &str) -> bool {
    let lower = stderr.to_ascii_lowercase();
    lower.contains("authentication required")
        || lower.contains("agent login")
        || lower.contains("cursor_api_key")
        || lower.contains("not authenticated")
        || lower.contains("unauthorized")
}

fn classify_cli_stderr(stderr: &str, code: Option<i32>) -> Option<CursorCliError> {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(CursorCliError::Exit {
        code: code.or(Some(1)),
        stderr: trimmed.to_owned(),
    })
}

fn failed_event(request_id: &RequestId, err: CursorCliError) -> SamplingEvent {
    warn!(error = %err, "Cursor CLI sampling failed");
    let (kind, status_code, is_retryable) = match &err {
        CursorCliError::NotFound => (SamplingErrorKind::Api, None, false),
        CursorCliError::Timeout => (SamplingErrorKind::IdleTimeout, None, true),
        CursorCliError::Exit { stderr, .. } if is_auth_stderr(stderr) => {
            (SamplingErrorKind::Auth, Some(401), false)
        }
        CursorCliError::Exit { .. } => (SamplingErrorKind::Api, Some(500), true),
        CursorCliError::Spawn(_) | CursorCliError::Io(_) => (SamplingErrorKind::Http, None, true),
    };
    SamplingEvent::Failed {
        request_id: request_id.clone(),
        error: SamplingErrorInfo {
            kind,
            status_code,
            message: err.to_string(),
            is_retryable,
            retry_after_secs: None,
            model_metadata: None,
            empty_response_context: None,
            doom_loop_triggers: None,
            doom_loop_aborted_at_chunk: None,
        },
    }
}
