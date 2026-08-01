//! `/usage` — session token/cost; consumer accounts can also manage billing.
//!
//! External-auth deployments (`auth_provider_command`) never reach grok.com
//! billing, so the command is hidden and refused via
//! [`AppCtx::usage_command_visible`].

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, ArgItem, CommandExecCtx, CommandResult, SlashCommand};
use agent_client_protocol as acp;

pub struct UsageCommand;

/// Detect external-auth installs once at pager startup.
pub(crate) fn detect_external_auth_provider(auth_methods: &[acp::AuthMethod]) -> bool {
    auth_methods.iter().any(auth_method_is_external_provider)
        || auth_provider_env_set()
        || auth_provider_config_set()
}

fn auth_method_is_external_provider(method: &acp::AuthMethod) -> bool {
    // Cursor CLI is additive and may historically have carried
    // `external_provider`; never treat it as enterprise auth that hides `/usage`.
    if method.id().0.as_ref() == xai_grok_shell::agent::auth_method::CURSOR_CLI_METHOD_ID {
        return false;
    }
    let meta = method.meta();
    let Some(meta) = meta.as_ref() else {
        return false;
    };
    if meta
        .get("cursor_cli")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return false;
    }
    meta.get("external_provider")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn auth_provider_env_set() -> bool {
    std::env::var("GROK_AUTH_PROVIDER_COMMAND")
        .ok()
        .is_some_and(|s| !s.trim().is_empty())
}

fn auth_provider_config_set() -> bool {
    let Ok(raw) = xai_grok_shell::config::load_effective_config() else {
        return false;
    };
    let Ok(cfg) = xai_grok_shell::agent::config::Config::new_from_toml_cfg(&raw) else {
        return false;
    };
    cfg.grok_com_config
        .auth_provider_command
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty())
}

impl SlashCommand for UsageCommand {
    fn name(&self) -> &str {
        "usage"
    }

    fn aliases(&self) -> &[&str] {
        &["cost"]
    }

    fn description(&self) -> &str {
        "View usage"
    }

    fn usage(&self) -> &str {
        "/usage [show|manage]"
    }

    fn takes_args(&self) -> bool {
        true
    }

    fn visible(&self, ctx: &AppCtx) -> bool {
        ctx.usage_command_visible
    }

    fn takes_args_now(&self, ctx: &AppCtx) -> bool {
        // Non-consumer: bare `/usage` only — Enter should send, not chain for args.
        ctx.usage_command_visible && ctx.billing_surface_visible
    }

    fn suggest_args(&self, ctx: &AppCtx, _args_query: &str) -> Option<Vec<ArgItem>> {
        if !ctx.usage_command_visible || !ctx.billing_surface_visible {
            return None;
        }
        Some(vec![
            ArgItem {
                display: "show".into(),
                match_text: "show".into(),
                insert_text: "show".into(),
                description: "View usage".into(),
            },
            ArgItem {
                display: "manage".into(),
                match_text: "manage".into(),
                insert_text: "manage".into(),
                description: "Manage billing".into(),
            },
        ])
    }

    fn run(&self, ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        if !ctx.usage_command_visible {
            return CommandResult::Error("/usage is not available.".into());
        }
        let arg = args.trim();
        if !ctx.billing_surface_visible {
            return match arg {
                "" => CommandResult::Action(Action::ShowUsage),
                _ => CommandResult::Error(format!("Unknown argument: {arg}. Use /usage")),
            };
        }
        match arg {
            "" | "show" => CommandResult::Action(Action::ShowUsage),
            "manage" => CommandResult::Action(Action::ManageBilling),
            _ => CommandResult::Error(format!(
                "Unknown argument: {arg}. Use /usage show or /usage manage"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_cli_method_does_not_hide_usage() {
        let cursor = xai_grok_shell::agent::auth_method::cursor_cli_auth_method();
        assert!(
            !detect_external_auth_provider(&[cursor]),
            "advertising cursor.cli must not hide /usage",
        );
    }

    #[test]
    fn cursor_cli_ignored_even_if_external_provider_flag_set() {
        let mut meta = acp::Meta::new();
        meta.insert("external_provider".to_owned(), serde_json::json!(true));
        meta.insert("cursor_cli".to_owned(), serde_json::json!(true));
        let cursor = acp::AuthMethod::Agent(
            acp::AuthMethodAgent::new(
                acp::AuthMethodId::new(
                    xai_grok_shell::agent::auth_method::CURSOR_CLI_METHOD_ID,
                ),
                "Cursor".to_string(),
            )
            .meta(Some(meta)),
        );
        assert!(
            !auth_method_is_external_provider(&cursor),
            "legacy cursor.cli + external_provider must still leave /usage visible",
        );
    }
}
