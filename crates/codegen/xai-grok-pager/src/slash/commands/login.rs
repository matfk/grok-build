//! `/login` -- log in or re-authenticate with your account.
//!
//! Usage:
//! - `/login` — SpaceXAI / enterprise login, or Cursor when the active
//!   model is a Cursor-billed route (`cursor/…`)
//! - `/login cursor` — Cursor Agent CLI subscription (`agent login`)

use crate::app::actions::Action;
use crate::slash::command::{CommandExecCtx, CommandResult, SlashCommand};

pub struct LoginCommand;

impl SlashCommand for LoginCommand {
    fn name(&self) -> &str {
        "login"
    }

    fn description(&self) -> &str {
        "Log in or re-authenticate (Grok or Cursor)"
    }

    fn usage(&self) -> &str {
        "/login [cursor]"
    }

    fn run(&self, ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        let provider = args.trim().to_ascii_lowercase();
        if provider.is_empty() {
            // On a Cursor-billed model, bare `/login` means Cursor login so
            // new users who picked `cursor/auto` are not sent through grok.com.
            let on_cursor = ctx
                .models
                .current_model_id_str()
                .is_some_and(xai_grok_shell::agent::cursor_cli::is_cursor_billed_model);
            if on_cursor {
                CommandResult::Action(Action::LoginCursor)
            } else {
                CommandResult::Action(Action::Login)
            }
        } else if provider == "cursor" || provider == "cursor.cli" {
            CommandResult::Action(Action::LoginCursor)
        } else {
            CommandResult::Message(format!(
                "Unknown login provider '{provider}'. Use `/login` or `/login cursor`."
            ))
        }
    }
}
