//! Cursor Agent CLI: model discovery + paywall policy.
//!
//! Two separate concerns:
//!
//! 1. **Enablement** ([`cursor_models_enabled`]): whether to inject `cursor/<id>`
//!    catalog entries. Auto-detects when `agent` is on `PATH`.
//! 2. **Paywall policy** ([`suppress_supergrok_paywalls`]): whether SuperGrok
//!    access gates / upgrade upsells should be suppressed. True only when the
//!    active (or selected) route is Cursor-billed, or the user set an explicit
//!    `cursor_cli = true` / `GROK_CURSOR_CLI=1` override — **not** merely
//!    because the CLI is installed.
//!
//! Inference always uses ask-mode; Grok still runs client-side tools.

use std::num::NonZeroU64;
use std::sync::OnceLock;

use indexmap::IndexMap;
use xai_grok_sampler::cursor_cli::{
    CURSOR_CLI_API_KEY, CURSOR_CLI_BASE_URL, DiscoveredCursorModel, agent_available, list_models,
};
use xai_grok_sampling_types::ApiBackend;

use super::config::{Config, ModelEntry, ModelInfo};

const DEFAULT_CURSOR_CONTEXT_WINDOW: u64 = 200_000;

/// Fallback model injected when the CLI is present but not yet authenticated,
/// so new users can pick a Cursor route and run `/login cursor`.
const BOOTSTRAP_CURSOR_MODEL_ID: &str = "auto";

/// Env override for Cursor CLI (`GROK_CURSOR_CLI`). `None` = fall through.
fn cursor_cli_env_override() -> Option<bool> {
    let Ok(raw) = std::env::var("GROK_CURSOR_CLI") else {
        return None;
    };
    let v = raw.trim();
    if matches!(v, "1" | "true" | "TRUE" | "yes" | "on") {
        return Some(true);
    }
    if matches!(v, "0" | "false" | "FALSE" | "no" | "off") {
        return Some(false);
    }
    None
}

/// Cached `agent --version` probe. Must not run on every TUI frame — spawning
/// the Node CLI takes hundreds of ms and freezes the pager.
fn agent_available_cached() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(agent_available)
}

/// Resolve whether Cursor CLI models should be injected into the catalog.
///
/// Order: `GROK_CURSOR_CLI` env → `[features].cursor_cli` → auto (`agent` on PATH).
///
/// This is **enablement only**. It does not grant SuperGrok-equivalent access;
/// see [`suppress_supergrok_paywalls`].
pub fn cursor_models_enabled(cfg: &Config) -> bool {
    if let Some(v) = cursor_cli_env_override() {
        return v;
    }
    match cfg.features.cursor_cli {
        Some(v) => v,
        None => agent_available_cached(),
    }
}

/// Alias kept for call sites that mean enablement (catalog injection).
#[inline]
pub fn cursor_cli_enabled(cfg: &Config) -> bool {
    cursor_models_enabled(cfg)
}

/// Explicit power-user opt-in for paywall suppression.
///
/// True only when the user set `GROK_CURSOR_CLI=1`/`true`/… or
/// `[features].cursor_cli = true`. Auto PATH detection does **not** count —
/// installing the CLI must not hide SuperGrok upsells for free-tier xAI users.
pub fn cursor_paywall_override(cfg: &Config) -> bool {
    if let Some(v) = cursor_cli_env_override() {
        return v;
    }
    matches!(cfg.features.cursor_cli, Some(true))
}

/// Disk/config variant of [`cursor_paywall_override`] (no live agent `Config`).
///
/// Process-cached: pager hot paths call this often; re-reading TOML each time
/// is wasteful. Env is read inside the init closure once per process — tests
/// that mutate `GROK_CURSOR_CLI` should prefer [`suppress_supergrok_paywalls`]
/// with an explicit model id, or set the env before first call.
pub fn cursor_paywall_override_from_disk() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        if let Some(v) = cursor_cli_env_override() {
            return v;
        }
        crate::config::load_effective_config()
            .ok()
            .and_then(|root| {
                root.get("features")
                    .and_then(|f| f.get("cursor_cli"))
                    .and_then(|v| v.as_bool())
            })
            .unwrap_or(false)
    })
}

/// Whether a catalog key / model id is a Cursor-billed inference route.
///
/// Auto-discovered entries use the `cursor/<id>` prefix; manual
/// `api_backend = "cursor_cli"` configs should also use that prefix (or set
/// `apiBackend` in ACP meta — see [`is_cursor_billed_model_info`]).
pub fn is_cursor_billed_model(model_id: &str) -> bool {
    let id = model_id.trim();
    id.starts_with("cursor/") || id.eq_ignore_ascii_case("cursor")
}

/// ACP [`acp::ModelInfo`] helper: Cursor-billed if id matches or meta says so.
pub fn is_cursor_billed_model_info(model_id: &str, meta: Option<&serde_json::Map<String, serde_json::Value>>) -> bool {
    if is_cursor_billed_model(model_id) {
        return true;
    }
    meta.and_then(|m| m.get("apiBackend").or_else(|| m.get("api_backend")))
        .and_then(|v| v.as_str())
        .is_some_and(|b| b.eq_ignore_ascii_case("cursor_cli"))
}

/// Suppress SuperGrok access gates / upgrade upsells / tier cosmetics.
///
/// - Explicit override ([`cursor_paywall_override_from_disk`]), **or**
/// - Active/selected model is a Cursor-billed route.
///
/// Installing `agent` alone does not suppress paywalls while the user is on
/// xAI free-tier models.
pub fn suppress_supergrok_paywalls(active_model_id: Option<&str>) -> bool {
    if cursor_paywall_override_from_disk() {
        return true;
    }
    active_model_id.is_some_and(is_cursor_billed_model)
}

/// Shell-side variant with live [`Config`] (explicit override from cfg, not disk cache).
pub fn suppress_supergrok_paywalls_cfg(cfg: &Config, active_model_id: Option<&str>) -> bool {
    if cursor_paywall_override(cfg) {
        return true;
    }
    active_model_id.is_some_and(is_cursor_billed_model)
}

/// Deprecated name: prefer [`suppress_supergrok_paywalls_cfg`] with the active model.
///
/// Without a model id this only honors the **explicit** override (not PATH auto).
#[deprecated(note = "pass the active model id to suppress_supergrok_paywalls_cfg")]
pub fn cursor_bypasses_access_gate(cfg: &Config) -> bool {
    suppress_supergrok_paywalls_cfg(cfg, None)
}

/// Deprecated name: prefer [`suppress_supergrok_paywalls`] with the active model.
#[deprecated(note = "pass the active model id to suppress_supergrok_paywalls")]
pub fn cursor_bypasses_access_gate_from_disk() -> bool {
    suppress_supergrok_paywalls(None)
}

/// True when Cursor credentials appear usable (`CURSOR_API_KEY` or `agent --list-models`).
pub fn cursor_authenticated() -> bool {
    if std::env::var_os("CURSOR_API_KEY").is_some_and(|v| !v.is_empty()) {
        return true;
    }
    cached_discovered_models().is_ok()
}

/// Run `agent login` (blocking). Used by `/login cursor`.
pub fn run_agent_login() -> Result<(), String> {
    xai_grok_sampler::cursor_cli::run_agent_login().map_err(|e| e.to_string())
}

/// Invalidate cached discovery so a post-login catalog refresh sees new models.
pub fn invalidate_discovery_cache() {
    DISCOVERY_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if let Ok(mut slot) = DISCOVERY_CACHE.lock() {
        *slot = None;
    }
}

static DISCOVERY_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISCOVERY_CACHE: std::sync::Mutex<Option<(u64, Result<Vec<DiscoveredCursorModel>, String>)>> =
    std::sync::Mutex::new(None);

/// Catalog key for a Cursor model id (`composer-2.5` → `cursor/composer-2.5`).
pub fn catalog_key(model_id: &str) -> String {
    format!("cursor/{model_id}")
}

/// Strip an optional `cursor/` prefix from a user-facing model id.
pub fn routing_slug(model_or_key: &str) -> &str {
    model_or_key
        .strip_prefix("cursor/")
        .unwrap_or(model_or_key)
}

fn cached_discovered_models() -> Result<Vec<DiscoveredCursorModel>, String> {
    let generation = DISCOVERY_GENERATION.load(std::sync::atomic::Ordering::SeqCst);
    if let Ok(guard) = DISCOVERY_CACHE.lock()
        && let Some((cached_gen, cached)) = guard.as_ref()
        && *cached_gen == generation
    {
        return cached.clone();
    }
    let fresh = list_models().map_err(|e| e.to_string());
    if let Ok(mut slot) = DISCOVERY_CACHE.lock() {
        *slot = Some((generation, fresh.clone()));
    }
    fresh
}

fn model_entry_for(discovered: &DiscoveredCursorModel) -> (String, ModelEntry) {
    let key = catalog_key(&discovered.id);
    let cw = NonZeroU64::new(DEFAULT_CURSOR_CONTEXT_WINDOW).expect("non-zero");
    let entry = ModelEntry {
        info: ModelInfo {
            id: Some(key.clone()),
            model: discovered.id.clone(),
            base_url: CURSOR_CLI_BASE_URL.to_owned(),
            name: Some(format!("Cursor: {}", discovered.name)),
            description: Some(
                "Routed through Cursor Agent CLI. Uses your Cursor subscription.".into(),
            ),
            max_completion_tokens: Some(32_768),
            temperature: None,
            top_p: None,
            api_backend: ApiBackend::CursorCli,
            auth_scheme: Default::default(),
            extra_headers: IndexMap::new(),
            query_params: IndexMap::new(),
            env_http_headers: IndexMap::new(),
            context_window: cw,
            auto_compact_threshold_percent: None,
            system_prompt_label: None,
            use_concise: false,
            agent_type: super::config::default_agent_type(),
            inference_idle_timeout_secs: Some(600),
            max_retries: Some(2),
            hidden: false,
            user_selectable: true,
            supported_in_api: true,
            reasoning_effort: None,
            supports_reasoning_effort: discovered.id.contains("thinking")
                || discovered.id.contains("-high")
                || discovered.id.contains("-xhigh")
                || discovered.id.contains("-medium")
                || discovered.id.contains("-low"),
            reasoning_efforts: Vec::new(),
            supports_backend_search: false,
            compactions_remaining: None,
            compaction_at_tokens: None,
            show_model_fingerprint: false,
            stream_tool_calls: Some(false),
            laziness_detector: Default::default(),
        },
        // Sentinel key so has_own_credentials() is true without Grok login.
        api_key: Some(CURSOR_CLI_API_KEY.to_owned()),
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    (key, entry)
}

fn bootstrap_cursor_entry() -> (String, ModelEntry) {
    model_entry_for(&DiscoveredCursorModel {
        id: BOOTSTRAP_CURSOR_MODEL_ID.to_owned(),
        name: "Auto".into(),
    })
}

/// Build catalog entries for discovered Cursor models.
///
/// When the CLI is present but `agent --list-models` fails (typically not
/// logged in), injects a bootstrap `cursor/auto` entry so `/login cursor` and
/// model selection remain discoverable for new users.
pub fn discover_cursor_model_entries() -> IndexMap<String, ModelEntry> {
    let mut out = IndexMap::new();
    match cached_discovered_models() {
        Ok(models) if !models.is_empty() => {
            for m in &models {
                let (key, entry) = model_entry_for(m);
                out.insert(key, entry);
            }
            tracing::info!(count = out.len(), "cursor_cli: injected Cursor models into catalog");
        }
        Ok(_) => {
            tracing::warn!("cursor_cli: list-models returned empty; injecting bootstrap cursor/auto");
            let (key, entry) = bootstrap_cursor_entry();
            out.insert(key, entry);
        }
        Err(err) => {
            if agent_available_cached() {
                tracing::warn!(
                    error = %err,
                    "cursor_cli: list-models failed (login required?); injecting bootstrap cursor/auto"
                );
                let (key, entry) = bootstrap_cursor_entry();
                out.insert(key, entry);
            } else {
                tracing::warn!(error = %err, "cursor_cli: failed to list models; skipping injection");
            }
        }
    }
    out
}

/// Fresh discovery bypassing the process cache (post `/login cursor`).
pub fn rediscover_cursor_model_entries() -> IndexMap<String, ModelEntry> {
    let mut out = IndexMap::new();
    match list_models() {
        Ok(models) if !models.is_empty() => {
            for m in &models {
                let (key, entry) = model_entry_for(m);
                out.insert(key, entry);
            }
        }
        Ok(_) | Err(_) => {
            if agent_available_cached() {
                let (key, entry) = bootstrap_cursor_entry();
                out.insert(key, entry);
            }
        }
    }
    out
}

/// Merge Cursor models into an existing catalog (does not overwrite keys).
pub fn merge_cursor_models(catalog: &mut IndexMap<String, ModelEntry>) {
    for (key, entry) in discover_cursor_model_entries() {
        catalog.entry(key).or_insert(entry);
    }
}

/// Merge a fresh discovery result (overwrites matching `cursor/` keys).
pub fn remesh_cursor_models(catalog: &mut IndexMap<String, ModelEntry>) {
    for (key, entry) in rediscover_cursor_model_entries() {
        catalog.insert(key, entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_key_and_slug() {
        assert_eq!(catalog_key("auto"), "cursor/auto");
        assert_eq!(routing_slug("cursor/auto"), "auto");
        assert_eq!(routing_slug("auto"), "auto");
    }

    #[test]
    fn cursor_billed_model_prefix() {
        assert!(is_cursor_billed_model("cursor/auto"));
        assert!(is_cursor_billed_model("cursor/composer-2.5"));
        assert!(!is_cursor_billed_model("grok-4"));
        assert!(!is_cursor_billed_model("composer-2.5"));
    }

    #[test]
    fn paywalls_not_suppressed_without_model_or_override() {
        // No explicit override in this process (tests must not rely on PATH).
        // suppress with None only true when override is set.
        let _ = cursor_paywall_override_from_disk(); // warm cache
        // Active Cursor model always suppresses regardless of override.
        assert!(suppress_supergrok_paywalls(Some("cursor/auto")));
        assert!(!suppress_supergrok_paywalls(Some("grok-4")));
    }

    #[test]
    fn model_info_meta_api_backend() {
        let mut meta = serde_json::Map::new();
        meta.insert(
            "apiBackend".into(),
            serde_json::Value::String("cursor_cli".into()),
        );
        assert!(is_cursor_billed_model_info("my-cursor-model", Some(&meta)));
        assert!(!is_cursor_billed_model_info("grok-4", None));
    }
}
