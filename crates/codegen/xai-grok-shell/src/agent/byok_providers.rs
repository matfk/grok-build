//! DeepSeek + OpenRouter BYOK: env-key auto-discovery into the model catalog.
//!
//! When `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY` are set (or forced via env/
//! `[features]`), inject `deepseek/<id>` and `openrouter/<id>` entries that
//! use chat-completions against those providers. Mirrors Cursor CLI injection
//! (`cursor_cli.rs`) but routes over HTTP — no new `ApiBackend`.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use indexmap::IndexMap;
use xai_grok_sampling_types::{ApiBackend, ReasoningEffort, ReasoningEffortOption};

use super::config::{Config, EnvKeys, ModelEntry, ModelInfo};

const DEEPSEEK_ENV_KEY: &str = "DEEPSEEK_API_KEY";
const OPENROUTER_ENV_KEY: &str = "OPENROUTER_API_KEY";

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Last resort when no provider `/models` response includes `context_length`
/// and the model has no hardcoded size.
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
/// DeepSeek V4 official context window (1M). DeepSeek's `/models` omits
/// `context_length`, so we hardcode this rather than guessing 128k.
const DEEPSEEK_V4_CONTEXT_WINDOW: u64 = 1_048_576;
const FETCH_TIMEOUT: Duration = Duration::from_secs(8);

/// Curated DeepSeek models (always injected when enabled).
/// Live `GET /models` currently lists V4 only. Context sizes: hardcoded V4
/// window below, else OpenRouter catalog enrichment, else default.
const DEEPSEEK_CURATED: &[(&str, &str)] = &[
    ("deepseek-v4-flash", "V4 Flash"),
    ("deepseek-v4-pro", "V4 Pro"),
];

/// Curated OpenRouter models (always injected when enabled).
const OPENROUTER_CURATED: &[(&str, &str)] = &[
    ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash"),
    ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro"),
    ("openai/gpt-4o", "GPT-4o"),
    ("anthropic/claude-sonnet-4", "Claude Sonnet 4"),
    ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("qwen/qwen3-coder", "Qwen3 Coder"),
    ("meta-llama/llama-4-maverick", "Llama 4 Maverick"),
];

fn resolve_context_window(discovered: Option<u64>) -> NonZeroU64 {
    let tokens = discovered
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);
    NonZeroU64::new(tokens)
        .unwrap_or_else(|| NonZeroU64::new(DEFAULT_CONTEXT_WINDOW).expect("non-zero"))
}

/// Hardcoded context for known DeepSeek ids (bare or `deepseek/<id>`).
/// Used when DeepSeek/OpenRouter omit `context_length`.
fn deepseek_hardcoded_context(model_id: &str) -> Option<u64> {
    let bare = model_id.strip_prefix("deepseek/").unwrap_or(model_id);
    match bare {
        "deepseek-v4-flash" | "deepseek-v4-pro" => Some(DEEPSEEK_V4_CONTEXT_WINDOW),
        _ => None,
    }
}

// ── Enablement ──────────────────────────────────────────────────────────────

fn bool_env_override(name: &str) -> Option<bool> {
    let Ok(raw) = std::env::var(name) else {
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

fn env_key_present(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| !v.trim().is_empty())
}

/// Resolve whether DeepSeek models should be injected.
///
/// Order: `GROK_DEEPSEEK` → `[features].deepseek` → auto (`DEEPSEEK_API_KEY`).
pub fn deepseek_enabled(cfg: &Config) -> bool {
    if let Some(v) = bool_env_override("GROK_DEEPSEEK") {
        return v;
    }
    match cfg.features.deepseek {
        Some(v) => v,
        None => env_key_present(DEEPSEEK_ENV_KEY),
    }
}

/// Resolve whether OpenRouter models should be injected.
///
/// Order: `GROK_OPENROUTER` → `[features].openrouter` → auto (`OPENROUTER_API_KEY`).
pub fn openrouter_enabled(cfg: &Config) -> bool {
    if let Some(v) = bool_env_override("GROK_OPENROUTER") {
        return v;
    }
    match cfg.features.openrouter {
        Some(v) => v,
        None => env_key_present(OPENROUTER_ENV_KEY),
    }
}

/// Whether to expand OpenRouter beyond the curated shortlist via `/models`.
///
/// Order: `GROK_OPENROUTER_FETCH_ALL` → `[features].openrouter_fetch_all` → false.
pub fn openrouter_fetch_all(cfg: &Config) -> bool {
    if let Some(v) = bool_env_override("GROK_OPENROUTER_FETCH_ALL") {
        return v;
    }
    cfg.features.openrouter_fetch_all.unwrap_or(false)
}

// ── Catalog keys ────────────────────────────────────────────────────────────

pub fn deepseek_catalog_key(model_id: &str) -> String {
    format!("deepseek/{model_id}")
}

pub fn openrouter_catalog_key(model_id: &str) -> String {
    format!("openrouter/{model_id}")
}

pub fn deepseek_routing_slug(model_or_key: &str) -> &str {
    model_or_key
        .strip_prefix("deepseek/")
        .unwrap_or(model_or_key)
}

pub fn openrouter_routing_slug(model_or_key: &str) -> &str {
    model_or_key
        .strip_prefix("openrouter/")
        .unwrap_or(model_or_key)
}

fn provider_display_name(brand: &str, discovered_name: &str) -> String {
    let trimmed = discovered_name.trim();
    if trimmed
        .to_ascii_lowercase()
        .starts_with(&brand.to_ascii_lowercase())
    {
        trimmed.to_owned()
    } else {
        format!("{brand}: {trimmed}")
    }
}

// ── Entry builders ──────────────────────────────────────────────────────────

fn base_entry(
    catalog_key: String,
    routing_id: &str,
    base_url: &str,
    name: String,
    description: &str,
    env_key: &str,
    extra_headers: IndexMap<String, String>,
    context_window: NonZeroU64,
) -> ModelEntry {
    ModelEntry {
        info: ModelInfo {
            id: Some(catalog_key),
            model: routing_id.to_owned(),
            base_url: base_url.to_owned(),
            name: Some(name),
            description: Some(description.into()),
            max_completion_tokens: Some(8192),
            temperature: None,
            top_p: None,
            api_backend: ApiBackend::ChatCompletions,
            auth_scheme: Default::default(),
            extra_headers,
            query_params: IndexMap::new(),
            env_http_headers: IndexMap::new(),
            context_window,
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
            supports_reasoning_effort: false,
            reasoning_efforts: Vec::new(),
            supports_backend_search: false,
            compactions_remaining: None,
            compaction_at_tokens: None,
            show_model_fingerprint: false,
            stream_tool_calls: Some(false),
            laziness_detector: Default::default(),
        },
        api_key: None,
        env_key: Some(EnvKeys::single(env_key)),
        auth_provider: None,
        api_base_url: None,
    }
}

/// DeepSeek `/effort` menu: none (thinking off) + low/high/max (thinking on).
fn deepseek_reasoning_efforts() -> Vec<ReasoningEffortOption> {
    vec![
        ReasoningEffortOption {
            id: "none".into(),
            value: ReasoningEffort::None,
            label: "None".into(),
            description: Some("Disable DeepSeek thinking mode".into()),
            default: false,
        },
        ReasoningEffortOption {
            id: "low".into(),
            value: ReasoningEffort::Low,
            label: "Low".into(),
            description: Some("Thinking on, low effort".into()),
            default: false,
        },
        ReasoningEffortOption {
            id: "high".into(),
            value: ReasoningEffort::High,
            label: "High".into(),
            description: Some("Thinking on, high effort (DeepSeek default)".into()),
            default: true,
        },
        ReasoningEffortOption {
            id: "max".into(),
            value: ReasoningEffort::Max,
            label: "Max".into(),
            description: Some("Thinking on, maximum effort".into()),
            default: false,
        },
    ]
}

fn deepseek_entry(
    model_id: &str,
    display: &str,
    discovered_context: Option<u64>,
) -> (String, ModelEntry) {
    let key = deepseek_catalog_key(model_id);
    let name = provider_display_name("DeepSeek", display);
    let mut entry = base_entry(
        key.clone(),
        model_id,
        DEEPSEEK_BASE_URL,
        name,
        "BYOK via DEEPSEEK_API_KEY. OpenAI-compatible chat completions. Use /effort for thinking.",
        DEEPSEEK_ENV_KEY,
        IndexMap::new(),
        resolve_context_window(
            discovered_context.or_else(|| deepseek_hardcoded_context(model_id)),
        ),
    );
    entry.info.supports_reasoning_effort = true;
    entry.info.reasoning_effort = Some(ReasoningEffort::High);
    entry.info.reasoning_efforts = deepseek_reasoning_efforts();
    (key, entry)
}

fn openrouter_extra_headers() -> IndexMap<String, String> {
    let mut headers = IndexMap::new();
    headers.insert(
        "HTTP-Referer".into(),
        "https://github.com/xai-org/grok-build".into(),
    );
    headers.insert("X-Title".into(), "Grok Build".into());
    headers
}

fn openrouter_entry(
    model_id: &str,
    display: &str,
    discovered_context: Option<u64>,
) -> (String, ModelEntry) {
    let key = openrouter_catalog_key(model_id);
    let name = provider_display_name("OpenRouter", display);
    let entry = base_entry(
        key.clone(),
        model_id,
        OPENROUTER_BASE_URL,
        name,
        "BYOK via OPENROUTER_API_KEY. OpenAI-compatible chat completions.",
        OPENROUTER_ENV_KEY,
        openrouter_extra_headers(),
        resolve_context_window(discovered_context),
    );
    (key, entry)
}

// ── HTTP /models fetch ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DiscoveredModel {
    id: String,
    name: String,
    /// From OpenRouter `context_length` (DeepSeek omits this field).
    context_length: Option<u64>,
}

#[derive(serde::Deserialize)]
struct ModelsListResponse {
    data: Vec<ModelsListItem>,
}

#[derive(serde::Deserialize)]
struct ModelsListItem {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u64>,
}

fn models_list_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    format!("{trimmed}/models")
}

/// Parse an OpenAI-compatible `{ "data": [ { "id", "name"? } ] }` body.
fn parse_models_list_json(body: &str) -> Result<Vec<DiscoveredModel>, String> {
    let parsed: ModelsListResponse =
        serde_json::from_str(body).map_err(|e| format!("invalid models list JSON: {e}"))?;
    let mut out = Vec::with_capacity(parsed.data.len());
    for item in parsed.data {
        let id = item.id.trim();
        if id.is_empty() {
            continue;
        }
        let name = item
            .name
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(id)
            .to_owned();
        out.push(DiscoveredModel {
            id: id.to_owned(),
            name,
            context_length: item.context_length.filter(|n| *n > 0),
        });
    }
    Ok(out)
}

fn fetch_models_list_blocking(
    base_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<DiscoveredModel>, String> {
    let url = models_list_url(base_url);
    let client = reqwest::blocking::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut req = client.get(&url).header("Accept", "application/json");
    if let Some(key) = api_key.map(str::trim).filter(|k| !k.is_empty()) {
        req = req.bearer_auth(key);
    }
    let resp = req.send().map_err(|e| format!("GET {url}: {e}"))?;
    let status = resp.status();
    let body = resp
        .text()
        .map_err(|e| format!("read body from {url}: {e}"))?;
    if !status.is_success() {
        let snippet: String = body.chars().take(200).collect();
        return Err(format!("GET {url} → {status}: {snippet}"));
    }
    parse_models_list_json(&body)
}

/// `resolve_model_catalog` can run on a Tokio worker. `reqwest::blocking` builds
/// its own runtime; dropping that runtime on a Tokio worker panics ("Cannot drop
/// a runtime in a context where blocking is not allowed"). Run the fetch on a
/// dedicated OS thread whenever a Tokio runtime is already active.
fn fetch_models_list(
    base_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<DiscoveredModel>, String> {
    if tokio::runtime::Handle::try_current().is_ok() {
        let base_url = base_url.to_owned();
        let api_key = api_key.map(str::to_owned);
        return std::thread::Builder::new()
            .name("byok-models-fetch".into())
            .spawn(move || fetch_models_list_blocking(&base_url, api_key.as_deref()))
            .map_err(|e| format!("spawn byok fetch thread: {e}"))?
            .join()
            .map_err(|_| "byok fetch thread panicked".to_owned())?;
    }
    fetch_models_list_blocking(base_url, api_key)
}

// ── Discovery cache ─────────────────────────────────────────────────────────

// Successful `/models` fetches only, keyed by (generation, env value).
// Errors are never cached: a transient startup failure must not pin the
// curated shortlist for the process lifetime, and a changed `*_API_KEY` env
// value invalidates the entry naturally (no production caller bumps the
// generation counter).

static DEEPSEEK_FETCH_GEN: AtomicU64 = AtomicU64::new(0);
static DEEPSEEK_FETCH_CACHE: Mutex<Option<(u64, String, Vec<DiscoveredModel>)>> =
    Mutex::new(None);

static OPENROUTER_FETCH_GEN: AtomicU64 = AtomicU64::new(0);
static OPENROUTER_FETCH_CACHE: Mutex<Option<(u64, String, Vec<DiscoveredModel>)>> =
    Mutex::new(None);

/// Public OpenRouter `/models` used only for `context_length` enrichment
/// (DeepSeek's list omits sizes; curated OpenRouter entries need sizes too).
/// The `Option<String>` cache key is `None` for unauthenticated fetches.
static OPENROUTER_CTX_GEN: AtomicU64 = AtomicU64::new(0);
static OPENROUTER_CTX_CACHE: Mutex<Option<(u64, Option<String>, IndexMap<String, u64>)>> =
    Mutex::new(None);

/// Invalidate cached `/models` fetches (tests / forced refresh). Key rotation
/// self-invalidates via the env-value cache key, so this is only needed when
/// callers want a fresh fetch for the same env value.
pub fn invalidate_byok_discovery_cache() {
    DEEPSEEK_FETCH_GEN.fetch_add(1, Ordering::SeqCst);
    OPENROUTER_FETCH_GEN.fetch_add(1, Ordering::SeqCst);
    OPENROUTER_CTX_GEN.fetch_add(1, Ordering::SeqCst);
    if let Ok(mut slot) = DEEPSEEK_FETCH_CACHE.lock() {
        *slot = None;
    }
    if let Ok(mut slot) = OPENROUTER_FETCH_CACHE.lock() {
        *slot = None;
    }
    if let Ok(mut slot) = OPENROUTER_CTX_CACHE.lock() {
        *slot = None;
    }
}

/// Fetch `/models`, caching only successful results.
///
/// The cache entry is keyed by (generation, env value): a mid-process key
/// rotation re-fetches automatically, and an `Err` is never stored so the
/// next catalog build retries a transiently-failed fetch.
fn cached_fetch(
    generation_counter: &AtomicU64,
    cache: &Mutex<Option<(u64, String, Vec<DiscoveredModel>)>>,
    base_url: &str,
    env_key: &str,
) -> Result<Vec<DiscoveredModel>, String> {
    let generation = generation_counter.load(Ordering::SeqCst);
    let key = std::env::var(env_key).map_err(|_| format!("{env_key} unset"))?;
    if key.trim().is_empty() {
        return Err(format!("{env_key} empty"));
    }
    if let Ok(guard) = cache.lock()
        && let Some((cached_gen, cached_key, cached)) = guard.as_ref()
        && *cached_gen == generation
        && *cached_key == key
    {
        return Ok(cached.clone());
    }
    let fresh = fetch_models_list(base_url, Some(&key))?;
    if let Ok(mut slot) = cache.lock() {
        *slot = Some((generation, key, fresh.clone()));
    }
    Ok(fresh)
}

/// OpenRouter model id → `context_length` from the public catalog.
///
/// Prefer `OPENROUTER_API_KEY` when set; otherwise fetch unauthenticated
/// (OpenRouter's model list is public). Failures return an empty map so
/// callers can fall back to [`DEFAULT_CONTEXT_WINDOW`]; the failure is not
/// cached, so the next catalog build retries.
fn openrouter_context_index() -> IndexMap<String, u64> {
    let generation = OPENROUTER_CTX_GEN.load(Ordering::SeqCst);
    let key = std::env::var(OPENROUTER_ENV_KEY)
        .ok()
        .filter(|v| !v.trim().is_empty());
    if let Ok(guard) = OPENROUTER_CTX_CACHE.lock()
        && let Some((cached_gen, cached_key, cached)) = guard.as_ref()
        && *cached_gen == generation
        && cached_key.as_deref() == key.as_deref()
    {
        return cached.clone();
    }

    let fresh = fetch_models_list(OPENROUTER_BASE_URL, key.as_deref()).map(|models| {
        let mut map = IndexMap::new();
        for m in models {
            if let Some(cw) = m.context_length {
                map.insert(m.id, cw);
            }
        }
        map
    });

    let index = match fresh {
        Ok(index) => index,
        Err(err) => {
            tracing::warn!(
                error = %err,
                "byok: OpenRouter context_length catalog fetch failed; using default context windows"
            );
            return IndexMap::new();
        }
    };

    if let Ok(mut slot) = OPENROUTER_CTX_CACHE.lock() {
        *slot = Some((generation, key, index.clone()));
    }
    index
}

/// Look up `context_length` for a DeepSeek or OpenRouter routing id.
fn context_length_for(routing_id: &str, index: &IndexMap<String, u64>) -> Option<u64> {
    if let Some(cw) = index.get(routing_id) {
        return Some(*cw);
    }
    // DeepSeek bare ids → OpenRouter `deepseek/<id>`.
    if !routing_id.contains('/') {
        let prefixed = format!("deepseek/{routing_id}");
        if let Some(cw) = index.get(&prefixed) {
            return Some(*cw);
        }
    }
    // OpenRouter `deepseek/<id>` → try bare id (unlikely in OR index).
    if let Some(bare) = routing_id.strip_prefix("deepseek/")
        && let Some(cw) = index.get(bare)
    {
        return Some(*cw);
    }
    deepseek_hardcoded_context(routing_id)
}

/// Insert or refresh: keep existing entries, but upgrade `context_window` when
/// the API reports a real size and the existing value is still the default.
fn upsert_byok_entry(out: &mut IndexMap<String, ModelEntry>, key: String, entry: ModelEntry) {
    match out.entry(key) {
        indexmap::map::Entry::Vacant(v) => {
            v.insert(entry);
        }
        indexmap::map::Entry::Occupied(mut o) => {
            let existing = o.get().info.context_window.get();
            let incoming = entry.info.context_window.get();
            if existing == DEFAULT_CONTEXT_WINDOW && incoming != DEFAULT_CONTEXT_WINDOW {
                o.get_mut().info.context_window = entry.info.context_window;
            }
        }
    }
}

// ── Discover + merge ────────────────────────────────────────────────────────

fn insert_curated(
    out: &mut IndexMap<String, ModelEntry>,
    curated: &[(&str, &str)],
    build: fn(&str, &str, Option<u64>) -> (String, ModelEntry),
    context_index: &IndexMap<String, u64>,
) {
    for &(id, name) in curated {
        let cw = context_length_for(id, context_index);
        let (key, entry) = build(id, name, cw);
        upsert_byok_entry(out, key, entry);
    }
}

/// Build DeepSeek catalog entries (curated + live `/models` when reachable).
///
/// DeepSeek's `/models` omits `context_length`. V4 uses a hardcoded 1M window;
/// other ids may still be enriched from OpenRouter's public catalog.
pub fn discover_deepseek_model_entries() -> IndexMap<String, ModelEntry> {
    let mut out = IndexMap::new();
    let context_index = openrouter_context_index();
    insert_curated(
        &mut out,
        DEEPSEEK_CURATED,
        deepseek_entry,
        &context_index,
    );

    match cached_fetch(
        &DEEPSEEK_FETCH_GEN,
        &DEEPSEEK_FETCH_CACHE,
        DEEPSEEK_BASE_URL,
        DEEPSEEK_ENV_KEY,
    ) {
        Ok(models) => {
            for m in models {
                let cw = m
                    .context_length
                    .or_else(|| context_length_for(&m.id, &context_index));
                let (key, entry) = deepseek_entry(&m.id, &m.name, cw);
                upsert_byok_entry(&mut out, key, entry);
            }
            tracing::info!(
                count = out.len(),
                "byok: injected DeepSeek models into catalog"
            );
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                curated = out.len(),
                "byok: DeepSeek /models fetch failed; keeping curated shortlist"
            );
        }
    }
    out
}

/// Build OpenRouter catalog entries (curated; full `/models` when opted in).
/// Context windows come from OpenRouter `context_length` (fetched even for the
/// curated shortlist).
pub fn discover_openrouter_model_entries(fetch_all: bool) -> IndexMap<String, ModelEntry> {
    let mut out = IndexMap::new();
    let context_index = openrouter_context_index();
    insert_curated(
        &mut out,
        OPENROUTER_CURATED,
        openrouter_entry,
        &context_index,
    );

    if !fetch_all {
        tracing::info!(
            count = out.len(),
            "byok: injected curated OpenRouter models (set GROK_OPENROUTER_FETCH_ALL=1 for full list)"
        );
        return out;
    }

    match cached_fetch(
        &OPENROUTER_FETCH_GEN,
        &OPENROUTER_FETCH_CACHE,
        OPENROUTER_BASE_URL,
        OPENROUTER_ENV_KEY,
    ) {
        Ok(models) => {
            for m in models {
                let cw = m
                    .context_length
                    .or_else(|| context_length_for(&m.id, &context_index));
                let (key, entry) = openrouter_entry(&m.id, &m.name, cw);
                upsert_byok_entry(&mut out, key, entry);
            }
            tracing::info!(
                count = out.len(),
                "byok: injected OpenRouter models (full fetch) into catalog"
            );
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                curated = out.len(),
                "byok: OpenRouter /models fetch failed; keeping curated shortlist"
            );
        }
    }
    out
}

/// Merge DeepSeek models into an existing catalog (does not overwrite keys).
pub fn merge_deepseek_models(catalog: &mut IndexMap<String, ModelEntry>) {
    for (key, entry) in discover_deepseek_model_entries() {
        catalog.entry(key).or_insert(entry);
    }
}

/// Merge OpenRouter models into an existing catalog (does not overwrite keys).
pub fn merge_openrouter_models(catalog: &mut IndexMap<String, ModelEntry>, fetch_all: bool) {
    for (key, entry) in discover_openrouter_model_entries(fetch_all) {
        catalog.entry(key).or_insert(entry);
    }
}

/// Test helper: curated-only discovery without network (hardcoded V4 windows).
#[cfg(test)]
pub(crate) fn curated_deepseek_only() -> IndexMap<String, ModelEntry> {
    let mut out = IndexMap::new();
    insert_curated(
        &mut out,
        DEEPSEEK_CURATED,
        deepseek_entry,
        &IndexMap::new(),
    );
    out
}

/// Test helper: curated-only discovery without network (hardcoded V4 windows).
#[cfg(test)]
pub(crate) fn curated_openrouter_only() -> IndexMap<String, ModelEntry> {
    let mut out = IndexMap::new();
    insert_curated(
        &mut out,
        OPENROUTER_CURATED,
        openrouter_entry,
        &IndexMap::new(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::config::Features;

    fn empty_cfg() -> Config {
        Config {
            features: Features::default(),
            ..Config::default()
        }
    }

    #[test]
    fn catalog_keys_and_slugs() {
        assert_eq!(deepseek_catalog_key("deepseek-chat"), "deepseek/deepseek-chat");
        assert_eq!(
            deepseek_routing_slug("deepseek/deepseek-chat"),
            "deepseek-chat"
        );
        assert_eq!(deepseek_routing_slug("deepseek-chat"), "deepseek-chat");
        assert_eq!(
            openrouter_catalog_key("openai/gpt-4o"),
            "openrouter/openai/gpt-4o"
        );
        assert_eq!(
            openrouter_routing_slug("openrouter/openai/gpt-4o"),
            "openai/gpt-4o"
        );
    }

    #[test]
    fn display_name_avoids_double_brand() {
        assert_eq!(
            provider_display_name("DeepSeek", "Chat"),
            "DeepSeek: Chat"
        );
        assert_eq!(
            provider_display_name("DeepSeek", "DeepSeek Chat"),
            "DeepSeek Chat"
        );
        assert_eq!(
            provider_display_name("OpenRouter", "OpenRouter: GPT-4o"),
            "OpenRouter: GPT-4o"
        );
    }

    #[test]
    fn curated_deepseek_entries_use_env_key() {
        let entries = curated_deepseek_only();
        assert!(entries.contains_key("deepseek/deepseek-v4-flash"));
        assert!(entries.contains_key("deepseek/deepseek-v4-pro"));
        let flash = entries.get("deepseek/deepseek-v4-flash").unwrap();
        assert_eq!(flash.info.model, "deepseek-v4-flash");
        assert_eq!(flash.info.base_url, DEEPSEEK_BASE_URL);
        assert_eq!(flash.info.api_backend, ApiBackend::ChatCompletions);
        // Offline curated V4 uses hardcoded 1M (DeepSeek /models omits sizes).
        assert_eq!(flash.info.context_window.get(), DEEPSEEK_V4_CONTEXT_WINDOW);
        assert!(flash.api_key.is_none());
        assert_eq!(
            flash.env_key.as_ref().and_then(EnvKeys::primary),
            Some(DEEPSEEK_ENV_KEY)
        );
        assert_eq!(flash.info.stream_tool_calls, Some(false));
    }

    #[test]
    fn curated_deepseek_entries_expose_effort_menu() {
        let entries = curated_deepseek_only();
        let flash = entries.get("deepseek/deepseek-v4-flash").unwrap();
        assert!(flash.info.supports_reasoning_effort);
        assert_eq!(flash.info.reasoning_effort, Some(ReasoningEffort::High));
        let ids: Vec<_> = flash
            .info
            .reasoning_efforts
            .iter()
            .map(|o| o.id.as_str())
            .collect();
        assert_eq!(ids, ["none", "low", "high", "max"]);
        let default = flash
            .info
            .reasoning_efforts
            .iter()
            .find(|o| o.default)
            .expect("default effort");
        assert_eq!(default.value, ReasoningEffort::High);

        let pro = entries.get("deepseek/deepseek-v4-pro").unwrap();
        assert!(pro.info.supports_reasoning_effort);
        assert_eq!(pro.info.reasoning_efforts.len(), 4);
        assert_eq!(pro.info.context_window.get(), DEEPSEEK_V4_CONTEXT_WINDOW);
    }

    #[test]
    fn curated_openrouter_entries_do_not_claim_effort() {
        let entries = curated_openrouter_only();
        let gpt = entries.get("openrouter/openai/gpt-4o").unwrap();
        assert!(!gpt.info.supports_reasoning_effort);
        assert!(gpt.info.reasoning_efforts.is_empty());
    }

    #[test]
    fn curated_openrouter_entries_use_env_key_and_headers() {
        let entries = curated_openrouter_only();
        assert!(entries.contains_key("openrouter/openai/gpt-4o"));
        let gpt = entries.get("openrouter/openai/gpt-4o").unwrap();
        assert_eq!(gpt.info.model, "openai/gpt-4o");
        assert_eq!(gpt.info.base_url, OPENROUTER_BASE_URL);
        assert_eq!(
            gpt.env_key.as_ref().and_then(EnvKeys::primary),
            Some(OPENROUTER_ENV_KEY)
        );
        assert_eq!(
            gpt.info.extra_headers.get("X-Title").map(String::as_str),
            Some("Grok Build")
        );
        assert!(gpt.info.extra_headers.contains_key("HTTP-Referer"));
    }

    #[test]
    fn merge_does_not_overwrite_existing_keys() {
        let mut catalog = IndexMap::new();
        let (key, mut custom) = deepseek_entry("deepseek-v4-flash", "Custom", None);
        custom.info.name = Some("User Override".into());
        catalog.insert(key.clone(), custom);

        for (k, entry) in curated_deepseek_only() {
            catalog.entry(k).or_insert(entry);
        }

        assert_eq!(
            catalog.get(&key).unwrap().info.name.as_deref(),
            Some("User Override")
        );
        assert!(catalog.contains_key("deepseek/deepseek-v4-pro"));
    }

    #[test]
    fn parse_models_list_fixture() {
        let body = r#"{
            "object": "list",
            "data": [
                {"id": "deepseek-v4-flash", "object": "model"},
                {"id": "deepseek-v4-pro", "name": "V4 Pro", "context_length": 1048576},
                {"id": "  ", "name": "skip"},
                {"id": "extra-model"}
            ]
        }"#;
        let models = parse_models_list_json(body).unwrap();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "deepseek-v4-flash");
        assert_eq!(models[0].name, "deepseek-v4-flash");
        assert_eq!(models[0].context_length, None);
        assert_eq!(models[1].id, "deepseek-v4-pro");
        assert_eq!(models[1].name, "V4 Pro");
        assert_eq!(models[1].context_length, Some(1_048_576));
        assert_eq!(models[2].id, "extra-model");
    }

    #[test]
    fn context_window_uses_discovered_else_hardcoded_or_default() {
        let (_, flash) = deepseek_entry("deepseek-v4-flash", "V4 Flash", None);
        assert_eq!(flash.info.context_window.get(), DEEPSEEK_V4_CONTEXT_WINDOW);

        let (_, from_api) =
            deepseek_entry("deepseek-v4-flash", "V4 Flash", Some(1_048_576));
        assert_eq!(from_api.info.context_window.get(), 1_048_576);

        // Unknown DeepSeek id with no discovery → generic default.
        let (_, unknown) = deepseek_entry("deepseek-unknown", "Unknown", None);
        assert_eq!(unknown.info.context_window.get(), DEFAULT_CONTEXT_WINDOW);

        let (_, override_cw) =
            openrouter_entry("deepseek/deepseek-v4-flash", "V4 Flash", Some(999_999));
        assert_eq!(override_cw.info.context_window.get(), 999_999);
    }

    #[test]
    fn context_length_for_matches_openrouter_ids() {
        let mut index = IndexMap::new();
        index.insert("deepseek/deepseek-v4-flash".into(), 1_048_576);
        index.insert("openai/gpt-4o".into(), 128_000);
        assert_eq!(
            context_length_for("deepseek-v4-flash", &index),
            Some(1_048_576)
        );
        assert_eq!(
            context_length_for("deepseek/deepseek-v4-flash", &index),
            Some(1_048_576)
        );
        assert_eq!(context_length_for("openai/gpt-4o", &index), Some(128_000));
        assert_eq!(context_length_for("missing", &index), None);
    }

    #[test]
    fn upsert_upgrades_default_context_from_api() {
        let mut out = IndexMap::new();
        // Unknown model starts at generic default; upsert upgrades from API.
        let (key, fallback) = deepseek_entry("deepseek-unknown", "Unknown", None);
        upsert_byok_entry(&mut out, key.clone(), fallback);
        assert_eq!(out[&key].info.context_window.get(), DEFAULT_CONTEXT_WINDOW);

        let (_, enriched) = deepseek_entry("deepseek-unknown", "Unknown", Some(1_048_576));
        upsert_byok_entry(&mut out, key.clone(), enriched);
        assert_eq!(out[&key].info.context_window.get(), 1_048_576);

        // Do not downgrade a real size back to default.
        let (_, again) = deepseek_entry("deepseek-unknown", "Unknown", None);
        upsert_byok_entry(&mut out, key.clone(), again);
        assert_eq!(out[&key].info.context_window.get(), 1_048_576);
    }

    #[test]
    fn context_length_for_falls_back_to_hardcoded_deepseek_v4() {
        let empty = IndexMap::new();
        assert_eq!(
            context_length_for("deepseek-v4-pro", &empty),
            Some(DEEPSEEK_V4_CONTEXT_WINDOW)
        );
        assert_eq!(
            context_length_for("deepseek/deepseek-v4-flash", &empty),
            Some(DEEPSEEK_V4_CONTEXT_WINDOW)
        );
    }

    #[test]
    #[serial_test::serial]
    fn enablement_respects_env_override() {
        let _ds = EnvGuard::set("GROK_DEEPSEEK", "0");
        let _or = EnvGuard::set("GROK_OPENROUTER", "1");
        let mut cfg = empty_cfg();
        // Even with keys "present" via feature, env override wins.
        cfg.features.deepseek = Some(true);
        cfg.features.openrouter = Some(false);
        assert!(!deepseek_enabled(&cfg));
        assert!(openrouter_enabled(&cfg));
    }

    #[test]
    #[serial_test::serial]
    fn enablement_feature_beats_auto_absent_key() {
        let _ds = EnvGuard::remove("GROK_DEEPSEEK");
        let _key = EnvGuard::remove(DEEPSEEK_ENV_KEY);
        let mut cfg = empty_cfg();
        assert!(!deepseek_enabled(&cfg));
        cfg.features.deepseek = Some(true);
        assert!(deepseek_enabled(&cfg));
    }

    #[test]
    #[serial_test::serial]
    fn openrouter_fetch_all_defaults_false() {
        let _e = EnvGuard::remove("GROK_OPENROUTER_FETCH_ALL");
        let cfg = empty_cfg();
        assert!(!openrouter_fetch_all(&cfg));
        let mut cfg = empty_cfg();
        cfg.features.openrouter_fetch_all = Some(true);
        assert!(openrouter_fetch_all(&cfg));
    }

    #[test]
    #[serial_test::serial]
    fn injected_deepseek_is_byok_in_auth_facts() {
        // Regression: auth-fact lookup used resolve_model_list (no injection),
        // so DeepSeek looked NotByok and reconstruct_full_config attached the
        // session JWT → DeepSeek 401 retry loop.
        let _force = EnvGuard::set("GROK_DEEPSEEK", "1");
        let _key = EnvGuard::set(DEEPSEEK_ENV_KEY, "sk-test-deepseek-facts");
        let facts = crate::agent::config::resolve_model_auth_facts_and_provider(
            "deepseek/deepseek-v4-flash",
        );
        assert_eq!(
            facts.0.byok,
            crate::agent::auth_method::ModelByok::Byok,
            "injected deepseek/deepseek-v4-flash must classify as BYOK"
        );
        let by_slug =
            crate::agent::config::resolve_model_auth_facts_and_provider("deepseek-v4-flash");
        assert_eq!(
            by_slug.0.byok,
            crate::agent::auth_method::ModelByok::Byok,
            "routing slug deepseek-v4-flash must also resolve to the injected entry"
        );
        assert!(!crate::agent::auth_method::session_token_auth_gate(
            true,
            facts.0.byok,
            false, // third-party host
        ));
    }

    #[test]
    #[serial_test::serial]
    fn has_own_credentials_follows_env_key() {
        let entries = curated_deepseek_only();
        let flash = entries.get("deepseek/deepseek-v4-flash").unwrap().clone();
        let _clear = EnvGuard::remove(DEEPSEEK_ENV_KEY);
        assert!(
            !flash.has_own_credentials(),
            "without DEEPSEEK_API_KEY, env_key must not count as own creds"
        );
        let _set = EnvGuard::set(DEEPSEEK_ENV_KEY, "sk-test-deepseek");
        assert!(flash.has_own_credentials());
        let creds = crate::agent::config::resolve_credentials(&flash, Some("session-jwt"));
        assert_eq!(creds.api_key.as_deref(), Some("sk-test-deepseek"));
        assert_eq!(creds.base_url, DEEPSEEK_BASE_URL);
    }

    /// RAII env var set/restore for unit tests.
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
        fn remove(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }
}
