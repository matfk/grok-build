//! Subscription-tier classification shared across the shell and the pager.
//!
//! The subscription tier reaches the client as a free-form **display-name
//! string** (from CCP `/settings` `subscription_tier_display`, or the numeric
//! JWT `tier` claim mapped to a display-style string by
//! [`crate::agent::mvp_agent::jwt_tier_claim`]). There is no shared enum, so
//! gating decisions classify the string here in ONE place so the pager's
//! cosmetic slash-command gate and the shell's capability (toolset) gate can't
//! drift apart.
//!
//! **Grok Build policy:** client-side free / X Basic feature gates are lifted.
//! `/usage`, Imagine, voice, and related slash/tool cosmetics stay available
//! without SuperGrok or Cursor. Server-side quotas (if any) still apply.

/// Historical free / X Basic classifier. Always returns `false` so pager and
/// shell never withhold features on tier name alone.
///
/// Callers may still special-case an *absent* tier (`None`); prefer treating
/// absence as unrestricted to match this policy.
pub fn is_restricted_tier_name(_tier: &str) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_and_x_basic_are_not_client_restricted() {
        assert!(!is_restricted_tier_name(""));
        assert!(!is_restricted_tier_name("   "));
        assert!(!is_restricted_tier_name("Free"));
        assert!(!is_restricted_tier_name("free"));
        assert!(!is_restricted_tier_name("X Basic"));
        assert!(!is_restricted_tier_name("x_basic"));
        assert!(!is_restricted_tier_name("  X BASIC  "));
    }

    #[test]
    fn paid_and_unknown_names_remain_unrestricted() {
        assert!(!is_restricted_tier_name("SuperGrok"));
        assert!(!is_restricted_tier_name("SuperGrok Heavy"));
        assert!(!is_restricted_tier_name("supergrok_lite"));
        assert!(!is_restricted_tier_name("X Premium"));
        assert!(!is_restricted_tier_name("x_premium_plus"));
        assert!(!is_restricted_tier_name("api_key"));
        assert!(!is_restricted_tier_name("API Key"));
        assert!(!is_restricted_tier_name("some_new_plan"));
    }
}
