use std::cell::Cell;

use frankensearch_core::SearchError;

use super::DetectOptions;
use super::discovery::{
    OFFLINE_ENV, RemoteIntentEnv, resolve_remote_intent, resolve_remote_intent_with,
};

fn expect_unverifiable(env: &RemoteIntentEnv, context: &str) {
    match resolve_remote_intent(DetectOptions::default(), env) {
        Err(SearchError::UnverifiableRemoteSpace { .. }) => {}
        Err(other) => panic!("{context}: expected UnverifiableRemoteSpace, got {other:?}"),
        Ok((_, Some(_))) => {
            panic!("{context}: expected typed failure, got a verified embedder")
        }
        Ok((_, None)) => panic!("{context}: explicit intent silently degraded to None"),
    }
}

#[test]
fn no_intent_and_no_keys_is_none() {
    let (offline, outcome) =
        resolve_remote_intent(DetectOptions::default(), &RemoteIntentEnv::default())
            .expect("no intent is not an error");
    assert!(!offline);
    assert!(outcome.is_none());
}

#[test]
fn ambient_provider_key_without_intent_stays_none() {
    let env = RemoteIntentEnv {
        openai_key: Some("sk-ambient-unrelated-tool".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let (offline, outcome) =
        resolve_remote_intent(DetectOptions::default(), &env).expect("ambient key is not intent");
    assert!(!offline);
    assert!(outcome.is_none());
}

#[test]
fn explicit_offline_rejects_remote_intent_before_construction() {
    let env = RemoteIntentEnv {
        offline: Some("false".to_owned()),
        provider: Some("openai".to_owned()),
        identity_json: Some("{malformed identity".to_owned()),
        openai_key: Some("sk-test".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let construction_calls = Cell::new(0);

    let outcome = resolve_remote_intent_with(
        DetectOptions {
            offline: Some(true),
        },
        &env,
        |_| {
            construction_calls.set(construction_calls.get() + 1);
            Ok(Some(()))
        },
    );

    assert!(matches!(
        outcome,
        Err(SearchError::EmbedderUnavailable { .. })
    ));
    assert_eq!(construction_calls.get(), 0);
}

#[test]
fn explicit_offline_ignores_ambient_credentials_without_construction() {
    let env = RemoteIntentEnv {
        openai_key: Some("sk-ambient-unrelated-tool".to_owned()),
        gemini_key: Some("ambient-gemini-key".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let construction_calls = Cell::new(0);

    let (offline, outcome) = resolve_remote_intent_with(
        DetectOptions {
            offline: Some(true),
        },
        &env,
        |_| {
            construction_calls.set(construction_calls.get() + 1);
            Ok(Some(()))
        },
    )
    .expect("ambient credentials are not explicit remote intent");

    assert!(offline);
    assert!(outcome.is_none());
    assert_eq!(construction_calls.get(), 0);
}

#[test]
fn explicit_online_overrides_offline_environment_and_constructs_once() {
    let env = RemoteIntentEnv {
        offline: Some("true".to_owned()),
        provider: Some("openai".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let construction_calls = Cell::new(0);

    let (offline, outcome) = resolve_remote_intent_with(
        DetectOptions {
            offline: Some(false),
        },
        &env,
        |_| {
            construction_calls.set(construction_calls.get() + 1);
            Ok(Some(41_u8))
        },
    )
    .expect("explicit online policy must override the environment");

    assert!(!offline);
    assert_eq!(outcome, Some(41));
    assert_eq!(construction_calls.get(), 1);
}

#[test]
fn explicit_offline_does_not_parse_malformed_environment() {
    let env = RemoteIntentEnv {
        offline: Some("definitely".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let construction_calls = Cell::new(0);

    let (offline, outcome) = resolve_remote_intent_with(
        DetectOptions {
            offline: Some(true),
        },
        &env,
        |_| {
            construction_calls.set(construction_calls.get() + 1);
            Ok(Some(()))
        },
    )
    .expect("explicit offline policy must dominate malformed ambient policy");

    assert!(offline);
    assert!(outcome.is_none());
    assert_eq!(construction_calls.get(), 0);
}

#[test]
fn explicit_online_does_not_parse_malformed_environment() {
    let env = RemoteIntentEnv {
        offline: Some("definitely".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let construction_calls = Cell::new(0);

    let (offline, outcome) = resolve_remote_intent_with(
        DetectOptions {
            offline: Some(false),
        },
        &env,
        |_| {
            construction_calls.set(construction_calls.get() + 1);
            Ok(Some(19_u8))
        },
    )
    .expect("explicit online policy must dominate malformed ambient policy");

    assert!(!offline);
    assert_eq!(outcome, Some(19));
    assert_eq!(construction_calls.get(), 1);
}

#[test]
fn deferred_malformed_offline_environment_fails_before_construction() {
    let env = RemoteIntentEnv {
        offline: Some("definitely".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let construction_calls = Cell::new(0);

    let outcome = resolve_remote_intent_with(DetectOptions::default(), &env, |_| {
        construction_calls.set(construction_calls.get() + 1);
        Ok(Some(()))
    });

    assert!(
        matches!(&outcome, Err(SearchError::InvalidConfig { .. })),
        "malformed ambient offline policy must fail as InvalidConfig"
    );
    let Err(SearchError::InvalidConfig {
        field,
        value,
        reason,
    }) = outcome
    else {
        return;
    };
    assert_eq!(field, OFFLINE_ENV);
    assert_eq!(value, "definitely");
    assert!(reason.contains("boolean flag"));
    assert_eq!(construction_calls.get(), 0);
}

#[test]
fn deferred_policy_honors_offline_environment_without_construction() {
    let env = RemoteIntentEnv {
        offline: Some("yes".to_owned()),
        provider: Some("gemini".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let construction_calls = Cell::new(0);

    let outcome = resolve_remote_intent_with(DetectOptions::default(), &env, |_| {
        construction_calls.set(construction_calls.get() + 1);
        Ok(Some(()))
    });

    assert!(matches!(
        outcome,
        Err(SearchError::EmbedderUnavailable { .. })
    ));
    assert_eq!(construction_calls.get(), 0);
}

#[test]
fn deferred_policy_honors_online_environment_and_constructs_once() {
    let env = RemoteIntentEnv {
        offline: Some("off".to_owned()),
        provider: Some("gemini".to_owned()),
        ..RemoteIntentEnv::default()
    };
    let construction_calls = Cell::new(0);

    let (offline, outcome) = resolve_remote_intent_with(DetectOptions::default(), &env, |_| {
        construction_calls.set(construction_calls.get() + 1);
        Ok(Some(73_u8))
    })
    .expect("deferred policy must honor an online environment");

    assert!(!offline);
    assert_eq!(outcome, Some(73));
    assert_eq!(construction_calls.get(), 1);
}

#[test]
fn explicit_provider_without_identity_fails_closed() {
    let env = RemoteIntentEnv {
        provider: Some("openai".to_owned()),
        openai_key: Some("sk-test".to_owned()),
        ..RemoteIntentEnv::default()
    };
    expect_unverifiable(&env, "provider without identity");
}

#[test]
fn explicit_model_alone_is_intent_and_fails_closed_without_identity() {
    let env = RemoteIntentEnv {
        model: Some("text-embedding-3-small".to_owned()),
        ..RemoteIntentEnv::default()
    };
    expect_unverifiable(&env, "model without identity");
}

#[cfg(feature = "api")]
#[test]
fn malformed_identity_fails_closed() {
    let env = RemoteIntentEnv {
        identity_json: Some("{not json".to_owned()),
        openai_key: Some("sk-test".to_owned()),
        ..RemoteIntentEnv::default()
    };
    expect_unverifiable(&env, "malformed identity");
}

#[cfg(feature = "api")]
#[test]
fn structurally_empty_identity_fails_closed() {
    let env = RemoteIntentEnv {
        identity_json: Some("{}".to_owned()),
        ..RemoteIntentEnv::default()
    };
    expect_unverifiable(&env, "empty identity object");
}
