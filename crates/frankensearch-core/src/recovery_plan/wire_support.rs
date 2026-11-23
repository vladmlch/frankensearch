use serde::Deserialize;

pub(super) mod recovery_model_tier_wire {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::traits::ModelTier;

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum WireModelTier {
        Fast,
        Quality,
    }

    // Serde's `with` module contract passes the field by reference even
    // though ModelTier is Copy.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S>(tier: &ModelTier, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match tier {
            ModelTier::Fast => WireModelTier::Fast,
            ModelTier::Quality => WireModelTier::Quality,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<ModelTier, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match WireModelTier::deserialize(deserializer)? {
            WireModelTier::Fast => ModelTier::Fast,
            WireModelTier::Quality => ModelTier::Quality,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) enum WireField<T> {
    #[default]
    Missing,
    Present(T),
}

impl<'de, T> Deserialize<'de> for WireField<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Self::Present)
    }
}

impl<T> WireField<Option<T>> {
    pub(super) fn required_option_ref(&self) -> Result<Option<&T>, ()> {
        match self {
            Self::Missing => Err(()),
            Self::Present(value) => Ok(value.as_ref()),
        }
    }
}
