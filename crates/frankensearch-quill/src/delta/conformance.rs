use sha2::{Digest, Sha256};

use crate::quiver::NumericValue;

use super::DeltaSegment;

pub(in crate::delta) fn conformance_hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update(u64::try_from(value).unwrap_or(u64::MAX).to_be_bytes());
}

pub(in crate::delta) fn conformance_hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    conformance_hash_usize(hasher, bytes.len());
    hasher.update(bytes);
}

fn conformance_hash_optional_u32(hasher: &mut Sha256, value: Option<u32>) {
    match value {
        None => hasher.update([0]),
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_be_bytes());
        }
    }
}

pub(in crate::delta) fn conformance_hash_delta_fields(hasher: &mut Sha256, delta: &DeltaSegment) {
    conformance_hash_usize(hasher, delta.fields.len());
    for field in &delta.fields {
        hasher.update(field.field_ord.to_be_bytes());
        hasher.update([u8::from(field.positions)]);
    }
}

pub(in crate::delta) fn conformance_hash_delta_documents(
    hasher: &mut Sha256,
    delta: &DeltaSegment,
) {
    conformance_hash_usize(hasher, delta.document_docids.len());
    for &global_docid in &delta.document_docids {
        hasher.update(global_docid.to_be_bytes());
    }

    conformance_hash_usize(hasher, delta.document_ids.len());
    for document_id in &delta.document_ids {
        conformance_hash_bytes(hasher, document_id.as_bytes());
    }

    conformance_hash_usize(hasher, delta.document_content_hashes.len());
    for &content_hash in &delta.document_content_hashes {
        match content_hash {
            None => hasher.update([0]),
            Some(content_hash) => {
                hasher.update([1]);
                hasher.update(content_hash.to_be_bytes());
            }
        }
    }

    conformance_hash_usize(hasher, delta.document_term_offsets.len());
    for &offset in &delta.document_term_offsets {
        hasher.update(offset.to_be_bytes());
    }
    conformance_hash_usize(hasher, delta.document_term_ids.len());
    for &term_id in &delta.document_term_ids {
        hasher.update(term_id.to_be_bytes());
    }
}

pub(in crate::delta) fn conformance_hash_delta_terms(hasher: &mut Sha256, delta: &DeltaSegment) {
    let terms = delta.sorted_terms();
    conformance_hash_usize(hasher, terms.len());
    for term in terms {
        let term_index = usize::try_from(term.term_index).expect("u32 Delta term index fits usize");
        let chain = &delta.chains[term_index];
        hasher.update(term.field_ord().to_be_bytes());
        conformance_hash_bytes(hasher, term.term());
        conformance_hash_optional_u32(hasher, chain.last_docid);
        hasher.update([chain.max_frequency_code, chain.min_fieldnorm_id]);
        conformance_hash_usize(hasher, chain.postings.len);
        conformance_hash_usize(hasher, chain.positions.len);

        for posting in term.postings() {
            hasher.update(posting.global_docid.to_be_bytes());
            hasher.update(posting.frequency.to_be_bytes());
            hasher.update(posting.position_start.to_be_bytes());
            hasher.update([u8::from(posting.has_positions)]);
            match term.positions(posting) {
                None => hasher.update([0]),
                Some(positions) => {
                    hasher.update([1]);
                    hasher.update(posting.frequency.to_be_bytes());
                    for position in positions {
                        hasher.update(position.to_be_bytes());
                    }
                }
            }
        }
    }
}

pub(in crate::delta) fn conformance_hash_delta_values(hasher: &mut Sha256, delta: &DeltaSegment) {
    conformance_hash_usize(hasher, delta.fieldnorms.len());
    for fieldnorms in &delta.fieldnorms {
        conformance_hash_usize(hasher, fieldnorms.raw_lengths.len());
        for &raw_length in &fieldnorms.raw_lengths {
            hasher.update(raw_length.to_be_bytes());
        }
        conformance_hash_bytes(hasher, &fieldnorms.fieldnorm_ids);
    }

    conformance_hash_usize(hasher, delta.numeric_fields.len());
    for field in &delta.numeric_fields {
        hasher.update(field.field_ord.to_be_bytes());
        conformance_hash_usize(hasher, field.values.len());
        for &value in &field.values {
            match value {
                None => hasher.update([0]),
                Some(NumericValue::I64(value)) => {
                    hasher.update([1]);
                    hasher.update(value.to_be_bytes());
                }
                Some(NumericValue::U64(value)) => {
                    hasher.update([2]);
                    hasher.update(value.to_be_bytes());
                }
            }
        }
    }

    conformance_hash_usize(hasher, delta.stored_fields.len());
    for field in &delta.stored_fields {
        hasher.update(field.field_ord.to_be_bytes());
        conformance_hash_usize(hasher, field.payload_bytes);
        conformance_hash_usize(hasher, field.values.len());
        for value in &field.values {
            match value {
                None => hasher.update([0]),
                Some(value) => {
                    hasher.update([1]);
                    conformance_hash_bytes(hasher, value);
                }
            }
        }
    }
}

pub(in crate::delta) fn conformance_hash_delta_visibility(
    hasher: &mut Sha256,
    delta: &DeltaSegment,
) {
    let mut live_ids = delta.live_ids.iter().collect::<Vec<_>>();
    live_ids.sort_unstable_by(|(left_id, left_docid), (right_id, right_docid)| {
        left_id
            .as_bytes()
            .cmp(right_id.as_bytes())
            .then_with(|| left_docid.cmp(right_docid))
    });
    conformance_hash_usize(hasher, live_ids.len());
    for (document_id, &global_docid) in live_ids {
        conformance_hash_bytes(hasher, document_id.as_bytes());
        hasher.update(global_docid.to_be_bytes());
    }

    conformance_hash_usize(hasher, delta.tombstone_words.len());
    for &word in &delta.tombstone_words {
        hasher.update(word.to_be_bytes());
    }
    conformance_hash_usize(hasher, delta.tombstone_count);
    conformance_hash_usize(hasher, delta.logical_bytes_used);
}
