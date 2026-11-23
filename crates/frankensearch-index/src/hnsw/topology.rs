use std::collections::{HashMap, HashSet, VecDeque};

use frankensearch_core::SearchError;
use hnsw_rs::prelude::{DistDot, Hnsw, Neighbour, PointId};

#[cfg(test)]
use super::types::HnswTopologyPoint;

pub(super) fn validate_hnsw_topology(
    hnsw: &Hnsw<'_, f32, DistDot>,
    expected_points: usize,
) -> Result<(), String> {
    let observed_count = hnsw.get_nb_point();
    if observed_count != expected_points {
        return Err(format!(
            "point count mismatch: expected {expected_points}, observed {observed_count}"
        ));
    }
    if expected_points == 0 {
        return hnsw
            .get_entry_point_id()
            .is_none()
            .then_some(())
            .ok_or_else(|| "empty graph unexpectedly exposes an entry point".to_owned());
    }

    // Retain one Arc per point, indexed below by origin id. Neighborhoods are
    // still cloned only one point at a time, so the connectivity and coverage
    // walks below never materialize a second O(edges) adjacency graph.
    let points = hnsw.get_point_indexation().into_iter().collect::<Vec<_>>();
    let identities = points
        .iter()
        .map(|point| (point.get_origin_id(), point.get_point_id()))
        .collect::<Vec<_>>();
    let point_by_internal_id = validate_hnsw_identity_table(&identities, expected_points)?;
    let entry_origin = validate_hnsw_entry_point(
        hnsw.get_entry_point_id(),
        &identities,
        &point_by_internal_id,
    )?;
    let mut points_by_origin = std::iter::repeat_with(|| None)
        .take(expected_points)
        .collect::<Vec<_>>();
    for point in points {
        let origin_id = point.get_origin_id();
        points_by_origin[origin_id] = Some(point);
    }
    for point in points_by_origin.iter().flatten() {
        let neighborhoods = point.get_neighborhood_id();
        validate_hnsw_point_neighborhoods(
            point.get_origin_id(),
            point.get_point_id(),
            &neighborhoods,
            expected_points,
            &point_by_internal_id,
        )?;
    }
    let mut base_neighbors = |origin_id: usize| -> Result<Vec<usize>, String> {
        let point = points_by_origin[origin_id]
            .as_ref()
            .ok_or_else(|| format!("missing origin id {origin_id} during connectivity walk"))?;
        Ok(point
            .get_neighborhood_id()
            .first()
            .into_iter()
            .flatten()
            .map(|neighbor| neighbor.d_id)
            .collect())
    };
    validate_weak_base_connectivity(entry_origin, expected_points, &mut base_neighbors)?;
    report_directed_base_coverage(entry_origin, expected_points, &mut base_neighbors)
}

#[cfg(test)]
pub(super) fn validate_hnsw_topology_observations(
    points: &[HnswTopologyPoint],
    expected_points: usize,
    entry_point: Option<(usize, PointId)>,
) -> Result<(), String> {
    if points.len() != expected_points {
        return Err(format!(
            "point iterator yielded {} entries for {expected_points} expected points",
            points.len()
        ));
    }

    let identities = points
        .iter()
        .map(|point| (point.origin_id, point.point_id))
        .collect::<Vec<_>>();
    let point_by_internal_id = validate_hnsw_identity_table(&identities, expected_points)?;
    if expected_points == 0 {
        return entry_point
            .is_none()
            .then_some(())
            .ok_or_else(|| "empty graph unexpectedly exposes an entry point".to_owned());
    }
    let entry_origin = validate_hnsw_entry_point(entry_point, &identities, &point_by_internal_id)?;
    let mut points_by_origin = vec![None; expected_points];
    for point in points {
        points_by_origin[point.origin_id] = Some(point);
        validate_hnsw_point_neighborhoods(
            point.origin_id,
            point.point_id,
            &point.neighborhoods,
            expected_points,
            &point_by_internal_id,
        )?;
    }
    let mut base_neighbors = |origin_id: usize| -> Result<Vec<usize>, String> {
        let point = points_by_origin[origin_id]
            .ok_or_else(|| format!("missing origin id {origin_id} during connectivity walk"))?;
        Ok(point
            .neighborhoods
            .first()
            .into_iter()
            .flatten()
            .map(|neighbor| neighbor.d_id)
            .collect())
    };
    validate_weak_base_connectivity(entry_origin, expected_points, &mut base_neighbors)?;
    report_directed_base_coverage(entry_origin, expected_points, &mut base_neighbors)
}

fn validate_hnsw_entry_point(
    entry_point: Option<(usize, PointId)>,
    identities: &[(usize, PointId)],
    point_by_internal_id: &HashMap<PointId, usize>,
) -> Result<usize, String> {
    let (entry_origin, entry_point_id) =
        entry_point.ok_or_else(|| "non-empty graph has no search entry point".to_owned())?;
    let mapped_origin = point_by_internal_id
        .get(&entry_point_id)
        .copied()
        .ok_or_else(|| format!("entry point {entry_point_id:?} is absent from the point table"))?;
    if mapped_origin != entry_origin {
        return Err(format!(
            "entry point {entry_point_id:?} maps to origin {mapped_origin}, \
             not advertised origin {entry_origin}"
        ));
    }
    let max_layer = identities
        .iter()
        .map(|(_, point_id)| point_id.0)
        .max()
        .ok_or_else(|| "non-empty graph has no point identities".to_owned())?;
    if entry_point_id.0 != max_layer {
        return Err(format!(
            "entry point {entry_point_id:?} is below maximum sampled layer {max_layer}"
        ));
    }
    Ok(entry_origin)
}

/// Attest that the base layer forms a single weak component containing the
/// search entry.
///
/// Universal *directed* reachability from the entry point is deliberately NOT
/// required here (#32). `hnsw_rs` mirrors every forward edge with a reverse
/// edge, but a neighbour list that exceeds its cap (`M`, or `2M` at the base
/// layer) is sorted by distance and its farthest edge popped, so later
/// insertions can legitimately prune away *all* of a point's in-edges. Such a
/// point becomes a directed "source": still weakly attached through its own
/// out-edges, but invisible to a directed walk from the entry. That outcome
/// is intrinsic to bounded-degree HNSW construction — serial builds exhibit
/// it as readily as parallel ones — so treating it as an integrity failure
/// rejects every sufficiently large or duplicate-heavy corpus.
///
/// A genuinely broken build (edgeless points, disjoint islands from misfiled
/// reverse edges) still fails this gate: those graphs split into multiple
/// weak components. Partial *directed* coverage is measured separately by
/// [`report_directed_base_coverage`] and surfaced as a warning, and observed
/// ANN quality has its own certificate in [`super::HnswIndex::certify_ef_search`].
pub(super) fn validate_weak_base_connectivity(
    entry_origin: usize,
    expected_points: usize,
    mut base_neighbors: impl FnMut(usize) -> Result<Vec<usize>, String>,
) -> Result<(), String> {
    let mut components = WeakComponents::new(expected_points);
    let mut isolated_count = 0_usize;
    let mut first_isolated = None;
    for origin_id in 0..expected_points {
        let neighbors = base_neighbors(origin_id)?;
        if neighbors.is_empty() {
            // A point with zero base out-edges violates the HNSW insertion
            // invariant (every non-seed insert attaches to an existing point),
            // so record it: it separates construction faults from
            // pruning-induced splits when the gate fires.
            isolated_count += 1;
            if first_isolated.is_none() {
                first_isolated = Some(origin_id);
            }
        }
        for neighbor_id in neighbors {
            if neighbor_id >= expected_points {
                return Err(format!(
                    "origin id {origin_id} references out-of-range base neighbor {neighbor_id}"
                ));
            }
            components.merge(origin_id, neighbor_id);
        }
    }

    let entry_component = components.root(entry_origin);
    let mut attached_count = 0_usize;
    let mut first_detached = None;
    let mut component_count = 0_usize;
    for origin_id in 0..expected_points {
        if components.root(origin_id) == origin_id {
            component_count += 1;
        }
        if components.root(origin_id) == entry_component {
            attached_count += 1;
        } else if first_detached.is_none() {
            first_detached = Some(origin_id);
        }
    }
    if let Some(first_detached) = first_detached {
        let first_isolated =
            first_isolated.map_or_else(|| "none".to_owned(), |origin| format!("origin {origin}"));
        return Err(format!(
            "base layer splits into multiple weak components: entry origin {entry_origin}'s \
             component holds only {attached_count}/{expected_points} points ignoring edge \
             direction; first detached origin is {first_detached}; {component_count} weak \
             components total; {isolated_count} points have zero base out-edges (first: \
             {first_isolated})"
        ));
    }
    Ok(())
}

/// Disjoint-set forest over base-layer origin ids, used to attest weak
/// connectivity in O(points) memory without materializing an adjacency graph.
struct WeakComponents {
    parent: Vec<usize>,
    size: Vec<u32>,
}

impl WeakComponents {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            size: vec![1; count],
        }
    }

    /// Iterative find with path halving; no recursion regardless of scale.
    fn root(&mut self, mut origin_id: usize) -> usize {
        while self.parent[origin_id] != origin_id {
            self.parent[origin_id] = self.parent[self.parent[origin_id]];
            origin_id = self.parent[origin_id];
        }
        origin_id
    }

    fn merge(&mut self, left_origin: usize, right_origin: usize) {
        let mut small = self.root(left_origin);
        let mut large = self.root(right_origin);
        if small == large {
            return;
        }
        if self.size[small] > self.size[large] {
            std::mem::swap(&mut small, &mut large);
        }
        self.parent[small] = large;
        self.size[large] = self.size[large].saturating_add(self.size[small]);
    }
}

/// Measure how much of the base layer a *directed* walk from the search entry
/// covers, and warn loudly when coverage is partial.
///
/// Partial coverage is admissible (see [`validate_weak_base_connectivity`])
/// but never silent: the points outside the covered set cannot be visited by
/// greedy base-layer expansion, so operators sizing a corpus should see the
/// exact count and follow up with [`super::HnswIndex::certify_ef_search`] to bound
/// the observed quality impact.
fn report_directed_base_coverage(
    entry_origin: usize,
    expected_points: usize,
    mut base_neighbors: impl FnMut(usize) -> Result<Vec<usize>, String>,
) -> Result<(), String> {
    let coverage =
        measure_directed_base_coverage(entry_origin, expected_points, &mut base_neighbors)?;
    if coverage.reached_count != expected_points {
        tracing::warn!(
            entry_origin,
            reached = coverage.reached_count,
            expected = expected_points,
            first_unreachable = ?coverage.first_unreachable,
            "HNSW base layer is weakly connected but a directed walk from the search entry \
             does not cover it; bounded neighbour pruning orphans in-edges at scale, so this \
             is not an integrity failure — verify quality via certify_ef_search"
        );
    }
    Ok(())
}

/// Outcome of the directed base-layer walk from the search entry.
pub(super) struct DirectedBaseCoverage {
    pub(super) reached_count: usize,
    pub(super) first_unreachable: Option<usize>,
}

pub(super) fn measure_directed_base_coverage(
    entry_origin: usize,
    expected_points: usize,
    mut base_neighbors: impl FnMut(usize) -> Result<Vec<usize>, String>,
) -> Result<DirectedBaseCoverage, String> {
    let mut reached = vec![false; expected_points];
    let mut queue = VecDeque::with_capacity(expected_points.min(4_096));
    reached[entry_origin] = true;
    queue.push_back(entry_origin);
    let mut reached_count = 1_usize;

    while let Some(origin_id) = queue.pop_front() {
        for neighbor_id in base_neighbors(origin_id)? {
            if neighbor_id >= expected_points {
                return Err(format!(
                    "origin id {origin_id} reaches out-of-range base neighbor {neighbor_id}"
                ));
            }
            if !reached[neighbor_id] {
                reached[neighbor_id] = true;
                reached_count += 1;
                queue.push_back(neighbor_id);
            }
        }
    }

    let first_unreachable = reached.iter().position(|is_reached| !is_reached);
    Ok(DirectedBaseCoverage {
        reached_count,
        first_unreachable,
    })
}

fn validate_hnsw_identity_table(
    identities: &[(usize, PointId)],
    expected_points: usize,
) -> Result<HashMap<PointId, usize>, String> {
    if identities.len() != expected_points {
        return Err(format!(
            "point iterator yielded {} entries for {expected_points} expected points",
            identities.len()
        ));
    }

    let mut point_by_internal_id = HashMap::with_capacity(identities.len());
    let mut origin_ids = HashSet::with_capacity(identities.len());
    for &(origin_id, point_id) in identities {
        if origin_id >= expected_points {
            return Err(format!(
                "origin id {} is outside 0..{expected_points}",
                origin_id
            ));
        }
        if !origin_ids.insert(origin_id) {
            return Err(format!("duplicate origin id {origin_id}"));
        }
        if point_id.1 < 0 {
            return Err(format!(
                "origin id {} has negative internal slot {:?}",
                origin_id, point_id
            ));
        }
        if point_by_internal_id.insert(point_id, origin_id).is_some() {
            return Err(format!("duplicate internal point id {point_id:?}"));
        }
    }

    for expected_origin_id in 0..expected_points {
        if !origin_ids.contains(&expected_origin_id) {
            return Err(format!("missing origin id {expected_origin_id}"));
        }
    }
    Ok(point_by_internal_id)
}

fn validate_hnsw_point_neighborhoods(
    origin_id: usize,
    point_id: PointId,
    neighborhoods: &[Vec<Neighbour>],
    expected_points: usize,
    point_by_internal_id: &HashMap<PointId, usize>,
) -> Result<(), String> {
    let source_max_layer = usize::from(point_id.0);
    if source_max_layer >= neighborhoods.len() {
        return Err(format!(
            "origin id {origin_id} sampled layer {} but exposes only {} neighborhoods",
            point_id.0,
            neighborhoods.len()
        ));
    }
    for (layer, neighbors) in neighborhoods.iter().enumerate() {
        if layer > source_max_layer && !neighbors.is_empty() {
            return Err(format!(
                "origin id {origin_id} has {} neighbors in layer {layer} above sampled layer \
                 {source_max_layer}",
                neighbors.len()
            ));
        }

        let mut seen_neighbors = HashSet::with_capacity(neighbors.len());
        for neighbor in neighbors {
            if !neighbor.distance.is_finite() {
                return Err(format!(
                    "origin id {origin_id} has a non-finite distance in layer {layer}"
                ));
            }
            if neighbor.d_id >= expected_points {
                return Err(format!(
                    "origin id {origin_id} references out-of-range neighbor {} in layer {layer}",
                    neighbor.d_id
                ));
            }
            if !seen_neighbors.insert(neighbor.d_id) {
                return Err(format!(
                    "origin id {origin_id} repeats neighbor {} in layer {layer}",
                    neighbor.d_id
                ));
            }
            if neighbor.d_id == origin_id {
                return Err(format!(
                    "origin id {origin_id} references itself in layer {layer}"
                ));
            }
            let Some(&internal_origin_id) = point_by_internal_id.get(&neighbor.p_id) else {
                return Err(format!(
                    "origin id {origin_id} references missing internal point {:?} in layer {layer}",
                    neighbor.p_id
                ));
            };
            if internal_origin_id != neighbor.d_id {
                return Err(format!(
                    "neighbor {:?} maps to origin {internal_origin_id}, not advertised origin {}",
                    neighbor.p_id, neighbor.d_id
                ));
            }
            if usize::from(neighbor.p_id.0) < layer {
                return Err(format!(
                    "origin id {origin_id} references neighbor {} at layer {layer}, above the \
                     neighbor's sampled layer {}",
                    neighbor.d_id, neighbor.p_id.0
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn ann_topology_error(detail: &str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "hnsw",
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("graph topology attestation failed: {detail}"),
        )),
    }
}
