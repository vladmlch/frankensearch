use std::ops::Bound;

/// One queryable field and its parser-time score multiplier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryField {
    /// Stable schema-local field identifier.
    pub field_id: u16,
    /// Multiplicative score boost applied when this branch is lowered.
    pub boost: f32,
}

impl QueryField {
    /// Construct a field target.
    #[must_use]
    pub const fn new(field_id: u16, boost: f32) -> Self {
        Self { field_id, boost }
    }
}

/// One normalized phrase term with its analyzer position.
///
/// Positions are retained even when an oversized predecessor was rejected, so
/// lowerers cannot accidentally turn a gapped phrase into an adjacent phrase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionedTerm {
    /// Logical analyzer position.
    pub position: u32,
    /// Normalized term text.
    pub text: String,
}

/// One schema-typed scalar used by numeric ranges and sets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryValue {
    /// Signed 64-bit value.
    I64(i64),
    /// Unsigned 64-bit value.
    U64(u64),
    /// Exact field-typed text value after analyzer normalization.
    Str(String),
}

impl PositionedTerm {
    /// Construct a positioned phrase term.
    #[must_use]
    pub fn new(position: u32, text: impl Into<String>) -> Self {
        Self {
            position,
            text: text.into(),
        }
    }
}

/// Clause occurrence shared by query planning and scorer lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occur {
    /// The child must match.
    Must,
    /// The child is optional unless the Boolean node has no required child.
    Should,
    /// The child must not match and contributes no score.
    MustNot,
}

/// Explicit source operator retained for fixture explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOperator {
    /// Explicit conjunction.
    And,
    /// Explicit disjunction.
    Or,
}

/// One owned child used while constructing a Boolean query arena.
#[derive(Debug, Clone, PartialEq)]
pub struct BooleanClause {
    /// Match occurrence.
    pub occur: Occur,
    /// Child query.
    pub query: Query,
}

impl BooleanClause {
    /// Construct a Boolean child.
    #[must_use]
    pub fn new(occur: Occur, query: Query) -> Self {
        Self { occur, query }
    }
}

/// Stable index into one [`Query`] arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryNodeId(usize);

impl QueryNodeId {
    pub(super) const ROOT: Self = Self(usize::MAX);
    pub(super) const ALL: Self = Self(usize::MAX - 1);
    pub(super) const EMPTY: Self = Self(usize::MAX - 2);

    pub(super) const fn index(self) -> usize {
        self.0
    }
}

/// One Boolean edge in stable construction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryClause {
    /// Match occurrence.
    pub occur: Occur,
    /// Child node in the owning query arena.
    pub query: QueryNodeId,
}

/// One node in an engine-neutral lexical query arena.
///
/// Text and phrase leaves may target several fields when analysis produced the
/// same token sequence for each field. This preserves the default parser's
/// ordered `[content, title]` expansion without smuggling backend handles into
/// the durable query boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryNode {
    /// Match nothing.
    Empty,
    /// A bounded arena allocation failed before the requested shape existed.
    AllocationFailure {
        /// Number of arena entries the failed construction attempted to reserve.
        requested: usize,
    },
    /// Match every live document.
    All,
    /// One analyzed term over one or more fields.
    Term {
        /// Ordered field expansion.
        fields: Vec<QueryField>,
        /// Normalized term text.
        text: String,
    },
    /// Exact-position phrase over one or more fields.
    Phrase {
        /// Ordered field expansion.
        fields: Vec<QueryField>,
        /// Normalized terms in phrase order.
        terms: Vec<PositionedTerm>,
        /// Permitted positional slop.
        slop: u32,
        /// Whether the last term is a prefix.
        prefix: bool,
    },
    /// Boolean combination.
    Boolean {
        /// Stable construction-order children.
        clauses: Vec<QueryClause>,
        /// Explicit source operator, when one was present.
        operator: Option<BooleanOperator>,
    },
    /// Inclusive/exclusive signed numeric range.
    Range {
        /// Stable numeric field identifier.
        field_id: u16,
        /// Lower bound.
        lower: Bound<QueryValue>,
        /// Upper bound.
        upper: Bound<QueryValue>,
    },
    /// Exact set membership over one typed field.
    Set {
        /// Stable field identifier.
        field_id: u16,
        /// Values in stable source order after exact duplicate removal.
        values: Vec<QueryValue>,
    },
    /// Glob pattern over an ordered set of fields.
    Glob {
        /// Stable field identifiers.
        field_ids: Vec<u16>,
        /// Lowercased glob pattern in source syntax.
        pattern: String,
    },
    /// Explicit score multiplier.
    Boost {
        /// Wrapped query.
        query: QueryNodeId,
        /// Finite, non-negative multiplier.
        factor: f32,
    },
}

/// Owned postorder query arena with an inline root.
///
/// Every descendant child id is smaller than its parent id; the inline root is
/// addressed by a sentinel id. Recursive Boolean and boost ownership therefore
/// becomes bounded heap storage, and cloning or dropping an adversarially
/// nested query never consumes the native call stack. Keeping the root inline
/// also makes leaf, match-all, and match-none construction allocation-free.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub(super) nodes: Vec<QueryNode>,
    pub(super) root: QueryNode,
}

impl Query {
    const fn node_id(index: usize) -> QueryNodeId {
        QueryNodeId(index)
    }

    fn new(node: QueryNode) -> Self {
        Self {
            nodes: Vec::new(),
            root: node,
        }
    }

    /// Match nothing.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(QueryNode::Empty)
    }

    pub(super) fn arena_allocation_failure(requested: usize) -> Self {
        Self::new(QueryNode::AllocationFailure { requested })
    }

    /// Match every live document.
    #[must_use]
    pub fn all() -> Self {
        Self::new(QueryNode::All)
    }

    fn leaf(node: QueryNode) -> Self {
        debug_assert!(!matches!(
            node,
            QueryNode::Boolean { .. } | QueryNode::Boost { .. }
        ));
        Self::new(node)
    }

    /// Construct a term leaf.
    #[must_use]
    pub fn term(fields: Vec<QueryField>, text: String) -> Self {
        Self::leaf(QueryNode::Term { fields, text })
    }

    /// Construct a phrase leaf.
    #[must_use]
    pub fn phrase(
        fields: Vec<QueryField>,
        terms: Vec<PositionedTerm>,
        slop: u32,
        prefix: bool,
    ) -> Self {
        Self::leaf(QueryNode::Phrase {
            fields,
            terms,
            slop,
            prefix,
        })
    }

    /// Construct a typed range leaf.
    #[must_use]
    pub fn range(field_id: u16, lower: Bound<QueryValue>, upper: Bound<QueryValue>) -> Self {
        Self::leaf(QueryNode::Range {
            field_id,
            lower,
            upper,
        })
    }

    /// Construct a typed set leaf.
    #[must_use]
    pub fn set(field_id: u16, values: Vec<QueryValue>) -> Self {
        Self::leaf(QueryNode::Set { field_id, values })
    }

    /// Construct a glob leaf.
    #[must_use]
    pub fn glob(field_ids: Vec<u16>, pattern: String) -> Self {
        Self::leaf(QueryNode::Glob { field_ids, pattern })
    }

    /// Construct a Boolean arena while retaining every clause occurrence.
    #[must_use]
    pub fn boolean(clauses: Vec<BooleanClause>, operator: Option<BooleanOperator>) -> Self {
        if let Some(requested) = clauses
            .iter()
            .find_map(|clause| clause.query.allocation_failure())
        {
            return Self::arena_allocation_failure(requested);
        }
        let mut query = Self {
            nodes: Vec::new(),
            root: QueryNode::Empty,
        };
        let descendants = clauses.iter().fold(0_usize, |count, clause| {
            count.saturating_add(clause.query.nodes.len().saturating_add(1))
        });
        if query.nodes.try_reserve_exact(descendants).is_err() {
            return Self::arena_allocation_failure(descendants);
        }
        let mut flat = Vec::new();
        if flat.try_reserve_exact(clauses.len()).is_err() {
            return Self::arena_allocation_failure(clauses.len());
        }
        for clause in clauses {
            let child = query.append(clause.query);
            flat.push(QueryClause {
                occur: clause.occur,
                query: child,
            });
        }
        query.root = QueryNode::Boolean {
            clauses: flat,
            operator,
        };
        query
    }

    /// Construct a boosted query without recursive ownership.
    #[must_use]
    pub fn boost(query: Self, factor: f32) -> Self {
        if let Some(requested) = query.allocation_failure() {
            return Self::arena_allocation_failure(requested);
        }
        let mut arena = Self {
            nodes: Vec::new(),
            root: QueryNode::Empty,
        };
        if arena
            .nodes
            .try_reserve_exact(query.nodes.len().saturating_add(1))
            .is_err()
        {
            return Self::arena_allocation_failure(query.nodes.len().saturating_add(1));
        }
        let child = arena.append(query);
        arena.root = QueryNode::Boost {
            query: child,
            factor,
        };
        arena
    }

    pub(super) fn push(&mut self, node: QueryNode) -> QueryNodeId {
        let id = Self::node_id(self.nodes.len());
        debug_assert!(self.nodes.len() < self.nodes.capacity());
        self.nodes.push(node);
        id
    }

    pub(super) fn append(&mut self, other: Self) -> QueryNodeId {
        if other.nodes.is_empty() {
            match &other.root {
                QueryNode::All => return QueryNodeId::ALL,
                QueryNode::Empty => return QueryNodeId::EMPTY,
                _ => {}
            }
        }
        let base = self.nodes.len();
        let child_root = base.saturating_add(other.nodes.len());
        let remap = |id: QueryNodeId| {
            if id == QueryNodeId::ROOT {
                QueryNodeId(child_root)
            } else if matches!(id, QueryNodeId::ALL | QueryNodeId::EMPTY) {
                id
            } else {
                QueryNodeId(base.saturating_add(id.index()))
            }
        };
        for node in other.nodes {
            self.push(remap_query_node(node, remap));
        }
        self.push(remap_query_node(other.root, remap))
    }

    /// Root node.
    #[must_use]
    pub fn root(&self) -> &QueryNode {
        &self.root
    }

    /// Root node id.
    #[must_use]
    pub const fn root_id(&self) -> QueryNodeId {
        QueryNodeId::ROOT
    }

    /// Node by id.
    #[must_use]
    pub fn node(&self, id: QueryNodeId) -> &QueryNode {
        if id == QueryNodeId::ROOT {
            &self.root
        } else if id == QueryNodeId::ALL {
            &QueryNode::All
        } else if id == QueryNodeId::EMPTY {
            &QueryNode::Empty
        } else {
            &self.nodes[id.index()]
        }
    }

    /// Number of logical nodes, including allocation-free `All` and `Empty`
    /// children represented by sentinel edges.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.iter().chain(std::iter::once(&self.root)).fold(
            self.nodes.len().saturating_add(1),
            |count, node| {
                let sentinels = match node {
                    QueryNode::Boolean { clauses, .. } => clauses
                        .iter()
                        .filter(|clause| {
                            clause.query == QueryNodeId::ALL || clause.query == QueryNodeId::EMPTY
                        })
                        .count(),
                    QueryNode::Boost { query, .. }
                        if *query == QueryNodeId::ALL || *query == QueryNodeId::EMPTY =>
                    {
                        1
                    }
                    _ => 0,
                };
                count.saturating_add(sentinels)
            },
        )
    }

    pub(super) fn node_mut(&mut self, id: QueryNodeId) -> &mut QueryNode {
        if id == QueryNodeId::ROOT {
            &mut self.root
        } else {
            &mut self.nodes[id.index()]
        }
    }

    /// Whether this tree is the match-none sentinel.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self.root(), QueryNode::Empty)
    }

    /// Failed arena reservation carried without masquerading as match-none.
    #[must_use]
    pub fn allocation_failure(&self) -> Option<usize> {
        match &self.root {
            QueryNode::AllocationFailure { requested } => Some(*requested),
            _ => None,
        }
    }
}

pub(super) fn remap_query_node(
    node: QueryNode,
    remap: impl Fn(QueryNodeId) -> QueryNodeId + Copy,
) -> QueryNode {
    match node {
        QueryNode::Boolean {
            mut clauses,
            operator,
        } => {
            for clause in &mut clauses {
                clause.query = remap(clause.query);
            }
            QueryNode::Boolean { clauses, operator }
        }
        QueryNode::Boost { query, factor } => QueryNode::Boost {
            query: remap(query),
            factor,
        },
        leaf => leaf,
    }
}
