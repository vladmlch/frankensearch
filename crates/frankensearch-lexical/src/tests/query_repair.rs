use super::*;

/// bd-f20ye: BOOSTING A GROUP CHANGES ITS BOOLEAN MEANING in tantivy
/// 0.26.1 whenever the group contains a negation. Pinned, not fixed.
///
/// The bead was filed — by me — claiming the lenient-parse fallback "drops"
/// the negation, which is what the Divergence Register had said about this
/// shape since DIV-007. Probing the parser falsified all of it: nothing is
/// dropped, the `MustNot` clause is present in the parsed query, strict and
/// lenient parses agree, and `parse_query_lenient` reports NO errors. The
/// mechanism is structural.
///
/// An unboosted group lowers its negation as a `MustNot` clause OF the
/// enclosing boolean, which excludes. A BOOSTED group instead nests the
/// negation in its own `BooleanQuery { [(MustNot, ...)], msm: 0 }` and
/// attaches THAT as a clause of the outer boolean — so a matcher meaning
/// "every document except B" becomes a positive alternative, and the group
/// stops meaning what it read as.
///
/// Both directions are wrong, which is why this is pinned as behaviour
/// rather than reasoned about:
///
///   (alpha NOT beta)        [p2]      correct exclusion
///   (alpha NOT beta)^2      [p1, p2]  the excluded document comes back
///   (alpha AND NOT beta)^2  []        and the AND form loses everything
///
/// NOT FIXED HERE, deliberately. `frankensearch-lexical` is simultaneously
/// the shipping tantivy backend and the gauntlet's PINNED CONFORMANCE
/// ORACLE. Rewriting queries here to restore the intuitive meaning would
/// silently move the conformance target that Quill is measured against —
/// the exact shape of "weaken the oracle to make the comparison pass". Quill
/// executes these shapes correctly, so the divergence is registered
/// (DIV-009, blocking on bd-f20ye) and this test exists so that a tantivy
/// upgrade which changes the behaviour becomes VISIBLE instead of quietly
/// shifting that target. Same posture as bd-nqeb4's `#[should_panic]` pin.
/// bd-eeq0q: `A AND NOT B` returns A-minus-B on the SHIPPING path, while
/// the ORACLE path still reproduces tantivy 0.26.1's empty result.
///
/// PLANTED NEGATIVES RUN BOTH DIRECTIONS, because this repair can fail two
/// opposite ways and only one of them is obvious. If the repair stops
/// working, the shipping assertions go red. If it LEAKS into the oracle
/// surface, the oracle assertions go red — that direction matters more,
/// since a repaired oracle would silently move the conformance target Quill
/// is measured against, which is the "weaken the oracle to make the
/// comparison pass" failure this crate's two-role split exists to prevent.
///
/// The quoted control is the third direction: a literal `"a AND NOT b"`
/// phrase must never be rewritten, or the repair would corrupt user text
/// rather than user syntax.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn and_not_returns_a_minus_b_on_shipping_while_the_oracle_stays_bit_faithful() {
    run_with_cx(|cx| async move {
        let index = TantivyIndex::in_memory().expect("bd-eeq0q index");
        index
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("p1", "alpha beta"),
                    IndexableDocument::new("p2", "alpha gamma"),
                ],
            )
            .await
            .expect("index bd-eeq0q documents");
        index.commit(&cx).await.expect("commit bd-eeq0q documents");

        let ship = |query: &'static str| {
            let index = &index;
            let cx = &cx;
            async move {
                let mut ids = index
                    .search(cx, query, 10)
                    .await
                    .expect("bd-eeq0q shipping search")
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            }
        };

        // Every spelling of "alpha but not beta" agrees on the shipping
        // path. The first two already worked; the rest are the repair.
        for query in [
            "alpha NOT beta",
            "alpha -beta",
            "alpha AND NOT beta",
            "(alpha AND NOT beta)",
            "(alpha AND NOT beta)^2",
        ] {
            assert_eq!(
                ship(query).await,
                vec!["p2".to_owned()],
                "shipping {query:?} must return A-minus-B"
            );
        }

        // Repeated exclusion still excludes, and does not resurrect p1.
        assert_eq!(
            ship("alpha AND NOT beta AND NOT delta").await,
            vec!["p2".to_owned()],
            "a second AND NOT must keep excluding"
        );

        // A literal phrase is user TEXT, not user SYNTAX: never rewritten.
        assert!(
            ship("\"alpha AND NOT beta\"").await.is_empty(),
            "a quoted phrase must not be rewritten into an operator form"
        );

        // THE ORACLE HAS NOT MOVED. Same index, same queries, through the
        // conformance surface: tantivy 0.26.1's behaviour, defect included.
        let oracle_ids = |query: &'static str| {
            let mut ids = index
                .oracle_observe_query(&cx, query, 10, 64, &SnippetConfig::default())
                .expect("bd-eeq0q oracle observation")
                .hits
                .into_iter()
                .map(|hit| hit.doc_id)
                .collect::<Vec<_>>();
            ids.sort();
            ids
        };
        let oracle_empty = oracle_ids("alpha AND NOT beta");
        assert!(
            oracle_empty.is_empty(),
            "the oracle must still reproduce the tantivy 0.26.1 defect, got {oracle_empty:?}"
        );
        assert!(
            oracle_ids("(alpha AND NOT beta)^2").is_empty(),
            "the boosted form must stay defective on the oracle surface too"
        );
        assert_eq!(
            oracle_ids("alpha NOT beta"),
            vec!["p2".to_owned()],
            "the oracle's own correct spelling must be unaffected"
        );
    });
}

/// bd-8a2a8: the normal form the repair emits, asserted as TEXT.
///
/// The end-to-end test below proves the answers; this one pins the shape,
/// because two of the properties that make the repair safe are invisible in
/// a result set. `(a AND NOT b)^2` must keep its `-` DIRECTLY inside the
/// boosted group — `repair_boosted_group_negation` records negation against
/// the innermost open group, so an extra layer of parentheses would silently
/// un-repair DIV-009 while every membership assertion stayed green. And a
/// query with nothing to repair must come back BORROWED, which is both the
/// allocation contract and a check that the scanner declines rather than
/// guesses on input it cannot decode.
#[test]
fn a_negated_conjunction_normalises_to_explicit_occurrences() {
    for (query, expected) in [
        // The reported defect: `AND` binds tighter than the default join.
        // Operand ORDER is preserved; only the occurrences become explicit.
        ("alpha NOT beta AND gamma", "alpha (-beta +gamma)"),
        // A chain that IS the level keeps its operand order and adds no
        // parentheses.
        ("alpha AND NOT beta AND gamma", "+alpha -beta +gamma"),
        ("gamma AND NOT beta", "+gamma -beta"),
        ("NOT beta AND gamma", "-beta +gamma"),
        // DIV-009 interlock: the `-` stays directly inside the boost.
        ("(alpha AND NOT beta)^2", "(+alpha -beta)^2"),
        // Operands are carried through whole: phrases, fields, boosts, and
        // a fielded group that must not be split at its parenthesis.
        ("alpha AND NOT \"beta gamma\"", "+alpha -\"beta gamma\""),
        ("content:alpha^2 AND NOT beta", "+content:alpha^2 -beta"),
        (
            "field:(alpha beta) AND NOT gamma",
            "+field:(alpha beta) -gamma",
        ),
        // Occurrence outside the chain survives: a `+` that is dropped
        // turns a Must clause into a Should one.
        ("+alpha delta AND NOT beta", "+alpha (+delta -beta)"),
        // Recursion into a group, with the outer level left alone.
        (
            "(alpha NOT beta AND gamma) OR delta",
            "(alpha (-beta +gamma)) OR delta",
        ),
        // Stacked negators lower to one exclusion.
        ("alpha NOT -beta AND gamma", "alpha (-beta +gamma)"),
        // bd-iiidv: an ALL-NEGATIVE chain normalises too, in either operand
        // order, and the normal form carries no `AND` -- which is why the
        // repair already covered this shape while the register recorded it
        // as agreement. There is no `+` to add: every operand is excluded,
        // and the complement anchoring is the parser's job, not this
        // rewrite's.
        ("NOT beta AND NOT gamma", "-beta -gamma"),
        ("NOT gamma AND NOT beta", "-gamma -beta"),
    ] {
        assert_eq!(
            repair_negated_conjunction(query).as_ref(),
            expected,
            "normal form for {query:?}"
        );
    }

    // PLANTED NEGATIVES: everything here must be returned untouched, and
    // `Cow::Borrowed` proves the scanner did not merely round-trip it.
    for untouched in [
        // No explicit `AND`: nothing to normalise.
        "alpha NOT beta",
        "alpha -beta",
        "alpha beta",
        // A pure-positive conjunction is not defective; leave the spelling.
        "alpha AND gamma",
        "alpha AND beta AND gamma",
        // User TEXT, not user syntax.
        "\"alpha AND NOT beta\"",
        // Lowercase `and` is a pair of ordinary terms in this grammar.
        "alpha and not beta",
        // Undecodable input is declined, never guessed at.
        "alpha AND",
        "AND NOT beta",
        "alpha AND OR beta",
        "alpha AND \"unbalanced",
    ] {
        assert!(
            matches!(repair_negated_conjunction(untouched), Cow::Borrowed(_)),
            "{untouched:?} must be left untouched, got {:?}",
            repair_negated_conjunction(untouched)
        );
    }
}

/// bd-8a2a8: `A NOT B AND C` returns the declared reading on the SHIPPING
/// path, while the ORACLE path still reproduces tantivy 0.26.1's answer.
///
/// THE DECLARED READING is `default join := OR; explicit AND has precedence
/// over OR` (`docs/contracts/quill-language-contract.md`), so
/// `alpha NOT beta AND gamma` is `alpha OR (gamma AND NOT beta)`. Tantivy
/// 0.26.1 lowers the conjunct's negation into a positive-less boolean, so
/// the conjunct matches nothing and the whole `AND` clause disappears —
/// `p3` is the document that only the correct reading returns.
///
/// THE ATTRIBUTION CONTROL is the same query with its grouping written out.
/// `alpha ((gamma) NOT beta)` carries no `AND`, is therefore untouched by
/// every repair, and this engine answers it CORRECTLY on both roles. So the
/// engine already agrees with the declared reading when the grouping is
/// explicit, which is what places the defect in the implicit lowering
/// rather than in either side's semantics.
///
/// PLANTED NEGATIVES RUN THREE DIRECTIONS. If the repair stops working the
/// shipping assertions go red; if it LEAKS into the oracle surface the
/// oracle assertions go red, which is the direction that matters more,
/// because a repaired oracle silently moves the conformance target Quill is
/// measured against. The third is the repair ORDER: `repair_and_not` deletes
/// the `AND` in front of a `NOT`, so if it ran first, `alpha AND NOT beta
/// AND gamma` would arrive here as the unrelated chain `NOT beta AND gamma`
/// and normalise to `alpha (+gamma -beta)` — a disjunction returning three
/// documents where the conjunction returns one.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn a_negated_conjunction_reads_as_a_disjunct_on_shipping_while_the_oracle_stays_bit_faithful() {
    run_with_cx(|cx| async move {
        let index = TantivyIndex::in_memory().expect("bd-8a2a8 index");
        index
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("p1", "alpha beta"),
                    IndexableDocument::new("p2", "alpha gamma"),
                    IndexableDocument::new("p3", "gamma"),
                    IndexableDocument::new("p4", "beta gamma"),
                    IndexableDocument::new("p5", "delta"),
                ],
            )
            .await
            .expect("index bd-8a2a8 documents");
        index.commit(&cx).await.expect("commit bd-8a2a8 documents");

        let ship = |query: &'static str| {
            let index = &index;
            let cx = &cx;
            async move {
                let mut ids = index
                    .search(cx, query, 10)
                    .await
                    .expect("bd-8a2a8 shipping search")
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            }
        };
        let ids = |wanted: &[&str]| wanted.iter().map(|id| (*id).to_owned()).collect::<Vec<_>>();

        // THE DEFECT, and the same reading spelled three other ways. `p3`
        // matches `gamma` without `beta`, so it appears only when the `AND`
        // conjunct survives.
        for query in [
            "alpha NOT beta AND gamma",
            "alpha NOT -beta AND gamma",
            "alpha ((gamma) NOT beta)",
            "alpha (gamma AND NOT beta)",
        ] {
            assert_eq!(
                ship(query).await,
                ids(&["p1", "p2", "p3"]),
                "shipping {query:?} must return alpha OR (gamma minus beta)"
            );
        }

        // THE CHAIN STILL BINDS AS A CONJUNCTION. This is the repair-order
        // negative: a disjunctive reading here would return three documents.
        assert_eq!(
            ship("alpha AND NOT beta AND gamma").await,
            ids(&["p2"]),
            "a single AND chain must stay a conjunction"
        );
        for query in ["gamma AND NOT beta", "NOT beta AND gamma"] {
            assert_eq!(
                ship(query).await,
                ids(&["p2", "p3"]),
                "shipping {query:?} must return gamma minus beta"
            );
        }

        // The contract's all-documents literal is an operand like any
        // other, and must survive being given an explicit occurrence.
        assert_eq!(
            ship("* AND NOT beta").await,
            ids(&["p2", "p3", "p5"]),
            "every document except the excluded ones"
        );

        // Controls: a pure-positive conjunction is unchanged, an explicit
        // `+` still means Must, and a quoted phrase is text.
        assert_eq!(ship("alpha AND gamma").await, ids(&["p2"]));
        assert_eq!(
            ship("+alpha delta AND NOT beta").await,
            ids(&["p1", "p2"]),
            "a required clause must not be demoted to optional"
        );
        assert_eq!(
            ship("alpha delta AND NOT beta").await,
            ids(&["p1", "p2", "p5"]),
            "without the `+` the same query is a disjunction"
        );
        assert!(
            ship("\"alpha NOT beta AND gamma\"").await.is_empty(),
            "a quoted phrase must not be rewritten into an operator form"
        );

        // THE ORACLE HAS NOT MOVED. Same index, same queries, through the
        // conformance surface: tantivy 0.26.1's behaviour, defect included.
        let oracle_ids = |query: &'static str| {
            let mut ids = index
                .oracle_observe_query(&cx, query, 10, 64, &SnippetConfig::default())
                .expect("bd-8a2a8 oracle observation")
                .hits
                .into_iter()
                .map(|hit| hit.doc_id)
                .collect::<Vec<_>>();
            ids.sort();
            ids
        };
        assert_eq!(
            oracle_ids("alpha NOT beta AND gamma"),
            ids(&["p1", "p2"]),
            "the oracle must still drop the AND conjunct entirely"
        );
        assert!(
            oracle_ids("gamma AND NOT beta").is_empty(),
            "the oracle must still return nothing for a bare negated conjunction"
        );
        assert_eq!(
            oracle_ids("alpha ((gamma) NOT beta)"),
            ids(&["p1", "p2", "p3"]),
            "the explicit grouping is answered correctly by BOTH roles, which \
             is what attributes the defect to the implicit lowering"
        );
    });
}

/// bd-iiidv: an ALL-NEGATIVE conjunction is DIV-010's defect too, and the
/// register said it was agreement.
///
/// The Divergence Register's DIV-010 entry claimed `NOT alpha AND NOT beta`
/// "still returns nothing on both, because an exclusion-only conjunction
/// has no positive term — that is agreement, not a defect, and is pinned so
/// a later change cannot silently turn it into match-all". Measured, it is
/// not agreement and there was no such pin. This is the pin.
///
/// The claim is wrong twice over. An all-negative ROOT does not match
/// nothing — both engines give it complement semantics for every spelling
/// that carries no explicit `AND` — and the `AND` spelling does not agree:
/// the pinned oracle empties it while the declared reading, Quill and the
/// repaired shipping path all return the complement.
///
/// It is the SAME mechanism as the rest of DIV-010, not a new one: an
/// explicit `AND` whose operand is a negation lowers to a positive-less
/// boolean that matches nothing and empties the conjunction. That is why
/// `repair_negated_conjunction` already fixes it — the normal form carries
/// no `AND` — and why the fix needed no new repair, only this proof that it
/// covers the shape.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn an_all_negative_conjunction_is_complement_on_shipping_and_empty_on_the_oracle() {
    run_with_cx(|cx| async move {
        let index = TantivyIndex::in_memory().expect("bd-iiidv index");
        index
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("p1", "alpha beta"),
                    IndexableDocument::new("p2", "alpha gamma"),
                    IndexableDocument::new("p3", "delta gamma"),
                    IndexableDocument::new("p4", "epsilon"),
                ],
            )
            .await
            .expect("index bd-iiidv documents");
        index.commit(&cx).await.expect("commit bd-iiidv documents");

        let shipping = |query: &'static str| {
            let index = &index;
            let cx = &cx;
            async move {
                let mut ids = index
                    .search(cx, query, 10)
                    .await
                    .expect("bd-iiidv search")
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            }
        };
        let oracle = |query: &'static str| {
            let index = &index;
            let cx = &cx;
            async move {
                let mut ids = index
                    .oracle_observe_query(cx, query, 10, 64, &SnippetConfig::default())
                    .expect("bd-iiidv oracle observation")
                    .hits
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            }
        };

        // THE DEFECT, in both operand orders. `p4` is the only document
        // matching neither term, so the complement is exactly `[p4]` and an
        // empty answer is not "no positive term" -- it is the conjunct
        // being emptied.
        for query in ["NOT beta AND NOT gamma", "NOT gamma AND NOT beta"] {
            assert_eq!(
                oracle(query).await,
                Vec::<String>::new(),
                "the pinned oracle must still empty an all-negative conjunction: {query}"
            );
            assert_eq!(
                shipping(query).await,
                vec!["p4".to_owned()],
                "the shipping path must answer the complement: {query}"
            );
        }

        // THE ATTRIBUTION, and the refutation of the "agreement" claim in
        // one measurement: the SAME oracle answers every spelling of the
        // same intent that carries no explicit `AND` with the complement.
        // An engine that gives an all-negative root complement semantics
        // whenever the conjunction is spelled another way is mislowering
        // the `AND` form, not asserting that exclusion-only matches
        // nothing.
        for query in ["NOT beta NOT gamma", "-beta -gamma", "NOT beta -gamma"] {
            assert_eq!(
                oracle(query).await,
                vec!["p4".to_owned()],
                "the oracle gives an all-negative root complement semantics here: {query}"
            );
            assert_eq!(
                shipping(query).await,
                vec!["p4".to_owned()],
                "and the shipping path agrees, so these spellings are not divergent: {query}"
            );
        }

        // NOT match-all, which is the failure the register's own sentence
        // was worried about: a single negation still excludes, and the
        // complement of a term that matches nothing is still bounded by the
        // corpus.
        assert_eq!(
            shipping("NOT beta").await,
            vec!["p2".to_owned(), "p3".to_owned(), "p4".to_owned()]
        );
        assert_eq!(
            oracle("NOT beta").await,
            vec!["p2".to_owned(), "p3".to_owned(), "p4".to_owned()]
        );
    });
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn boosting_a_group_that_negates_changes_its_meaning_in_the_pinned_oracle() {
    run_with_cx(|cx| async move {
        let index = TantivyIndex::in_memory().expect("bd-f20ye index");
        index
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("p1", "alpha beta"),
                    IndexableDocument::new("p2", "alpha gamma"),
                ],
            )
            .await
            .expect("index bd-f20ye documents");
        index.commit(&cx).await.expect("commit bd-f20ye documents");

        // The SHIPPING role, through the public search surface.
        let hits = |query: &'static str| {
            let index = &index;
            let cx = &cx;
            async move {
                let mut ids = index
                    .search(cx, query, 10)
                    .await
                    .expect("bd-f20ye search")
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            }
        };
        // The ORACLE role, through the conformance observation surface.
        let oracle_hits = |query: &'static str| {
            let index = &index;
            let cx = &cx;
            async move {
                let mut ids = index
                    .oracle_observe_query(cx, query, 10, 64, &SnippetConfig::default())
                    .expect("bd-f20ye oracle observation")
                    .hits
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            }
        };

        // The negation is NOT dropped and recovery is NOT involved: the
        // lenient parser reports no errors at all for the broken shape.
        let (_parsed, errors) = index
            .query_parser()
            .parse_query_lenient("(alpha NOT beta)^2");
        assert!(
            errors.is_empty(),
            "bd-f20ye is not a lenient-recovery defect; the parser reported {errors:?}"
        );

        // CORRECT, unboosted — both roles agree here.
        assert_eq!(hits("alpha NOT beta").await, vec!["p2".to_owned()]);
        assert_eq!(hits("(alpha NOT beta)").await, vec!["p2".to_owned()]);
        assert!(hits("(alpha NOT alpha)").await.is_empty());

        // The page-observation surface must ALSO stay bit-faithful: a
        // repair leaking into any oracle_observe_* method is what would
        // move the conformance target without a comparison going red.
        let page = index
            .oracle_observe_page(&cx, "(alpha NOT beta)^2", 10, 0)
            .expect("bd-f20ye oracle page observation");
        assert_eq!(
            page.hits.len(),
            2,
            "the oracle page surface must reproduce the defect, not the repair"
        );

        // STILL WRONG IN THE ORACLE ROLE, and that is the point: the
        // comparator must reproduce Tantivy 0.26.1 byte for byte, defects
        // included, or the conformance target moves under Quill.
        assert_eq!(
            oracle_hits("(alpha NOT beta)^2").await,
            vec!["p1".to_owned(), "p2".to_owned()],
            "if this is now [p2], tantivy fixed the boosted-group negation and DIV-009 \
             must be re-measured before it is retired"
        );
        assert_eq!(
            oracle_hits("(alpha NOT alpha)^2").await,
            vec!["p1".to_owned(), "p2".to_owned()],
            "if this is now empty, tantivy fixed the boosted-group negation"
        );
        assert!(
            oracle_hits("(alpha AND NOT beta)^2").await.is_empty(),
            "if this is now [p2], tantivy fixed the boosted-group negation"
        );

        // REPAIRED IN THE SHIPPING ROLE (bd-f20ye owner ruling): a boost
        // must never change which documents match.
        assert_eq!(
            hits("(alpha NOT beta)^2").await,
            vec!["p2".to_owned()],
            "the shipping path must not return a document the query excluded"
        );
        assert!(
            hits("(alpha NOT alpha)^2").await.is_empty(),
            "a self-contradictory group must match nothing however it is boosted"
        );
        // THE INVARIANT THE RULING ASKS FOR, stated as an invariant rather
        // than as hardcoded expectations: a boost must never change WHICH
        // documents match. Asserting `boosted == unboosted` says exactly
        // that, and stays true whatever the unboosted form happens to
        // return.
        for group in [
            "(alpha NOT beta)",
            "(alpha NOT alpha)",
            "(alpha AND NOT beta)",
            "(alpha -beta)",
            "(alpha OR beta)",
        ] {
            let boosted = format!("{group}^2");
            assert_eq!(
                hits(Box::leak(boosted.into_boxed_str())).await,
                hits(Box::leak(group.to_owned().into_boxed_str())).await,
                "boosting {group} changed which documents match in the shipping path"
            );
        }
        // SUPERSEDED BY bd-eeq0q, deliberately flipped rather than deleted.
        // When this test landed, `A AND NOT B` returned nothing on the
        // shipping path too — a SEPARATE defect from the boosted-group one,
        // filed as bd-eeq0q and repaired there by `repair_and_not`. This
        // assertion recorded the defect as it stood; it now records the
        // repair. The invariant ABOVE is what belongs to this bead and is
        // unchanged: a boost may not alter membership. Keeping the line
        // (rather than dropping it) preserves the fact that the two defects
        // were separable and were separated.
        assert_eq!(
            hits("alpha AND NOT beta").await,
            vec!["p2".to_owned()],
            "bd-eeq0q repairs the ungrouped form on the shipping path"
        );
        // NESTED, so the repair is not merely a whole-query special case.
        assert_eq!(
            hits("beta OR (alpha NOT beta)^2").await,
            vec!["p1".to_owned(), "p2".to_owned()]
        );
        assert_eq!(
            hits("(alpha -beta)^3").await,
            vec!["p2".to_owned()],
            "the `-term` exclusion form must be repaired too"
        );

        // CONTROL: a boosted group WITHOUT a negation keeps its meaning in
        // BOTH roles, so the repair is specific to negation and does not
        // quietly disarm boosts in general.
        for query in ["(alpha OR beta)^2", "(beta)^2"] {
            assert_eq!(
                hits(query).await,
                oracle_hits(query).await,
                "{query} must be identical in both roles"
            );
        }
        assert_eq!(
            hits("(alpha OR beta)^2").await,
            vec!["p1".to_owned(), "p2".to_owned()]
        );
        assert_eq!(hits("(beta)^2").await, vec!["p1".to_owned()]);
    });
}
