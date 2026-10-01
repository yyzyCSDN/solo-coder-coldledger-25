import unittest
from datetime import datetime, timezone

from coldledger.quality.release import Policy, Case, select, may_release, governs
from coldledger.quality.evidence_graph import CaseEdge, reachable
from coldledger.quality import case_review as cr

UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 6, 1, tzinfo=UTC)
T2 = datetime(2027, 1, 1, tzinfo=UTC)

P_OLD = Policy(1, 5.0, frozenset({"pod"}), T0)
# New rule: tighter score and an extra document. Effective only in 2027.
P_NEW = Policy(2, 1.0, frozenset({"pod", "coa"}), T2)


def _case(score=4.0, docs=frozenset({"pod"}), at=T1):
    return Case(at, score, frozenset(docs))


class ReleaseSelectionTests(unittest.TestCase):
    def test_selects_rule_in_force_at_case_receipt_not_now(self):
        # Today is well past T2, but the case entered review at T1.
        self.assertEqual(select(_case(), [P_OLD, P_NEW]), P_OLD)

    def test_old_case_still_releases_under_old_rule(self):
        # Score 4 with only 'pod' passes the old rule but fails the new one;
        # look-back must keep the historical "release" conclusion.
        self.assertTrue(may_release(_case(), [P_OLD, P_NEW]))

    def test_new_case_takes_new_rule(self):
        case = _case(at=T2)
        self.assertEqual(select(case, [P_OLD, P_NEW]), P_NEW)
        self.assertFalse(may_release(case, [P_OLD, P_NEW]))  # missing coa

    def test_explicit_at_overrides_case_time(self):
        self.assertIs(select(_case(), [P_OLD, P_NEW], at=T2) is P_NEW, True)

    def test_governs_is_bound_to_effective_time(self):
        self.assertTrue(governs(P_NEW, T2))
        self.assertFalse(governs(P_NEW, T1))
        self.assertTrue(cr.applies_to(P_NEW, T2))
        self.assertFalse(cr.applies_to(P_NEW, T1))

    def test_naive_datetime_treated_as_utc(self):
        case = Case(datetime(2026, 6, 1), 4.0, frozenset({"pod"}))
        self.assertEqual(select(case, [P_OLD, P_NEW]), P_OLD)


class EvidenceClosureTests(unittest.TestCase):
    def test_reachable_traverses_new_nodes(self):
        edges = [CaseEdge("pod", "probe"), CaseEdge("probe", "report")]
        self.assertEqual(reachable({"pod"}, edges), {"pod", "probe", "report"})

    def test_reachable_respects_max_depth(self):
        edges = [CaseEdge("pod", "probe"), CaseEdge("probe", "report")]
        self.assertEqual(reachable({"pod"}, edges, max_depth=1), {"pod", "probe"})


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.edges = [CaseEdge("pod", "probe"), CaseEdge("probe", "report")]

    def test_binding_freezes_old_rule_for_old_case(self):
        b = cr.bind("C1", _case(), [P_OLD, P_NEW], self.edges)
        self.assertEqual(b.policy_snapshot.selected_version, 1)
        self.assertEqual(b.policy_snapshot.max_score, 5.0)
        self.assertEqual(b.evidence_snapshot.reached,
                         {"pod", "probe", "report"})
        self.assertEqual(b.state, "in_review")
        self.assertIsNone(b.decision)

    def test_binding_freezes_new_rule_for_new_case(self):
        b = cr.bind("C2", _case(at=T2, docs=frozenset({"pod", "coa"})),
                    [P_OLD, P_NEW], self.edges)
        self.assertEqual(b.policy_snapshot.selected_version, 2)

    def test_adjudication_uses_snapshot_and_is_stable_after_rule_change(self):
        b = cr.bind("C1", _case(score=4.0), [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1)
        self.assertTrue(adj.binding.decision)
        self.assertEqual(adj.binding.state, "adjudicated")
        # A newer, stricter rule appears; the stored decision is unchanged.
        cr.verify_binding(adj.binding, [P_OLD, P_NEW])
        result = cr.review_history(adj.binding, [P_OLD, P_NEW])
        self.assertTrue(result.decision)
        self.assertEqual(result.state, "adjudicated")

    def test_case_failing_old_rule_records_failure(self):
        b = cr.bind("C3", _case(score=9.0), [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1)
        self.assertFalse(adj.binding.decision)

    def test_new_rule_does_not_reach_back_into_old_case(self):
        b = cr.bind("C4", _case(score=4.0, docs=frozenset({"pod"})),
                    [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        adj = cr.close(adj, "qa", T1).binding
        # Same facts would be rejected under P_NEW, but locked replay keeps
        # the historical release and reports no decision change.
        result = cr.review_history(adj, [P_OLD, P_NEW])
        self.assertTrue(result.decision)
        self.assertEqual(result.snapshot_version, 1)

    def test_report_mode_flags_what_if_drift_without_mutating(self):
        b = cr.bind("C5", _case(score=4.0, docs=frozenset({"pod"})),
                    [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        before = adj
        result = cr.review_history(adj, [P_OLD, P_NEW], mode="report", now=T2)
        self.assertTrue(result.decision)            # historical stays
        self.assertFalse(result.current_decision)   # new rule would reject
        self.assertTrue(result.drifted)
        self.assertEqual(result.current_version, 2)
        # binding is untouched by a read-only report
        self.assertEqual(adj.digest, before.digest)

    def test_report_mode_before_new_rule_effective_shows_no_drift(self):
        b = cr.bind("C5b", _case(score=4.0, docs=frozenset({"pod"})),
                    [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        # looking back while still inside the old rule's window
        result = cr.review_history(adj, [P_OLD, P_NEW], mode="report", now=T1)
        self.assertEqual(result.current_version, 1)
        self.assertFalse(result.drifted)

    def test_report_mode_no_drift_when_conclusions_agree(self):
        b = cr.bind("C6", _case(score=0.5, docs=frozenset({"pod", "coa"})),
                    [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        result = cr.review_history(adj, [P_OLD, P_NEW], mode="report")
        self.assertFalse(result.drifted)
        self.assertTrue(result.current_decision)

    def test_silent_rewrite_of_old_conclusion_is_rejected(self):
        b = cr.bind("C7", _case(score=4.0), [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        # Tamper: someone with write access flips the stored conclusion and
        # re-seals the binding, while rules and evidence stay consistent.
        forged = self._forge_decision(adj, False)
        with self.assertRaises(cr.DecisionDrift):
            cr.verify_binding(forged, [P_OLD, P_NEW])
        with self.assertRaises(cr.DecisionDrift):
            cr.review_history(forged, [P_OLD, P_NEW])
        # non-strict call reports instead of raising, still no mutation
        result = cr.review_history(forged, [P_OLD, P_NEW], strict=False)
        self.assertFalse(result.decision)

    def test_unsigned_mutation_of_conclusion_is_detected(self):
        b = cr.bind("C7b", _case(score=4.0), [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        # Naive flip without re-sealing breaks the binding digest itself.
        import dataclasses
        with self.assertRaises(cr.SnapshotError):
            cr.verify_binding(dataclasses.replace(adj, decision=False),
                              [P_OLD, P_NEW])

    @staticmethod
    def _forge_decision(binding, value):
        import dataclasses
        sealed = dataclasses.replace(binding, decision=value, digest="")
        return dataclasses.replace(sealed, digest=cr._digest(sealed.canonical))

    def test_rule_backdating_that_changes_old_case_is_detected(self):
        b = cr.bind("C8", _case(score=4.0), [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        # History rewritten: the "new" rule is back-dated before review entry.
        backdated = Policy(2, 1.0, frozenset({"pod", "coa"}), T0)
        with self.assertRaises(cr.SnapshotError):
            cr.verify_binding(adj, [P_OLD, backdated])

    def test_appending_future_rule_leaves_old_case_verifiable(self):
        b = cr.bind("C9", _case(score=4.0), [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        cr.verify_binding(adj, [P_OLD, P_NEW])  # P_NEW is future at T1 -> fine

    def test_evidence_and_state_tampering_detected(self):
        b = cr.bind("C10", _case(), [P_OLD, P_NEW], self.edges)
        import dataclasses
        bad_evidence = dataclasses.replace(
            b, evidence_snapshot=dataclasses.replace(
                b.evidence_snapshot, reached=frozenset({"pod"})))
        with self.assertRaises(cr.SnapshotError):
            cr.verify_binding(bad_evidence)
        bad_state = dataclasses.replace(b, state="closed")
        with self.assertRaises(cr.SnapshotError):
            cr.verify_binding(bad_state)

    def test_close_requires_decision_and_replay_reproduces_state(self):
        b = cr.bind("C11", _case(), [P_OLD, P_NEW], self.edges)
        with self.assertRaises(ValueError):
            cr.close(b, "qa", T1)
        adj = cr.adjudicate(b, "qa", T1)
        closed = cr.close(adj.binding, "qa", T1)
        self.assertEqual(closed.binding.state, "closed")
        # replay from the binding + recorded transitions restores the state
        replayed = cr.replay(b, [adj.event, closed.event])
        self.assertEqual(replayed.state, "closed")
        self.assertEqual(replayed.revision, 2)
        self.assertTrue(replayed.decision)

    def test_replay_rejects_broken_chain(self):
        b = cr.bind("C12", _case(), [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1)
        forged = self._forge_decision(adj.binding, False)
        with self.assertRaises(ValueError):
            cr.replay(forged, [adj.event])

    def test_reopen_keeps_original_snapshot(self):
        b = cr.bind("C13", _case(score=4.0), [P_OLD], self.edges)
        adj = cr.adjudicate(b, "qa", T1).binding
        reopened = cr.reopen(adj, "qa", T1).binding
        self.assertEqual(reopened.state, "reopened")
        self.assertIsNone(reopened.decision)
        self.assertEqual(reopened.policy_snapshot.selected_version, 1)
        readj = cr.adjudicate(reopened, "qa", T1).binding
        self.assertTrue(readj.decision)  # still judged under the old rule

    def test_binding_rejects_foreign_snapshots(self):
        b1 = cr.bind("C14", _case(), [P_OLD], self.edges)
        b2 = cr.bind("C15", _case(), [P_OLD], self.edges)
        import dataclasses
        with self.assertRaises(ValueError):
            dataclasses.replace(
                b1, policy_snapshot=b2.policy_snapshot).validate()


if __name__ == "__main__":
    unittest.main()
