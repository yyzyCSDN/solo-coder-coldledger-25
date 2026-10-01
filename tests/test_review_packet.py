import unittest
from dataclasses import replace
from datetime import datetime, timezone

from coldledger.quality.release import Case, Policy, evaluate, may_release, select_for
from coldledger.quality.evidence_graph import CaseEdge, reachable
from coldledger.quality.review_packet import (
    EvidenceSnapshot,
    PacketTampered,
    ReplayDiverged,
    compare_with_current,
    open_review,
    replay,
)

UTC = timezone.utc
JAN = datetime(2026, 1, 1, tzinfo=UTC)
MAR = datetime(2026, 3, 1, tzinfo=UTC)
JUN = datetime(2026, 6, 1, tzinfo=UTC)
SEP = datetime(2026, 9, 1, tzinfo=UTC)


def edges():
    # seed -> pod -> coa ; a separate branch that must be traversed
    return (
        CaseEdge("seed", "pod"),
        CaseEdge("pod", "coa", weight=0.5, kind="supports"),
        CaseEdge("seed", "log"),
    )


def snapshot():
    return EvidenceSnapshot.capture(edges(), ["seed"])


class RuleAnchorTests(unittest.TestCase):
    def test_policy_pinned_to_case_time_not_wall_clock(self):
        old = Policy(1, 5.0, frozenset({"pod"}), JAN)
        strict = Policy(2, 1.0, frozenset({"pod", "coa"}), JUN)
        case = Case(MAR, score=2.0, documents=frozenset({"pod"}))
        # Under the March-effective rule v1 the case releases.
        self.assertTrue(may_release(case, [old, strict]))
        # The strict June rule must NOT leak back into the historical decision.
        self.assertEqual(select_for(case, [old, strict]).version, 1)

    def test_review_opened_at_overrides_received_at(self):
        old = Policy(1, 5.0, frozenset(), JAN)
        strict = Policy(2, 1.0, frozenset(), JUN)
        case = Case(MAR, score=2.0, documents=frozenset(), review_opened_at=SEP)
        # Case physically arrived in March but only entered review in Sept,
        # so the rule governing the review is v2.
        self.assertEqual(select_for(case, [old, strict]).version, 2)
        self.assertFalse(may_release(case, [old, strict]))

    def test_explicit_anchor_for_historical_lookup(self):
        old = Policy(1, 5.0, frozenset(), JAN)
        strict = Policy(2, 1.0, frozenset(), JUN)
        case = Case(SEP, score=2.0, documents=frozenset())
        # Re-looking-up "what applied back in March" must return v1.
        self.assertEqual(select_for(case, [old, strict], at=MAR).version, 1)

    def test_no_policy_before_case_time_is_an_error_not_a_silent_pick(self):
        future = Policy(9, 1.0, frozenset(), JUN)
        case = Case(MAR, score=0.0, documents=frozenset())
        with self.assertRaises(ValueError):
            select_for(case, [future])

    def test_naive_datetime_rejected(self):
        p = Policy(1, 5.0, frozenset(), JAN)
        case = Case(datetime(2026, 2, 1), 1.0, frozenset())
        with self.assertRaises(ValueError):
            may_release(case, [p])


class EvidenceGraphTests(unittest.TestCase):
    def test_reachable_traverses_multiple_hops(self):
        got = reachable(["seed"], edges())
        self.assertEqual(got, {"seed", "pod", "coa", "log"})

    def test_reachable_respects_max_depth(self):
        got = reachable(["seed"], edges(), max_depth=1)
        self.assertEqual(got, {"seed", "pod", "log"})


class ReviewPacketTests(unittest.TestCase):
    def _case(self, docs=("pod", "coa")):
        return Case(MAR, score=2.0, documents=frozenset(docs))

    def _policies(self):
        return [
            Policy(1, 5.0, frozenset({"pod"}), JAN),
            Policy(2, 1.0, frozenset({"pod", "coa"}), JUN),
        ]

    def test_open_pins_rule_in_effect_at_review_entry(self):
        packet = open_review("C-1", self._case(), self._policies(), snapshot())
        self.assertEqual(packet.policy.version, 1)
        self.assertTrue(packet.recorded_allowed)
        self.assertEqual(packet.state, "in_review")
        self.assertTrue(packet.verify())

    def test_new_rule_does_not_change_historical_packet_on_replay(self):
        packet = open_review("C-1", self._case(), self._policies(), snapshot())
        # Even though v2 now exists, replay reproduces the v1 verdict.
        result = replay(packet)
        self.assertTrue(result.consistent)
        self.assertTrue(result.allowed)
        self.assertEqual(result.policy_version, 1)
        self.assertEqual(result.evidence_nodes, {"seed", "pod", "coa", "log"})

    def test_compare_with_current_is_read_only_and_surfaces_drift(self):
        # Case passes the lenient v1 (score 2 <= 5, only pod required) but would
        # fail v2 (score 2 > 1 and coa required-but-present... make it fail).
        case = Case(MAR, score=2.0, documents=frozenset({"pod"}))  # missing coa
        packet = open_review("C-2", case, self._policies(), snapshot())
        self.assertTrue(packet.recorded_allowed)  # v1 released it

        report = compare_with_current(packet, self._policies(), now=SEP)
        self.assertTrue(report.drifts)
        self.assertTrue(report.recorded_allowed)
        self.assertFalse(report.current_allowed)
        self.assertEqual(report.recorded_policy_version, 1)
        self.assertEqual(report.current_policy_version, 2)
        # The comparison must not have mutated the sealed packet.
        self.assertTrue(packet.verify())
        self.assertTrue(packet.recorded_allowed)

    def test_no_drift_when_rules_unchanged(self):
        packet = open_review("C-3", self._case(), self._policies(), snapshot())
        report = compare_with_current(packet, [self._policies()[0]])
        self.assertFalse(report.drifts)

    def test_replay_raises_on_verdict_drift_and_preserves_record(self):
        packet = open_review("C-4", self._case(), self._policies(), snapshot())
        # Simulate a post-hoc patch of the SEALED rule (e.g. a rule-engine fix
        # that retroactively tightens the pinned snapshot).
        patched = replace(packet.policy, max_score=0.5)
        tampered_like = replace(packet, policy=patched)
        # Re-digest at the packet level but the embedded rule change is what we
        # want replay to catch; emulate a legitimate-looking engine re-seal.
        tampered_like = replace(tampered_like, digest=tampered_like.compute_digest())
        with self.assertRaises(ReplayDiverged) as ctx:
            replay(tampered_like)
        self.assertTrue(ctx.exception.recorded_allowed)
        self.assertFalse(ctx.exception.replayed_allowed)
        self.assertEqual(ctx.exception.recorded_policy, 1)
        # report mode keeps the OLD verdict instead of silently adopting new one
        result = replay(tampered_like, on_divergence="report")
        self.assertFalse(result.consistent)
        self.assertTrue(result.allowed)  # recorded verdict preserved
        self.assertIn("DIVERGED", result.note)

    def test_tampered_snapshot_is_detected(self):
        packet = open_review("C-5", self._case(), self._policies(), snapshot())
        # Mutate the case state inside the sealed packet without re-sealing.
        hacked = replace(packet, case=replace(packet.case, score=99.0))
        with self.assertRaises(PacketTampered):
            replay(hacked)
        with self.assertRaises(PacketTampered):
            compare_with_current(hacked, self._policies())

    def test_evidence_snapshot_tampering_is_detected(self):
        packet = open_review("C-6", self._case(), self._policies(), snapshot())
        bad_evidence = replace(packet.evidence, edges=(CaseEdge("seed", "forged"),))
        hacked = replace(packet, evidence=bad_evidence)
        with self.assertRaises(PacketTampered):
            replay(hacked)

    def test_opened_at_explicitly_pins_later_rule_for_new_case(self):
        # A case that arrives in March but is opened for review in September is
        # a "new case" w.r.t. the June rule and must be judged under v2.
        case = Case(MAR, score=2.0, documents=frozenset({"pod", "coa"}))
        packet = open_review("C-7", case, self._policies(), snapshot(), opened_at=SEP)
        self.assertEqual(packet.policy.version, 2)
        # score 2.0 exceeds v2 max_score 1.0 -> rejected under the newer rule
        self.assertFalse(packet.recorded_allowed)
        self.assertEqual(packet.case.review_opened_at, SEP)

    def test_packets_chain_digests(self):
        p1 = open_review("C-8", self._case(), self._policies(), snapshot())
        p2 = open_review("C-8", self._case(), self._policies(), snapshot(),
                         opened_at=SEP, previous_digest=p1.digest)
        self.assertEqual(p2.previous_digest, p1.digest)
        self.assertTrue(p2.verify())

    def test_evaluate_records_missing_docs(self):
        case = Case(MAR, score=1.0, documents=frozenset())
        decision = evaluate(case, self._policies())
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.missing_docs, frozenset({"pod"}))
        self.assertEqual(decision.policy_version, 1)


if __name__ == "__main__":
    unittest.main()
