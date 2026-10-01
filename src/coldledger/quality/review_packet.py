"""Immutable quality-review packets and drift-aware replay.

When a case enters quality review we freeze *everything* the release decision
depends on into a single :class:`ReviewPacket`:

* the exact rule version (policy snapshot) effective at that instant,
* the evidence graph edges and the seed evidence the case presented,
* the case state (score / documents / lifecycle state),
* the decision that was originally recorded.

Re-evaluation (回看/重放) is then performed against the **snapshot**, never
against the current rulebook. If re-running the decision yields a different
outcome than the one recorded, replay raises / reports an explicit divergence
instead of silently overwriting the historical conclusion. Rules published
after the packet was opened cannot affect it; they only govern new packets.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable

from .evidence_graph import CaseEdge, reachable
from .release import Case, Policy, ReleaseDecision, evaluate

GENESIS = "GENESIS"


class ReplayDiverged(RuntimeError):
    """Replaying the snapshot no longer reproduces the recorded conclusion.

    The historical decision is never overwritten automatically. A caller must
    acknowledge the drift explicitly (e.g. open a fresh review packet).
    """

    def __init__(self, packet_id: str, recorded_allowed: bool, replayed_allowed: bool,
                 recorded_policy: int, replayed_policy: int, reason: str):
        super().__init__(
            f"review packet {packet_id!r} replay diverged: "
            f"recorded allowed={recorded_allowed} (policy v{recorded_policy}) vs "
            f"replayed allowed={replayed_allowed} (policy v{replayed_policy}): {reason}"
        )
        self.packet_id = packet_id
        self.recorded_allowed = recorded_allowed
        self.replayed_allowed = replayed_allowed
        self.recorded_policy = recorded_policy
        self.replayed_policy = replayed_policy
        self.reason = reason


class PacketTampered(RuntimeError):
    """The packet digest does not match its contents (snapshot was mutated)."""


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError("timezone-aware datetime required")
    return v.astimezone(timezone.utc)


def _edge_row(e: CaseEdge) -> list:
    return [e.source, e.target, e.weight, e.kind]


def _policy_row(p: Policy) -> list:
    return [p.version, p.max_score, sorted(p.required_docs), _utc(p.effective_at).isoformat()]


def _case_row(case: Case) -> list:
    return [
        _utc(case.received_at).isoformat(),
        case.score,
        sorted(case.documents),
        _utc(case.review_opened_at).isoformat() if case.review_opened_at else None,
    ]


def select_policy(case: Case, policies: Iterable[Policy]) -> Policy:
    """Pin the rule version effective at the case's review-entry instant."""
    anchor = case.decision_anchor()
    rows = [p for p in policies if _utc(p.effective_at) <= anchor]
    if not rows:
        raise ValueError("no effective policy at case review time")
    rows.sort(key=lambda p: (_utc(p.effective_at), p.version))
    return rows[-1]


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Frozen evidence graph plus the seed evidence the case presented."""
    edges: tuple[CaseEdge, ...]
    seeds: frozenset
    digest: str

    @staticmethod
    def _compute(edges: tuple[CaseEdge, ...], seeds: frozenset) -> str:
        raw = json.dumps(
            [sorted(seeds), [_edge_row(e) for e in sorted(edges, key=lambda x: (x.source, x.target, x.kind))]],
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def capture(cls, edges: Iterable[CaseEdge], seeds: Iterable[str]) -> "EvidenceSnapshot":
        edge_tuple = tuple(edges)
        seed_set = frozenset(seeds)
        for e in edge_tuple:
            if e.weight < 0:
                raise ValueError("negative weight")
            if e.source == e.target:
                raise ValueError("self edge")
        return cls(edge_tuple, seed_set, cls._compute(edge_tuple, seed_set))

    def verify(self) -> bool:
        return self.digest == self._compute(self.edges, self.seeds)

    def covered_nodes(self, max_depth: int | None = None) -> set[str]:
        """All evidence reachable from the case seeds within this snapshot."""
        return reachable(self.seeds, self.edges, max_depth)


@dataclass(frozen=True)
class ReviewPacket:
    case_id: str
    case: Case
    opened_at: datetime
    policy: Policy                 # the single rule version pinned at open time
    evidence: EvidenceSnapshot
    state: str                     # lifecycle state captured at open time
    recorded_allowed: bool
    recorded_policy_version: int
    previous_digest: str = GENESIS
    digest: str = ""

    def verify(self) -> bool:
        return self.digest == self.compute_digest() and self.evidence.verify()

    def compute_digest(self) -> str:
        raw = json.dumps([
            self.case_id,
            _case_row(self.case),
            _utc(self.opened_at).isoformat(),
            _policy_row(self.policy),
            self.evidence.digest,
            self.state,
            self.recorded_allowed,
            self.recorded_policy_version,
            self.previous_digest,
        ], separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ReplayResult:
    packet_id: str
    allowed: bool
    policy_version: int
    evidence_nodes: frozenset
    consistent: bool
    note: str = ""


def open_review(case_id: str, case: Case, policies: Iterable[Policy],
                evidence: EvidenceSnapshot, *, state: str = "in_review",
                opened_at: datetime | None = None,
                previous_digest: str = GENESIS) -> ReviewPacket:
    """Freeze rule version, evidence graph and case state into one packet.

    The governing policy is selected by the case's review-entry instant. The
    ``policies`` iterable is only consulted to *pin* one version here; the
    resulting packet keeps its own immutable copy, so later rule versions never
    reach it.
    """
    if opened_at is not None:
        opened_at = _utc(opened_at)
        case = replace(case, review_opened_at=opened_at)
    else:
        opened_at = _utc(case.decision_anchor())

    pinned = select_policy(case, policies)
    decision = evaluate(case, [pinned])
    packet = ReviewPacket(
        case_id=case_id,
        case=case,
        opened_at=opened_at,
        policy=pinned,
        evidence=evidence,
        state=state,
        recorded_allowed=decision.allowed,
        recorded_policy_version=pinned.version,
        previous_digest=previous_digest,
    )
    return replace(packet, digest=packet.compute_digest())


def replay(packet: ReviewPacket, *, on_divergence: str = "raise") -> ReplayResult:
    """Re-evaluate a historical packet against its own frozen snapshot.

    The current rulebook is never consulted: the verdict comes exclusively
    from ``packet.policy`` and ``packet.evidence``. If the replayed verdict
    differs from the one recorded (e.g. the evaluation rule itself was patched
    after the packet was sealed), drift is surfaced rather than applied:

      * ``"raise"`` (default): raise :class:`ReplayDiverged`.
      * ``"report"``: return a result with ``consistent=False``; the returned
        ``allowed`` keeps the *recorded* value, never the new one.

    Any mutation of the snapshot raises :class:`PacketTampered`.
    """
    if on_divergence not in {"raise", "report"}:
        raise ValueError("on_divergence must be 'raise' or 'report'")
    if not packet.verify():
        raise PacketTampered(f"review packet {packet.case_id!r} digest mismatch")

    replayed = evaluate(packet.case, [packet.policy])
    evidence_nodes = frozenset(packet.evidence.covered_nodes())

    consistent = (
        replayed.allowed == packet.recorded_allowed
        and replayed.policy_version == packet.recorded_policy_version
    )
    if not consistent:
        reason = _divergence_reason(packet, replayed)
        if on_divergence == "raise":
            raise ReplayDiverged(
                packet.case_id,
                packet.recorded_allowed,
                replayed.allowed,
                packet.recorded_policy_version,
                replayed.policy_version,
                reason,
            )
        return ReplayResult(
            packet_id=packet.case_id,
            allowed=packet.recorded_allowed,  # preserve history, never overwrite
            policy_version=packet.recorded_policy_version,
            evidence_nodes=evidence_nodes,
            consistent=False,
            note=f"DIVERGED: {reason}",
        )

    return ReplayResult(
        packet_id=packet.case_id,
        allowed=replayed.allowed,
        policy_version=replayed.policy_version,
        evidence_nodes=evidence_nodes,
        consistent=True,
        note="reproduced recorded decision from frozen snapshot",
    )


@dataclass(frozen=True)
class DriftReport:
    """Read-only comparison of a sealed packet against the current rulebook.

    This never mutates the packet and never adopts the current verdict; it only
    makes the drift observable so a human can decide whether a fresh review is
    warranted.
    """
    packet_id: str
    recorded_allowed: bool
    recorded_policy_version: int
    current_allowed: bool
    current_policy_version: int
    drifts: bool
    reason: str


def compare_with_current(packet: ReviewPacket, current_policies: Iterable[Policy],
                         *, now: datetime | None = None) -> DriftReport:
    """Show what today's rulebook would decide WITHOUT touching the packet.

    The packet stays authoritative for the historical record. A ``drifts=True``
    report means new rules exist that *would* change this case, confirming the
    packet's snapshot is what keeps the old conclusion stable. ``now`` is the
    lookup time for "current" rules (defaults to the wall clock, but is
    injectable for deterministic review).
    """
    if not packet.verify():
        raise PacketTampered(f"review packet {packet.case_id!r} digest mismatch")

    lookup_at = _utc(now) if now is not None else datetime.now(timezone.utc)
    current = evaluate(packet.case, current_policies, at=lookup_at)
    drifts = (
        current.allowed != packet.recorded_allowed
        or current.policy_version != packet.recorded_policy_version
    )
    reason = ""
    if drifts:
        reason = _drift_reason(packet, current)
    return DriftReport(
        packet_id=packet.case_id,
        recorded_allowed=packet.recorded_allowed,
        recorded_policy_version=packet.recorded_policy_version,
        current_allowed=current.allowed,
        current_policy_version=current.policy_version,
        drifts=drifts,
        reason=reason,
    )


def _divergence_reason(packet: ReviewPacket, replayed: ReleaseDecision) -> str:
    bits = []
    if replayed.policy_version != packet.recorded_policy_version:
        bits.append(
            f"policy version changed v{packet.recorded_policy_version} -> v{replayed.policy_version}"
        )
    if replayed.allowed != packet.recorded_allowed:
        if replayed.missing_docs:
            bits.append(f"missing docs under replayed rule: {sorted(replayed.missing_docs)}")
        elif packet.case.score > replayed.max_score:
            bits.append(f"score {packet.case.score} exceeds replayed max {replayed.max_score}")
        else:
            bits.append(
                f"verdict flip {'reject -> release' if replayed.allowed else 'release -> reject'}"
            )
    return "; ".join(bits) or "recorded decision differs from replay"


def _drift_reason(packet: ReviewPacket, current: ReleaseDecision) -> str:
    bits = [f"current rulebook selects v{current.policy_version}, packet sealed v{packet.recorded_policy_version}"]
    if current.allowed != packet.recorded_allowed:
        if current.missing_docs:
            bits.append(f"current rule would reject for missing docs: {sorted(current.missing_docs)}")
        elif packet.case.score > current.max_score:
            bits.append(f"current max score {current.max_score} would reject score {packet.case.score}")
        else:
            bits.append(
                f"current rule would {'release' if current.allowed else 'reject'}; "
                f"recorded verdict {'released' if packet.recorded_allowed else 'rejected'}"
            )
    return "; ".join(bits)
