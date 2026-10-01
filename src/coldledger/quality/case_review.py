"""Immutable bindings between release-rule snapshots, evidence graphs and
quality-case state.

A case is decided under the rule set that was in force at the instant it
entered review. That rule set is frozen into a *policy snapshot*, the
evidence closure used for the decision is frozen into an *evidence snapshot*,
and both are bound (with content digests) to the case state. Historical
look-backs and replays evaluate the recorded snapshot, never the current
rule book, so an old conclusion can never be silently rewritten by a newer
rule. New rules are effective only for cases that enter review at or after
their ``effective_at`` (see :func:`governs` / :func:`applies_to`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from typing import Iterable

from .release import Case, Policy, governs, select
from .evidence_graph import CaseEdge, reachable

REVIEW_STATES = ("in_review", "adjudicated", "reopened", "closed")


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError("timezone-aware datetime required")
    return v.astimezone(timezone.utc)


def _digest(canonical: object) -> str:
    raw = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


# --- Rule snapshot ---------------------------------------------------------

@dataclass(frozen=True)
class PolicySnapshot:
    """The release rule frozen for a case: the policy that was selected when
    the case entered review plus a digest over *every* policy that was known
    at that instant. The digest lets replays detect post-hoc tampering with
    the policy registry (a changed threshold, removed document, ...)."""
    case_ref: str
    entered_review_at: datetime
    selected_version: int
    max_score: float
    required_docs: frozenset[str]
    registry_digest: str
    digest: str

    @staticmethod
    def _canonical(case_ref, entered_at, selected, registry_digest) -> list:
        return [
            "policy-snapshot",
            case_ref,
            _utc(entered_at).isoformat(),
            selected.version,
            selected.max_score,
            sorted(selected.required_docs),
            registry_digest,
        ]

    @property
    def canonical(self) -> list:
        return self._canonical(
            self.case_ref, self.entered_review_at,
            Policy(self.selected_version, self.max_score, self.required_docs,
                   self.entered_review_at),
            self.registry_digest,
        )


def _registry_rows(policies: Iterable[Policy]) -> list[list]:
    rows = sorted(policies, key=lambda p: (_utc(p.effective_at), p.version))
    return [
        [p.version, p.max_score, sorted(p.required_docs),
         _utc(p.effective_at).isoformat()]
        for p in rows
    ]


def freeze_policy(case_ref: str, entered_review_at: datetime,
                  policies: Iterable[Policy]) -> PolicySnapshot:
    """Freeze the rule governing a case at its review-entry instant."""
    at = _utc(entered_review_at)
    rows = sorted(policies, key=lambda p: (_utc(p.effective_at), p.version))
    registry_digest = _digest(_registry_rows(rows))
    probe = Case(at, 0.0, frozenset())
    selected = select(probe, rows, at=at)
    if not governs(selected, at):
        raise ValueError("selected policy is not effective at review entry")
    canonical = PolicySnapshot._canonical(case_ref, at, selected, registry_digest)
    return PolicySnapshot(
        case_ref, at, selected.version, selected.max_score,
        frozenset(selected.required_docs), registry_digest, _digest(canonical),
    )


def policy_registry_digest(policies: Iterable[Policy]) -> str:
    return _digest(_registry_rows(policies))


def applies_to(policy: Policy, entered_review_at: datetime) -> bool:
    """True iff a (new) rule is in force for a case entering review at
    ``entered_review_at``. Rules never reach back before their effective
    time, so new rules apply only to new cases."""
    return governs(policy, entered_review_at)


# --- Evidence snapshot ------------------------------------------------------

@dataclass(frozen=True)
class EvidenceSnapshot:
    """Frozen evidence closure for a case: the asserted edges, the seeds the
    closure started from and the reachable evidence set at bind time."""
    case_ref: str
    seeds: frozenset[str]
    edges: tuple[CaseEdge, ...]
    reached: frozenset[str]
    max_depth: int | None
    digest: str

    @property
    def canonical(self) -> list:
        return [
            "evidence-snapshot",
            self.case_ref,
            sorted(self.seeds),
            sorted(
                [e.source, e.target, e.weight, e.kind] for e in sorted(
                    self.edges, key=lambda e: (e.source, e.target, e.kind, e.weight)
                )
            ),
            self.max_depth,
            sorted(self.reached),
        ]


def freeze_evidence(case_ref: str, seeds: Iterable[str],
                    edges: Iterable[CaseEdge],
                    max_depth: int | None = None) -> EvidenceSnapshot:
    edge_rows = tuple(edges)
    seed_rows = frozenset(seeds)
    reached = frozenset(reachable(seed_rows, edge_rows, max_depth=max_depth))
    snap = EvidenceSnapshot(case_ref, seed_rows, edge_rows, reached,
                            max_depth, "")
    return replace(snap, digest=_digest(snap.canonical))


# --- Bound case state -------------------------------------------------------

@dataclass(frozen=True)
class CaseBinding:
    """A quality case bound to the exact rule snapshot and evidence snapshot
    under which it was decided. ``digest`` chains every field, so any
    mutation (of the rule threshold, the evidence, the stored conclusion or
    the state) invalidates the binding and is caught on replay."""
    reference: str
    case: Case
    state: str
    policy_snapshot: PolicySnapshot
    evidence_snapshot: EvidenceSnapshot
    decision: bool | None
    revision: int
    updated_at: datetime
    digest: str

    def validate(self) -> None:
        if self.state not in REVIEW_STATES:
            raise ValueError("unknown review state")
        if self.revision < 0:
            raise ValueError("negative revision")
        _utc(self.updated_at)
        if self.policy_snapshot.case_ref != self.reference:
            raise ValueError("policy snapshot belongs to another case")
        if self.evidence_snapshot.case_ref != self.reference:
            raise ValueError("evidence snapshot belongs to another case")

    @property
    def canonical(self) -> list:
        return [
            "case-binding",
            self.reference,
            _utc(self.case.received_at).isoformat(),
            self.case.score,
            sorted(self.case.documents),
            self.state,
            self.policy_snapshot.digest,
            self.evidence_snapshot.digest,
            self.decision,
            self.revision,
            _utc(self.updated_at).isoformat(),
        ]


def bind(reference: str, case: Case, policies: Iterable[Policy],
         edges: Iterable[CaseEdge] = (), seeds: Iterable[str] | None = None,
         max_depth: int | None = None, *, at: datetime | None = None) -> CaseBinding:
    """Freeze rule snapshot + evidence snapshot onto a case as it enters
    review. Evidence seeds default to the documents the case actually
    carries."""
    entered_at = _utc(at if at is not None else case.received_at)
    policies = tuple(policies)
    policy_snap = freeze_policy(reference, entered_at, policies)
    evidence_snap = freeze_evidence(
        reference,
        frozenset(case.documents) if seeds is None else seeds,
        tuple(edges), max_depth,
    )
    binding = CaseBinding(
        reference, case, "in_review", policy_snap, evidence_snap,
        None, 0, entered_at, "",
    )
    binding.validate()
    return replace(binding, digest=_digest(binding.canonical))


def evaluate_snapshot(case: Case, snapshot: PolicySnapshot) -> bool:
    """Decide a case strictly against its frozen rule snapshot."""
    return (case.score <= snapshot.max_score
            and snapshot.required_docs.issubset(case.documents))


@dataclass(frozen=True)
class Adjudication:
    binding: CaseBinding
    event: "CaseTransition"


@dataclass(frozen=True)
class CaseTransition:
    reference: str
    before: str
    after: str
    revision: int
    actor: str
    at: datetime
    decision: bool | None
    reason: str
    binding_digest: str


_STATE_TRANSITIONS = {
    "in_review": frozenset({"adjudicated"}),
    "adjudicated": frozenset({"reopened", "closed"}),
    "reopened": frozenset({"adjudicated", "closed"}),
    "closed": frozenset(),
}


def _next(binding: CaseBinding, target: str, actor: str, at: datetime,
          decision: bool | None, reason: str) -> Adjudication:
    binding.validate()
    if target not in _STATE_TRANSITIONS.get(binding.state, frozenset()):
        raise ValueError(f"transition {binding.state} -> {target} not allowed")
    when = _utc(at)
    if when < _utc(binding.updated_at):
        raise ValueError("event time moved backward")
    nxt = replace(
        binding,
        state=target,
        decision=decision,
        revision=binding.revision + 1,
        updated_at=when,
    )
    nxt.validate()
    event = CaseTransition(
        binding.reference, binding.state, target, nxt.revision,
        actor, when, decision, reason, binding.digest,
    )
    return Adjudication(replace(nxt, digest=_digest(nxt.canonical)), event)


def adjudicate(binding: CaseBinding, actor: str, at: datetime,
               reason: str = "") -> Adjudication:
    """Adjudicate using the bound rule snapshot only. The conclusion is
    computed once, from the frozen rule and the bound evidence, and stored
    on the binding; later rule changes cannot touch it."""
    allowed = evaluate_snapshot(binding.case, binding.policy_snapshot)
    return _next(binding, "adjudicated", actor, at, allowed, reason)


def reopen(binding: CaseBinding, actor: str, at: datetime,
           reason: str = "") -> Adjudication:
    """Reopen a case for further work. The original rule snapshot remains
    bound (the case is re-examined under the rules it entered review with);
    the recorded decision is cleared until re-adjudication."""
    return _next(binding, "reopened", actor, at, None, reason)


def close(binding: CaseBinding, actor: str, at: datetime,
          reason: str = "") -> Adjudication:
    if binding.decision is None:
        raise ValueError("cannot close a case without a recorded decision")
    return _next(binding, "closed", actor, at, binding.decision, reason)


def replay(binding: CaseBinding, events: Iterable[CaseTransition]) -> CaseBinding:
    """Rebuild case state from its transition log. Every event must chain to
    the previous binding digest and replay to the same stored conclusion;
    otherwise the history was tampered with."""
    state = binding
    for event in sorted(events, key=lambda e: e.revision):
        if event.reference != state.reference:
            raise ValueError("mixed references")
        if event.revision != state.revision + 1:
            raise ValueError("revision gap")
        if event.binding_digest != state.digest:
            raise ValueError("event does not chain to case state")
        if event.before != state.state:
            raise ValueError("event before-state mismatch")
        if event.after not in _STATE_TRANSITIONS.get(state.state, frozenset()):
            raise ValueError("event contains illegal transition")
        state = replace(
            state, state=event.after, decision=event.decision,
            revision=event.revision, updated_at=_utc(event.at),
        )
        state.validate()
        state = replace(state, digest=_digest(state.canonical))
    return state


# --- Historical look-back ---------------------------------------------------

class SnapshotError(ValueError):
    """The bound snapshots do not match the recorded history."""


class DecisionDrift(SnapshotError):
    """Re-evaluation under the *bound* (historical) rule no longer yields the
    stored conclusion. Raised rather than silently rewriting the decision."""


def verify_binding(binding: CaseBinding, policies: Iterable[Policy] | None = None,
                   ) -> None:
    """Check integrity of a bound case.

    * binding digest chains rule snapshot, evidence snapshot, state;
    * evidence closure recomputes to the frozen closure;
    * the frozen policy snapshot still matches the policy registry (if
      supplied) — a post-hoc rule edit is drift, not a new decision;
    * the stored decision still follows from the frozen rule.
    """
    if _digest(binding.canonical) != binding.digest:
        raise SnapshotError("case binding digest mismatch")
    ps = binding.policy_snapshot
    if _digest(ps.canonical) != ps.digest:
        raise SnapshotError("policy snapshot digest mismatch")
    es = binding.evidence_snapshot
    if _digest(es.canonical) != es.digest:
        raise SnapshotError("evidence snapshot digest mismatch")
    if frozenset(reachable(es.seeds, es.edges, es.max_depth)) != es.reached:
        raise SnapshotError("evidence closure no longer matches snapshot")
    if policies is not None:
        rows = tuple(policies)
        # Re-selecting at the historical entry instant must reproduce the
        # frozen snapshot. Appending rules effective *after* that instant is
        # fine (they govern only later cases); back-dating or rewriting a
        # rule so it would change this case is drift, never an auto-update.
        try:
            historical = freeze_policy(binding.reference,
                                       ps.entered_review_at, rows)
        except ValueError as exc:
            raise SnapshotError(
                f"rule of record missing for case {binding.reference}"
            ) from exc
        if historical.selected_version != ps.selected_version \
                or (historical.max_score, historical.required_docs) \
                != (ps.max_score, ps.required_docs):
            raise SnapshotError("rule of record no longer matches snapshot")
    if binding.state in ("adjudicated", "closed"):
        if binding.decision is None:
            raise SnapshotError("decided case has no recorded decision")
        if evaluate_snapshot(binding.case, ps) != binding.decision:
            raise DecisionDrift(
                f"case {binding.reference}: stored decision "
                f"{binding.decision} disagrees with its bound rule snapshot"
            )


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a historical look-back.

    ``decision`` is always the bound historical conclusion; it is never
    overwritten. ``current_decision`` (only present in ``report`` mode) shows
    what a *new* case with the same facts would decide under today's rules,
    and ``drifted`` flags the discrepancy without changing anything.
    """
    reference: str
    mode: str
    decision: bool | None
    state: str
    snapshot_version: int
    current_version: int | None = None
    current_decision: bool | None = None
    drifted: bool = False


def review_history(binding: CaseBinding,
                   current_policies: Iterable[Policy] | None = None,
                   *, mode: str = "locked",
                   strict: bool = True,
                   now: datetime | None = None) -> ReplayResult:
    """Look back at a historical case.

    * ``locked`` (default): evaluate only the bound snapshot. Any integrity
      violation or disagreement with the stored conclusion raises
      :class:`DecisionDrift` (when ``strict``) and never rewrites history.
    * ``report``: additionally compare against the rule in force at ``now``
      (default: current time), purely as an informational what-if. The
      discrepancy is reported via ``drifted``; the historical decision stays
      authoritative and unchanged.
    """
    if mode not in ("locked", "report"):
        raise ValueError("mode must be 'locked' or 'report'")
    try:
        verify_binding(binding, current_policies if mode == "locked" else None)
    except DecisionDrift:
        if strict:
            raise

    current_version: int | None = None
    current_decision: bool | None = None
    drifted = False
    if mode == "report" and current_policies is not None:
        rows = tuple(current_policies)
        at_now = _utc(now) if now is not None else datetime.now(timezone.utc)
        selected_now = select(binding.case, rows, at=at_now)
        current_version = selected_now.version
        current_decision = (
            binding.case.score <= selected_now.max_score
            and selected_now.required_docs.issubset(binding.case.documents)
        )
        drifted = (
            binding.decision is not None
            and selected_now.version != binding.policy_snapshot.selected_version
            and current_decision != binding.decision
        )
    return ReplayResult(
        binding.reference, mode, binding.decision, binding.state,
        binding.policy_snapshot.selected_version,
        current_version, current_decision, drifted,
    )
