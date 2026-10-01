from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Policy:
    version: int
    max_score: float
    required_docs: frozenset
    effective_at: datetime


@dataclass(frozen=True)
class Case:
    received_at: datetime
    score: float
    documents: frozenset
    # When the case entered quality review. Rules are pinned to this instant so
    # that later rule versions cannot change the outcome of a historical case.
    # Falls back to received_at for cases that were never explicitly opened.
    review_opened_at: datetime | None = None

    def decision_anchor(self) -> datetime:
        anchor = self.review_opened_at or self.received_at
        if anchor.tzinfo is None or anchor.utcoffset() is None:
            raise ValueError("timezone-aware datetime required")
        return anchor.astimezone(timezone.utc)


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError("timezone-aware datetime required")
    return v.astimezone(timezone.utc)


def select_for(case: Case, policies, at: datetime | None = None) -> Policy:
    """Select the rule version that governed the case at a fixed point in time.

    ``at`` defaults to the case's review-entry instant, never the wall clock.
    This is what keeps historical decisions stable when newer rules are added.
    """
    anchor = _utc(at) if at is not None else case.decision_anchor()
    rows = [p for p in policies if _utc(p.effective_at) <= anchor]
    if not rows:
        raise ValueError("no effective policy at case review time")
    rows.sort(key=lambda p: (_utc(p.effective_at), p.version))
    return rows[-1]


# Backwards-compatible name.
select = select_for


@dataclass(frozen=True)
class ReleaseDecision:
    allowed: bool
    policy_version: int
    max_score: float
    required_docs: frozenset
    missing_docs: frozenset = field(default_factory=frozenset)
    anchored_at: datetime | None = None

    def __bool__(self) -> bool:
        return self.allowed


def evaluate(case: Case, policies, at: datetime | None = None) -> ReleaseDecision:
    p = select_for(case, policies, at)
    missing = frozenset(d for d in p.required_docs if d not in case.documents)
    return ReleaseDecision(
        allowed=case.score <= p.max_score and not missing,
        policy_version=p.version,
        max_score=p.max_score,
        required_docs=frozenset(p.required_docs),
        missing_docs=missing,
        anchored_at=(at if at is not None else case.decision_anchor()),
    )


def may_release(case: Case, policies, at: datetime | None = None) -> bool:
    return bool(evaluate(case, policies, at))
