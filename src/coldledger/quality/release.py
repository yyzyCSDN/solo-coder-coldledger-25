from dataclasses import dataclass
from datetime import datetime,timezone
@dataclass(frozen=True)
class Policy: version:int; max_score:float; required_docs:frozenset; effective_at:datetime
@dataclass(frozen=True)
class Case: received_at:datetime; score:float; documents:frozenset
def _utc(v:datetime)->datetime:
    return v.astimezone(timezone.utc) if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
def select(case,policies,at=None):
    # Rule selection is pinned to the moment the case entered review (never to
    # wall-clock now), so re-reviewing a historical case after a new rule took
    # effect cannot silently re-decide it under the new rule.
    x=_utc(at if at is not None else case.received_at)
    rows=[p for p in policies if _utc(p.effective_at)<=x]
    if not rows:raise ValueError('no effective policy')
    rows.sort(key=lambda p:(_utc(p.effective_at),p.version))
    return rows[-1]
def governs(policy,at:datetime)->bool:
    """A release rule governs a case iff it was already in force when the
    case entered review. A newer rule therefore applies only to newer cases."""
    return _utc(policy.effective_at)<=_utc(at)
def may_release(case,policies,at=None):
    p=select(case,policies,at);return case.score<=p.max_score and p.required_docs.issubset(case.documents)
