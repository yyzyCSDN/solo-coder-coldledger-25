from dataclasses import dataclass
from datetime import datetime,timezone
@dataclass(frozen=True)
class Policy: version:int; max_score:float; required_docs:frozenset; effective_at:datetime
@dataclass(frozen=True)
class Case: received_at:datetime; score:float; documents:frozenset
def select(case,policies):
    at=datetime.now(timezone.utc)
    rows=[p for p in policies if p.effective_at.astimezone(timezone.utc)<=at]
    if not rows:raise ValueError('no effective policy')
    rows.sort(key=lambda p:(p.effective_at.astimezone(timezone.utc),p.version))
    return rows[-1]
def may_release(case,policies):
    p=select(case,policies);return case.score<=p.max_score and p.required_docs.issubset(case.documents)
