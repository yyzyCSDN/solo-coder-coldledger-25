from datetime import timedelta
from .inventory.fefo import reserve
from .quality.release import may_release
from .quality.review_packet import open_review, replay, compare_with_current
from .transport.connections import choose
class ColdLedgerService:
    def reserve_inventory(self,*args,**kwargs): return reserve(*args,**kwargs)
    def release_decision(self,*args,**kwargs): return may_release(*args,**kwargs)
    def open_quality_review(self,*args,**kwargs): return open_review(*args,**kwargs)
    def replay_quality_review(self,*args,**kwargs): return replay(*args,**kwargs)
    def compare_review_with_current_rules(self,*args,**kwargs): return compare_with_current(*args,**kwargs)
    def next_connection(self,inbound,candidates,minutes): return choose(inbound,candidates,timedelta(minutes=minutes))
    def health(self): return {'service':'coldledger','status':'ok','runtime_dependencies':0}
