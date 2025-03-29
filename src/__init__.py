from src.lightning_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)
