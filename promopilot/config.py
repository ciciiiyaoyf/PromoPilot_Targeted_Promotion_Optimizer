"""Current modeling and campaign decisions, without filesystem side effects."""
from dataclasses import dataclass
import math

ACTIONS = ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]


@dataclass(frozen=True)
class PipelineConfig:
    budget: float = 62_000.0
    max_offer_share: float = 0.60
    max_pct20_share: float = 0.15
    control_share: float = 0.05
    seed: int = 42
    outcome_days: int = 14
    free_shipping_cutoff: float = 75.0
    standard_shipping_fee: float = 5.99
    development_campaigns: tuple[str, ...] = ("CP04", "CP05", "CP06")
    evaluation_campaign: str = "CP07"
    cost_selection_campaign: str = "CP06"
    bootstrap_draws: int = 2000

    def __post_init__(self):
        if not math.isfinite(self.budget) or self.budget < 0:
            raise ValueError("Budget must be finite and nonnegative")
        for name in ("max_offer_share", "max_pct20_share", "control_share"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.control_share < .05:
            raise ValueError("This project requires at least 5% measurement controls")
        if self.outcome_days != 14 or self.bootstrap_draws < 2:
            raise ValueError("This project uses 14-day outcomes; bootstrap draws must be at least two")
        if not math.isfinite(self.free_shipping_cutoff) or self.free_shipping_cutoff <= 0:
            raise ValueError("Free shipping cutoff must be positive")
        if not math.isfinite(self.standard_shipping_fee) or self.standard_shipping_fee < 0:
            raise ValueError("Standard shipping fee must be nonnegative")
