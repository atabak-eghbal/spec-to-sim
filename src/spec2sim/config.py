"""Central config: model IDs, prices, endpoints and budget limits, loaded from `.env`.

Every pipeline step imports its model ID and per-token price from here, so the UI can show
which model ran each step and what it cost.
Model IDs verified against Token Factory `GET /v1/models` on 2026-10-28.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str  # short badge name shown in the UI
    usd_in_per_m: float  # $ per 1M input tokens
    usd_out_per_m: float  # $ per 1M output tokens

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_in_per_m + tokens_out * self.usd_out_per_m) / 1e6


# Prices from docs/research/03-models.md (third-party listings) - recheck in the Token Factory console.
NANO = ModelInfo(os.getenv("MODEL_NANO", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"), "Nano", 0.06, 0.24)
SUPER = ModelInfo(os.getenv("MODEL_SUPER", "nvidia/nemotron-3-super-120b-a12b"), "Super", 0.30, 0.90)
VL = ModelInfo(os.getenv("MODEL_VL", "nvidia/Nemotron-3.5-Super-VL"), "Super VL", 0.30, 0.90)
ULTRA = ModelInfo(os.getenv("MODEL_ULTRA", "nvidia/Nemotron-3-Ultra-550b-a55b"), "Ultra", 1.00, 3.00)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "4"))
DEV_SPEND_LIMIT_USD = float(os.getenv("DEV_SPEND_LIMIT_USD", "20"))
