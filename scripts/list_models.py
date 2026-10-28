"""Smoke test: list Token Factory models and print the NVIDIA/Nemotron ones.

Run: uv run python scripts/list_models.py
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
    api_key=os.environ["NEBIUS_API_KEY"],
)
ids = sorted(m.id for m in client.models.list().data)
nv = [i for i in ids if "nemotron" in i.lower() or i.lower().startswith("nvidia/")]
print(f"{len(ids)} models total; NVIDIA/Nemotron:")
for i in nv:
    print("  ", i)
