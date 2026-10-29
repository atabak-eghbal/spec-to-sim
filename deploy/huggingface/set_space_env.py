"""Copy the Space's settings from .env to a Hugging Face Space (no values are ever printed).

    uvx --from huggingface_hub python deploy/huggingface/set_space_env.py AustinTransformer/spec-to-sim
    uvx --from huggingface_hub python deploy/huggingface/set_space_env.py AustinTransformer/spec-to-sim --secrets
    uvx --from huggingface_hub python deploy/huggingface/set_space_env.py AustinTransformer/spec-to-sim --share-traces

Variables (visible in the Space settings) are always set. API keys are sent as Space *secrets* only
with --secrets. --share-traces adds SPEC2SIM_SHARE_TRACES=1 (submission day). Needs a prior
`hf auth login` with a write token.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
VARIABLES = ["NEBIUS_BASE_URL", "MODEL_NANO", "MODEL_SUPER", "MODEL_VL", "LANGSMITH_PROJECT"]
FIXED = {"LANGSMITH_TRACING": "true", "DAILY_SPEND_LIMIT_USD": "3"}
SECRETS = ["NEBIUS_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"]


def read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.split(" #")[0].strip().strip('"').strip("'")
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("space", help="e.g. AustinTransformer/spec-to-sim")
    ap.add_argument("--secrets", action="store_true", help="also send the API keys as Space secrets")
    ap.add_argument("--share-traces", action="store_true", help="also set SPEC2SIM_SHARE_TRACES=1")
    args = ap.parse_args()
    env, api = read_env(ROOT / ".env"), HfApi()
    values = {k: env[k] for k in VARIABLES if env.get(k)} | FIXED
    if args.share_traces:
        values["SPEC2SIM_SHARE_TRACES"] = "1"
    for k, v in values.items():
        api.add_space_variable(args.space, k, v)
        print(f"variable {k}: set")
    for k in VARIABLES:
        if not env.get(k):
            print(f"variable {k}: missing from .env, skipped (the app's default is used)")
    if args.secrets:
        for k in SECRETS:
            if env.get(k):
                api.add_space_secret(args.space, k, env[k])
                print(f"secret {k}: set")
            else:
                print(f"secret {k}: missing from .env, skipped")
    else:
        print("secrets: not sent (add --secrets to send " + ", ".join(SECRETS) + ")")


if __name__ == "__main__":
    main()
