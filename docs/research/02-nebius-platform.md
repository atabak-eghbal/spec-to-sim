# Nebius Platform

## Token Factory (inference API) ✅
- Docs: https://docs.tokenfactory.nebius.com/ · Quickstart: https://docs.tokenfactory.nebius.com/quickstart
- Cookbook: https://github.com/nebius/token-factory-cookbook
- OpenAI-compatible: `base_url="https://api.tokenfactory.nebius.com/v1/"`, key in `NEBIUS_API_KEY`.
- 📌 One `openai.OpenAI` client, swap `model=` per pipeline step → the cascade is trivial to implement.

## Serverless AI — Endpoints and Jobs ✅
- Overview: https://docs.nebius.com/serverless/overview
- Endpoints quickstart: https://docs.nebius.com/serverless/quickstart/endpoints
- Manage endpoints: https://docs.nebius.com/serverless/endpoints/manage
- Jobs quickstart: https://docs.nebius.com/serverless/quickstart/jobs
- Manage jobs: https://docs.nebius.com/serverless/jobs/manage
- Launch blog (Jobs support CPU-only resources): https://nebius.com/blog/posts/introducing-serverless
- Facts:
  - Endpoints and jobs are containers running over VMs; billed **per second while active**.
  - Endpoint = persistent, accepts requests until stopped/deleted. Job = runs to completion, then releases resources.
  - CLI: `nebius ai endpoint create --image ... --container-port 8000 --public --auth token --token ... --subnet-id ...`
    then `nebius ai endpoint get/logs/start/delete`.
  - CLI: `nebius ai job create --name ... --image ... --container-command ... --args ... --preset ... --timeout 1h`.
  - Service is in Preview.
- 📌 App + simulator ship as **one container** deployed as an Endpoint (demo URL).
  Batch evaluation of ~20 prompts runs as a **Job**.

## ⚠️ Open question: cost of keeping the endpoint alive for judging
Rules say judges get free access until **Dec 15, 2026**. An always-on endpoint for ~7 weeks costs money.
- 🔎 Check the smallest CPU preset price and whether we have AI Cloud credits.
- Options: (a) keep it running, (b) host the UI on a free host that still calls Token Factory at runtime
  and keep the Nebius Endpoint for the simulator only during submission, (c) "start on request" note in README.
- Judges may judge from the video alone, so the video must stand on its own.
