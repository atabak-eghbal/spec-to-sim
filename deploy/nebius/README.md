# Deploying on Nebius AI Cloud

One image, two Nebius services:
- **Serverless Endpoint**: the web app (public demo URL). CPU-only; MuJoCo renders with OSMesa.
- **Serverless Job**: the batch eval (`scripts/eval.py`), printing a success-rate table to the job logs.

API keys are never baked into the image or committed. They are passed as environment variables
when the endpoint/job is created (read from your local `.env`).

## 0. One-time setup (done)
```bash
nebius profile create spec2sim            # tenant sapphire-llama-tenant-7x2
nebius registry configure-helper          # docker can push to cr.<region>.nebius.cloud
```
Region: **us-north1**, project `project-u02pyk0wig0082n0cwnqda` (CPU quota 200 vCPU),
subnet `vpcsubnet-u02ypft8s5q07tntvm`.

## 1. Build and push (amd64)
```bash
REGISTRY_ID=<from: nebius registry create --name spec2sim --parent-id project-u02pyk0wig0082n0cwnqda>
IMAGE=cr.us-north1.nebius.cloud/$REGISTRY_ID/spec2sim:$(git rev-parse --short HEAD)
docker buildx build --platform linux/amd64 -t $IMAGE --push .
```
Note: the amd64 image cannot be smoke-tested on an Apple-silicon Mac. MuJoCo's x86 build needs
AVX, which Docker's emulation lacks (the process is killed with exit 137). Test the same
Dockerfile natively with `--platform linux/arm64` instead.

## 2. Endpoint (web app) — billed per second while running
```bash
set -a; source .env; set +a
nebius ai endpoint create \
  --parent-id project-u02pyk0wig0082n0cwnqda --name spec2sim \
  --image $IMAGE --platform cpu-d3 --preset 4vcpu-16gb \
  --container-port 8000 --public --disk-size 30Gi \
  --subnet-id vpcsubnet-u02ypft8s5q07tntvm \
  --env NEBIUS_API_KEY=$NEBIUS_API_KEY --env TAVILY_API_KEY=$TAVILY_API_KEY \
  --env LANGSMITH_API_KEY=$LANGSMITH_API_KEY --env LANGSMITH_TRACING=true \
  --env LANGSMITH_PROJECT=spec2sim
```
Stop it when idle: `nebius ai endpoint stop <id>` (a stopped endpoint is not billed).

Sizing (measured in the arm64 container): a 7 s clip renders in 12 s on 5 CPUs and 16 s on 2 CPUs,
with shadows off (`SPEC2SIM_SHADOWS=0`, set in the Dockerfile). A 30 s multi-object task takes ~1 min.
4 vCPU allows the app's 2 concurrent runs.

## 3. Job (batch eval)
```bash
nebius ai job create \
  --parent-id project-u02pyk0wig0082n0cwnqda --name spec2sim-eval-$(date +%m%d%H%M) \
  --image $IMAGE --platform cpu-d3 --preset 4vcpu-16gb --disk-size 30Gi \
  --subnet-id vpcsubnet-u02ypft8s5q07tntvm --timeout 1h \
  --container-command python --args "scripts/eval.py --reps 3 --workers 2" \
  --env NEBIUS_API_KEY=$NEBIUS_API_KEY --env TAVILY_API_KEY=$TAVILY_API_KEY \
  --env LANGSMITH_API_KEY=$LANGSMITH_API_KEY --env LANGSMITH_TRACING=true \
  --env LANGSMITH_PROJECT=spec2sim
nebius ai job logs <job-id>     # the summary table is printed at the end
```

## Status (2026-10-28)
Blocked: `registry create` and `endpoint create --dry-run` return PermissionDenied on the tenant.
Reads work, so AI Cloud billing/credits are likely not activated yet (Token Factory credits are separate).
