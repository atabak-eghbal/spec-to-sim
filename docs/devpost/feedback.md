# Devpost: feedback answers (for "Most Valuable Feedback")

The form has 9 feedback questions (see your screenshots). These answers are grouped by topic; map
each to the matching question. Every point comes from what actually happened while building, with
numbers from our logs (`.cache/spend.jsonl`, 1,413 calls, $5.00 total).

## Nemotron models: what worked
- **Nemotron 3 Super is the right default for structured planning.** With strict JSON schema it reliably produced nested plans we validate with Pydantic (799 calls, median 9.6 s, $2.33 in total). It was cheap enough to sample three repairs in parallel and let a simulator choose, which matched escalating to Ultra at lower cost.
- **Reasoning is worth it for planning.** With Super's reasoning off, success on our 11 development prompts fell from 11/11 to 6/11 and the median cost per task doubled ($0.0055 → $0.0108), because failed plans need extra repairs. Please keep the per-request `enable_thinking` switch; it is one of the most useful knobs.
- **Nemotron 3.5 Super VL with reasoning off is fast** (median 1.9 s, versus ~25 s with reasoning on) and good at describing a contact sheet of frames.
- **Nano is a good extractor** for numbers and quotes from web pages ($0.11 for 333 calls).

## Nemotron models: what to improve
- **Ultra sometimes ignores strict JSON output.** Even with `response_format: json_schema`, Ultra 550B wrote its reasoning before the JSON in some replies, so we had to recover the last JSON object from the text. Its 67 calls were 48 % of our spend, at a median 32 s and 8,500 output tokens, and its repairs fixed 4 of the 10 runs we escalated. A guaranteed schema mode with reasoning kept separate (e.g. reasoning in its own field, never in the content) would make Ultra usable in pipelines.
- **The VL model mostly restated the numbers we gave it.** When we passed measured failures alongside frames, its diagnoses leaned on the numbers rather than the images; in our ablation it made no measurable difference to repair success. Clear guidance on prompting VL models to describe frames first, or a "vision-only" mode, would help.
- **Expected spatial conventions:** the VL model gave a left/right fix in the wrong axis until we stated the coordinate frame in the prompt. A short doc on how Nemotron VL handles 3D frames and coordinates would save others the debugging.

## Nebius Token Factory: what worked
- **One OpenAI-compatible client for every model** made swapping model sizes a one-word change.
- **Strict JSON schema + per-request reasoning control** were the two features we leaned on most.
- **Pricing per token is low enough** that a full 104-run evaluation cost $0.94.

## Nebius Token Factory and AI Cloud: what to improve
- **Spending limits:** we built our own daily cap from logged token counts. A per-key hard daily limit with an alert, set in the console and enforced server-side, is the single feature we missed most for a public demo.
- **Cost in the API response:** returning the cost (or the price per token) with each response would remove the need to hard-code prices from listings, which we couldn't verify.
- **Model IDs:** model names differ between third-party listings and the API; we had to call `GET /v1/models` to find them. A stable, documented ID list with aliases would help.
- **AI Cloud onboarding:** with hackathon credits, `registry create` and `endpoint create --dry-run` returned `PermissionDenied`, with no message saying which permission or billing step was missing. We couldn't deploy the Serverless Endpoint and Job we had packaged, so the demo runs elsewhere. An actionable error ("activate billing in project X" or "your role lacks Y") would have saved hours.
- **Latency visibility:** Nano's median latency in our logs was 7 s for long inputs; a time-to-first-token and queue-time breakdown in the response would help tune pipelines.

## Developer experience suggestions
- A worked example of a **verifier-in-the-loop pattern** (sample N candidates, score with an external checker, keep the best) would showcase where small Nemotron models shine.
- A **"structured output with reasoning" cookbook** covering when to turn reasoning on, token budgets, and recovering JSON.
- **LangSmith or OpenTelemetry tracing examples** for Token Factory, since tracing made our cascade debuggable.

## Tavily (if asked)
- `/extract` with a `query` returning only the relevant chunks was ideal for reading engineering tables.
- An option to return **numbers with their units already parsed**, and a **date and source type** per result, would make grounding pipelines simpler.
