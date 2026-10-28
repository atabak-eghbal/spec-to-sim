# Tavily + LangSmith

## Tavily (web search for agents)
- Docs: https://docs.tavily.com/
- Python: `pip install tavily-python` → `TavilyClient(api_key=...).search(query, max_results=5)`
- Bonus prize requires a **functional runtime call** — ours happens on every run (Grounder step).
- 📌 Use it where it changes the output: real object sizes/masses ("standard coffee mug dimensions",
  "Rubik's cube size and weight") and real robot facts (payload, reach). Show sources as chips in the UI.
- 📌 Cache by query hash in `.cache/tavily/` to protect the $25 credit.

## LangSmith (tracing)
- Docs: https://docs.smith.langchain.com/
- Env: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
- `from langsmith.wrappers import wrap_openai` → wrap the Token Factory client; `@traceable` on each pipeline step.
- 📌 Each run in the UI links to its LangSmith trace. Add a screenshot of the trace tree to the README.
