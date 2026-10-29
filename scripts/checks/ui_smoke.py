"""UI smoke check (needs the app on :8000 and Playwright): click every mode, attempt tab and the ghost button on several recorded runs; report page errors and failed requests."""
from playwright.sync_api import sync_playwright

runs = ["", "?run=20261028-211134-04da", "?run=20261028-180549-7019", "?run=20261028-142918-490a", "?run=20261028-151115-e705"]
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    for r in runs:
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("pageerror", lambda e, r=r: errs.append(f"{r or '/'}: {e}"))
        pg.on("response", lambda resp, r=r: errs.append(f"{r or '/'}: HTTP {resp.status} {resp.url}") if resp.status >= 400 and "localhost" in resp.url else None)
        pg.goto("http://localhost:8000/" + r); pg.wait_for_timeout(4500)
        for sel in ["#tabs [data-att]", "#modes button[data-mode]:not([hidden])", "#ghost-btn:not([hidden])", "#tabs [data-att]"]:
            for i in range(len(pg.query_selector_all(sel))):
                try:
                    pg.query_selector_all(sel)[i].click(); pg.wait_for_timeout(700)
                except Exception as e:
                    errs.append(f"{r}: click {sel}: {e}"[:160])
        pg.close()
    b.close()
print("\n".join(errs) or "no errors")
