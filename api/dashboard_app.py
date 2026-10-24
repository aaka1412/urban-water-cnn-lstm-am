"""ASGI entry point for the dashboard.

Wraps the ORIGINAL FastAPI app (api/main.py, unchanged) and adds the dashboard router, so the existing
chatbot endpoints (/api/chat, /api/runs, ...) keep working exactly as before.

  uvicorn api.dashboard_app:app --port 8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.main import app  # original app, untouched
from dashboard import config, db
from dashboard.routes import model_unavailable_handler, router
from dashboard.service import ModelUnavailable

db.init_db()
app.title = "Urban Water Consumption Prediction and Analytics System"
app.include_router(router)
app.add_exception_handler(ModelUnavailable, model_unavailable_handler)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


# Optional: serve the production Angular build from the same port (ng build -> frontend/dist/frontend/browser).
DIST = config.ROOT / "frontend" / "dist" / "frontend" / "browser"
if DIST.is_dir():
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not found."}, status_code=404)
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
