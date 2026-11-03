import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.logistics.api import router as logistics_router
from src.optimization.api import router as phase2_router
from src.web.api.pcos_routes import router as pcos_router
from src.web.api.routes import router


WEB_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = WEB_DIR.parents[1]
LOGGER = logging.getLogger("src.web")
APP_PROFILE = os.getenv("APP_PROFILE", "full")
if APP_PROFILE not in ("full", "diagnostics"):
    raise ValueError("APP_PROFILE deve ser 'full' ou 'diagnostics'.")

app = FastAPI(
    title="Clinical Model Lab",
    description="Dashboards comparativos de modelos de classificação clínica.",
    version="1.1.0",
)
app.include_router(router)
app.include_router(pcos_router)
app.include_router(phase2_router)
if APP_PROFILE == "full":
    app.include_router(logistics_router)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

templates = Jinja2Templates(directory=WEB_DIR / "templates")


@app.middleware("http")
async def request_logging(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        LOGGER.info(
            "http_request method=%s path=%s status=%d duration_ms=%.2f",
            request.method,
            request.url.path,
            status_code,
            (time.perf_counter() - started) * 1000,
        )


@app.get("/healthz")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> JSONResponse:
    required_files = (
        REPOSITORY_ROOT
        / "src"
        / "db"
        / "outputs"
        / "phase2-genetic"
        / "optimization_results.json",
        REPOSITORY_ROOT
        / "src"
        / "db"
        / "outputs"
        / "breast-cancer-wisconsin-data"
        / "08-export-web"
        / "metrics"
        / "metricas_comparativo.json",
        REPOSITORY_ROOT
        / "src"
        / "db"
        / "outputs"
        / "polycystic-ovary-syndrome-pcos"
        / "05-export-web"
        / "metrics"
        / "metricas_web.json",
    )
    if APP_PROFILE == "full":
        required_files += (
            REPOSITORY_ROOT
            / "src"
            / "db"
            / "outputs"
            / "phase2-logistics"
            / "latest_route_plan.json",
        )
    output_dir = REPOSITORY_ROOT / "src" / "db" / "outputs"
    for study, export, charts in (
        ("breast-cancer-wisconsin-data", "08-export-web", "graficos.json"),
        ("polycystic-ovary-syndrome-pcos", "05-export-web", "graficos_web.json"),
    ):
        required_files += (
            output_dir / study / export / "metrics" / charts,
            output_dir / study / export / "artifacts" / "amostras_comparativo.parquet",
            output_dir / study / export / "artifacts" / "predicoes_longas.parquet",
        )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "missing_files": missing},
        )
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"logistics_enabled": APP_PROFILE == "full"},
    )


@app.get("/pcos", response_class=HTMLResponse)
def pcos_dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="pcos.html",
        context={"logistics_enabled": APP_PROFILE == "full"},
    )


def logistics_dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="routes.html",
        context={},
    )


if APP_PROFILE == "full":
    app.add_api_route("/rotas", logistics_dashboard, response_class=HTMLResponse)
