from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from deepdive import __version__
from deepdive.config import DeepDiveConfig
from deepdive.models import ResearchReport
from deepdive.pipeline import ResearchPipeline
from deepdive.report.markdown import report_to_markdown

app = FastAPI(title="DeepDive", version=__version__)


class ResearchRequest(BaseModel):
    question: str


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "version": __version__}


@app.post("/api/research")
async def research(req: ResearchRequest) -> EventSourceResponse:
    pipeline = ResearchPipeline(config=DeepDiveConfig())

    async def event_stream():
        try:
            async for event in pipeline.run(req.question):
                yield {"event": event.type, "data": json.dumps(event.data)}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_stream())


@app.post("/api/research/markdown", response_class=PlainTextResponse)
async def research_markdown(req: ResearchRequest) -> str:
    """Run research end-to-end and return the report as Markdown (no SSE)."""
    pipeline = ResearchPipeline(config=DeepDiveConfig())
    report: ResearchReport | None = None
    async for event in pipeline.run(req.question):
        if event.type == "done":
            report = ResearchReport.model_validate(event.data["report"])
    if report is None:
        return "# Research failed\n\nNo report was generated."
    return report_to_markdown(report)


def run() -> None:
    import uvicorn

    cfg = DeepDiveConfig()
    uvicorn.run(app, host=cfg.host, port=cfg.port)
