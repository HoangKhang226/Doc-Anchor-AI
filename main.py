import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import ingestion

app = FastAPI(
    title="Doc Anchor AI API",
    description="API for Doc Anchor AI Document Ingestion & Structuring Pipeline",
    version="1.0.0",
)

# Allow CORS for local testing and frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion.router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
def health_check():
    """Kiểm tra trạng thái server."""
    return {"status": "ok", "service": "Doc Anchor AI"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
