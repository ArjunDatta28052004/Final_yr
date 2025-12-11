from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag_engine_free import PolicyRAGEngine
from pydantic import BaseModel
import os
import shutil
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
INGESTED_FILEPATH = None

class ValidateRequest(BaseModel):
    claim_text: str

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Accepts a multipart file upload (PDF), saves it locally on the backend,
    then runs ingestion (PolicyRAGEngine).
    """
    global engine, INGESTED_FILEPATH

    # Ensure upload folder exists
    os.makedirs("uploaded_policies", exist_ok=True)

    # Save file with a unique name
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    saved_path = os.path.join("uploaded_policies", unique_name)

    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        return {"error": f"Failed to save file: {str(e)}"}

    try:
        engine = PolicyRAGEngine(saved_path)
        pages, chunks = engine.ingest_policy_document()
        INGESTED_FILEPATH = saved_path
        return {
            "message": "Ingested successfully",
            "pages": pages,
            "chunks": chunks,
            "summary": engine.get_policy_summary()
        }
    except Exception as e:
        return {"error": f"Ingestion failed: {str(e)}"}

@app.post("/validate")
def validate_claim(req: ValidateRequest):
    global engine
    if engine is None:
        return {"error": "Policy not ingested"}

    try:
        result = engine.validate_claim(req.claim_text)
        return result
    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}
