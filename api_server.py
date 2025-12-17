from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rag_engine_free import PolicyRAGEngine
from pydantic import BaseModel
import os
import shutil
import uuid
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Setting up FASTAPI
app = FastAPI(
    title="Insurance Claim Validator API",
    description="AI-powered insurance claim validation using RAG (Optimized)",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
engine = None
INGESTED_FILEPATH = None

class ValidateRequest(BaseModel):
    claim_text: str

class ValidateResponse(BaseModel):
    is_valid: bool
    decision_reason: str
    extracted_entities: dict
    validation_checks: list
    policy_clauses_used: list
    # ADDED: This allows the performance data from the RAG engine to pass to the UI
    performance_metrics: dict 

@app.on_event("startup")
async def startup_event():
    """Pre-warm the LLM on startup for faster first request."""
    logger.info("Starting Insurance Claim Validator API (Optimized)")
    logger.info("Using optimized RAG engine with parallel processing")
    logger.info("Make sure Ollama is running: ollama serve")
    logger.info("Recommended model: ollama pull mistral")

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Insurance Claim Validator API (Optimized)",
        "version": "2.1.0",
        "optimizations": [
            "Parallel processing",
            "Faster model (mistral recommended)",
            "Reduced LLM calls",
            "Cached retrieval",
            "Regex-first extraction"
        ],
        "endpoints": {
            "health": "/",
            "ingest": "/ingest_file",
            "validate": "/validate",
            "reset": "/reset"
        }
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "policy_loaded": engine is not None,
        "policy_file": INGESTED_FILEPATH if INGESTED_FILEPATH else "None",
        "optimization_level": "high"
    }

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and ingest PDF policy document.
    """
    global engine, INGESTED_FILEPATH
    
    logger.info(f"Received file upload: {file.filename}")
    
    if not file.filename.endswith('.pdf'):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    upload_dir = "uploaded_policies"
    os.makedirs(upload_dir, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    saved_path = os.path.join(upload_dir, unique_name)

    try:
        logger.info(f"Saving file to: {saved_path}")
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        file_size = os.path.getsize(saved_path)
        if file_size == 0:
            raise ValueError("Uploaded file is empty")
        
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save file: {str(e)}"}
        )

    try:
        logger.info("Initializing optimized RAG engine...")
        engine = PolicyRAGEngine(saved_path)
        
        logger.info("Starting policy ingestion with parallel processing...")
        pages, chunks = engine.ingest_policy_document()
        
        INGESTED_FILEPATH = saved_path
        summary = engine.get_policy_summary()
        
        return {
            "message": "Policy ingested successfully",
            "pages": pages,
            "chunks": chunks,
            "summary": summary,
            "filename": file.filename,
            "optimization": "enabled"
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        if os.path.exists(saved_path):
            os.remove(saved_path)
        
        return JSONResponse(
            status_code=500,
            content={"error": f"Ingestion failed: {str(e)}"}
        )

@app.post("/validate", response_model=ValidateResponse)
def validate_claim(req: ValidateRequest):
    """
    Validate insurance claim against loaded policy.
    """
    global engine
    
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="No policy loaded. Please upload a policy first using /ingest_file"
        )

    try:
        # The engine.validate_claim() function returns a dict containing 'performance_metrics'
        result = engine.validate_claim(req.claim_text)
        return result
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Validation failed: {str(e)}",
                "is_valid": False,
                "decision_reason": "System error during validation",
                "extracted_entities": {},
                "validation_checks": [],
                "policy_clauses_used": [],
                "performance_metrics": {} # Ensure error response matches schema
            }
        )

@app.post("/reset")
def reset_engine():
    global engine, INGESTED_FILEPATH
    engine = None
    if INGESTED_FILEPATH and os.path.exists(INGESTED_FILEPATH):
        try:
            os.remove(INGESTED_FILEPATH)
        except:
            pass
    INGESTED_FILEPATH = None
    return {"message": "Engine reset successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)