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

@app.on_event("startup")
async def startup_event():
    """Pre-warm the LLM on startup for faster first request."""
    logger.info("🚀 Starting Insurance Claim Validator API (Optimized)")
    logger.info("⚡ Using optimized RAG engine with parallel processing")
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
    
    Returns:
        - message: Success message
        - pages: Number of pages processed
        - chunks: Number of chunks created
        - summary: Policy summary
    """
    global engine, INGESTED_FILEPATH
    
    logger.info(f"📄 Received file upload: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Ensure upload directory exists
    upload_dir = "uploaded_policies"
    os.makedirs(upload_dir, exist_ok=True)

    # Save file with unique name
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    saved_path = os.path.join(upload_dir, unique_name)

    try:
        # Save uploaded file
        logger.info(f"💾 Saving file to: {saved_path}")
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Check file size
        file_size = os.path.getsize(saved_path)
        logger.info(f"✅ File saved. Size: {file_size / 1024:.2f} KB")
        
        if file_size == 0:
            raise ValueError("Uploaded file is empty")
        
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save file: {str(e)}"}
        )

    # Ingest the policy
    try:
        logger.info("🔧 Initializing optimized RAG engine...")
        engine = PolicyRAGEngine(saved_path)
        
        logger.info("📚 Starting policy ingestion with parallel processing...")
        pages, chunks = engine.ingest_policy_document()
        
        logger.info(f"✅ Ingestion complete. Pages: {pages}, Chunks: {chunks}")
        
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
        
        # Clean up the file if ingestion failed
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except:
            pass
        
        return JSONResponse(
            status_code=500,
            content={"error": f"Ingestion failed: {str(e)}"}
        )

@app.post("/validate", response_model=ValidateResponse)
def validate_claim(req: ValidateRequest):
    """
    Validate insurance claim against loaded policy (Optimized).
    
    Args:
        req: ValidateRequest with claim_text
    
    Returns:
        ValidateResponse with validation results
    """
    global engine
    
    logger.info(f"⚡ Received FAST validation request. Claim length: {len(req.claim_text)}")
    
    if engine is None:
        logger.error("Validation attempted without loaded policy")
        raise HTTPException(
            status_code=400,
            detail="No policy loaded. Please upload a policy first using /ingest_file"
        )

    if not req.claim_text or len(req.claim_text.strip()) < 10:
        logger.error("Invalid claim text")
        raise HTTPException(
            status_code=400,
            detail="Claim text must be at least 10 characters long"
        )

    try:
        logger.info("🚀 Starting optimized claim validation...")
        result = engine.validate_claim(req.claim_text)
        logger.info(f"✅ Validation complete. Result: {result.get('is_valid')}")
        
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
                "policy_clauses_used": []
            }
        )

@app.post("/reset")
def reset_engine():
    """
    Reset the engine and clear loaded policy.
    """
    global engine, INGESTED_FILEPATH
    
    logger.info("🔄 Resetting engine...")
    
    old_filepath = INGESTED_FILEPATH
    
    engine = None
    INGESTED_FILEPATH = None
    
    # Optionally delete the uploaded file
    if old_filepath and os.path.exists(old_filepath):
        try:
            os.remove(old_filepath)
            logger.info(f"🗑️ Deleted policy file: {old_filepath}")
        except:
            logger.warning(f"Could not delete file: {old_filepath}")
    
    return {
        "message": "Engine reset successfully",
        "status": "ready for new policy"
    }

@app.delete("/cleanup")
def cleanup_files():
    """
    Clean up old uploaded files (maintenance endpoint).
    """
    upload_dir = "uploaded_policies"
    
    if not os.path.exists(upload_dir):
        return {"message": "No files to clean up"}
    
    try:
        files = os.listdir(upload_dir)
        count = 0
        
        for filename in files:
            filepath = os.path.join(upload_dir, filename)
            
            # Don't delete currently loaded policy
            if filepath != INGESTED_FILEPATH:
                try:
                    os.remove(filepath)
                    count += 1
                except:
                    pass
        
        return {
            "message": f"Cleaned up {count} old files",
            "files_deleted": count
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Cleanup failed: {str(e)}"}
        )

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*60)
    logger.info("🚀 Starting Optimized Insurance Claim Validator API")
    logger.info("="*60)
    logger.info("⚡ Optimizations enabled:")
    logger.info("  - Parallel processing")
    logger.info("  - Faster model support (mistral)")
    logger.info("  - Reduced LLM calls")
    logger.info("  - Regex-first extraction")
    logger.info("  - Cached retrieval")
    logger.info("="*60)
    logger.info("📋 Setup checklist:")
    logger.info("  1. Ollama running: ollama serve")
    logger.info("  2. Fast model installed: ollama pull mistral")
    logger.info("  3. (Optional) GPU enabled for PyTorch")
    logger.info("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
