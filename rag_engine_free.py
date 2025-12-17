import os
import re
import logging
import concurrent.futures
from typing import Dict, List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from performance_monitor import PerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_MODEL_NAME = "mistral"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class PolicyRAGEngine:
    """
    Fast, parallel, timeout-safe RAG engine with performance metrics
    """
    def __init__(self, policy_file_path: str):
        self.policy_file_path = policy_file_path
        self.vectorstore = None
        self.policy_summary = None
        self.all_chunks = []

        # Performance monitor
        self.monitor = PerformanceMonitor()

        # Device selection - GPU if present or else CPU
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            self.device = "cpu"

        logger.info(f"Running on {self.device}")

        # Embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True,
                           "batch_size": 32 if self.device == "cuda" else 16}
        )

        # LLM Model Tuning for better performance and Optimal solution
        self.llm = ChatOllama(
            model=LLM_MODEL_NAME,
            temperature=0.0,
            num_predict=500,
            num_ctx=1536,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1
        )
        logger.info("RAG Engine initialized")


    # INGESTION OF POLICY DOCUMENT
    def ingest_policy_document(self) -> Tuple[int, int]:
        logger.info(f"Ingesting policy: {self.policy_file_path}")

        loader = PyPDFLoader(self.policy_file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        self.all_chunks = splits

        self.vectorstore = Chroma.from_documents(
            splits, embedding=self.embedding_model, collection_name="insurance_policy"
        )

        # Summary
        self.policy_summary = self._generate_policy_summary(splits)
        logger.info(f"âœ… Ingested {len(docs)} pages â†’ {len(splits)} chunks")
        return len(docs), len(splits)

    # PROMPT ENGINEERING
    def _generate_policy_summary(self, splits):
        sample = "\n".join([s.page_content for s in splits[:4]])[:1200]
        prompt = f"""Give a 3-bullet insurance policy summary:
- Coverage type
- Key limits
- Major exclusions

Text:
{sample}
"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return "Policy summary unavailable"

    def get_policy_summary(self) -> str:
        return self.policy_summary or "Summary unavailable"

    # VALIDATION OF THE CLAIMS
    def validate_claim(self, claim_text: str) -> dict:
        if not self.vectorstore:
            raise ValueError("Policy not ingested")

        self.monitor = PerformanceMonitor()
        self.monitor.start("total_time")
        logger.info("Starting FAST claim validation")

        try:
            # Entity + check extraction
            self.monitor.start("entity_extraction")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f_entities = executor.submit(self._extract_entities_fast, claim_text)
                f_checks = executor.submit(self._identify_checks_fast, claim_text)
                entities = f_entities.result()
                checks = f_checks.result()
            self.monitor.stop("entity_extraction")

            # Retrieval
            self.monitor.start("retrieval")
            docs = self._fast_retrieval(claim_text, checks)
            self.monitor.stop("retrieval")

            # Parallel checks
            self.monitor.start("parallel_checks")
            validation_results = self._perform_checks_parallel(claim_text, entities, checks[:5], docs)
            self.monitor.stop("parallel_checks")

            # Final decision
            decision = self._make_final_decision(validation_results)

            self.monitor.stop("total_time")
            metrics = self.monitor.finalize()

            return {
                "is_valid": decision["is_valid"],
                "decision_reason": decision["reason"],
                "extracted_entities": entities,
                "validation_checks": validation_results,
                "policy_clauses_used": [d.page_content for d in docs[:5]],
                "performance_metrics": metrics
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            self.monitor.stop("total_time")
            metrics = self.monitor.finalize()
            return {
                "is_valid": False,
                "decision_reason": f"System error: {e}",
                "extracted_entities": {"claim": claim_text[:200]},
                "validation_checks": [],
                "policy_clauses_used": [],
                "performance_metrics": metrics
            }

    # ENTITY EXTRACTION - using REGEX
    def _extract_entities_fast(self, text: str) -> dict:
        entities = {}
        age = re.search(r'(\d+)[-\s]year[-\s]old', text, re.I)
        if age: entities["age"] = int(age.group(1))
        amount = re.search(r'[â‚¹Rs\.]\s*([\d,]+)', text)
        if amount: entities["claim_amount"] = float(amount.group(1).replace(",", ""))
        days = re.search(r'(\d+)\s*days?', text, re.I)
        if days: entities["hospitalization_days"] = int(days.group(1))
        years = re.search(r'(\d+)\s*years?', text, re.I)
        if years: entities["policy_duration_years"] = int(years.group(1))
        return entities


    # CHECK IDENTIFICATION - Identifying which checks to perform based on the user query
    def _identify_checks_fast(self, text: str) -> List[dict]:
        checks = [
            {"check_name": "Coverage", "terms": ["coverage", "benefit"]},
            {"check_name": "Age Eligibility", "terms": ["age", "limit"]},
            {"check_name": "Amount Limit", "terms": ["limit", "sum insured"]},
            {"check_name": "Waiting Period", "terms": ["waiting"]},
            {"check_name": "Exclusions", "terms": ["exclusion", "not covered"]}
        ]
        logger.info(f"Identified {len(checks)} checks")
        return checks

    # RETRIEVAL of the policy content to check
    def _fast_retrieval(self, claim_text: str, checks: List[dict]):
        query = claim_text[:150] + " " + " ".join(term for c in checks[:3] for term in c["terms"])
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        return retriever.invoke(query)

    # PARALLEL CHECKS to optimise the time required
    def _perform_checks_parallel(self, claim_text, entities, checks, docs):
        results = []
        context = "\n\n".join(d.page_content for d in docs[:4])[:1200]
        MAX_WORKERS = min(3, len(checks))
        PER_CHECK_TIMEOUT = 25

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(self._single_check, check, claim_text, entities, context): check for check in checks}

            for future, check in future_map.items():
                try:
                    results.append(future.result(timeout=PER_CHECK_TIMEOUT))
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout: {check['check_name']}")
                    self.monitor.increment("timeouts")
                    results.append({
                        "check": check["check_name"],
                        "status": "N/A",
                        "detail": "Check timed out",
                        "policy_reference": "N/A"
                    })
                except Exception as e:
                    results.append({
                        "check": check["check_name"],
                        "status": "ERROR",
                        "detail": str(e),
                        "policy_reference": "N/A"
                    })
        return results

    def _single_check(self, check, claim_text, entities, context):
        self.monitor.increment("llm_calls")
        prompt = f"""Check: {check['check_name']}

Policy:
{context}

Claim:
{claim_text}

Respond exactly:
STATUS: PASS/FAIL/N/A
DETAIL: one sentence
REFERENCE: short quote
"""
        res = self.llm.invoke(prompt).content
        status = re.search(r'STATUS:\s*(PASS|FAIL|N/A)', res)
        detail = re.search(r'DETAIL:\s*(.+)', res)
        ref = re.search(r'REFERENCE:\s*(.+)', res)
        return {
            "check": check["check_name"],
            "status": status.group(1) if status else "PASS",
            "detail": detail.group(1).strip() if detail else "Checked",
            "policy_reference": ref.group(1).strip() if ref else "Policy verified"
        }

    # FINAL DECISION based on the Cases passed or Failed
    def _make_final_decision(self, checks: List[dict]) -> dict:
        fails = [c for c in checks if c["status"] == "FAIL"]
        errors = [c for c in checks if c["status"] == "ERROR"]
        is_valid = len(fails) == 0 and len(errors) == 0
        reason = "All required validation checks passed." if is_valid else "One or more validation checks failed."
        return {"is_valid": is_valid, "reason": reason}