import os
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from typing import Any, Dict, List


LLM_MODEL_NAME = "phi3"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ----------------- Utility: Safe Number Cast -----------------
def _safe_number_cast(value: str):
    """Convert string to int/float if possible; else return original or None."""
    if value is None:
        return None
    v = str(value).strip()

    if v.lower() in ("null", "none", "na", "n/a", ""):
        return None

    v_clean = re.sub(r"[^\d\.\-]", "", v)
    if v_clean == "":
        return v

    try:
        if "." in v_clean:
            return float(v_clean)
        return int(v_clean)
    except:
        return v


# ----------------- Policy RAG Engine -----------------
class PolicyRAGEngine:
    """Insurance RAG engine with robust JSON parsing + fallback."""

    def __init__(self, policy_file_path: str):
        self.policy_file_path = policy_file_path
        self.vectorstore = None
        self.policy_summary = None

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

        self.llm = ChatOllama(
            model=LLM_MODEL_NAME,
            temperature=0.0,
            num_predict=2000
        )

    # ----------------- Document ingestion -----------------
    def ingest_policy_document(self):
        loader = PyPDFLoader(self.policy_file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            splits,
            embedding=self.embedding_model,
            collection_name="universal_insurance_policy"
        )

        self._generate_policy_summary(splits)
        return len(docs), len(splits)

    # ----------------- Policy Summary -----------------
    def _generate_policy_summary(self, splits):
        sample = "\n\n".join([s.page_content for s in splits[:5]])
        prompt = (
            "Summarize this policy (3-5 sentences): type, eligibility, coverage, exclusions.\n\n"
            f"{sample[:2000]}"
        )
        try:
            res = self.llm.invoke(prompt)
            self.policy_summary = res.content
        except:
            self.policy_summary = "Summary generation failed."

    # ----------------- Claim Validation -----------------
    def validate_claim(self, claim_text: str) -> dict:
        if not self.vectorstore:
            raise ValueError("Policy not ingested.")

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        retrieved_docs = retriever.invoke(claim_text)
        policy_context = "\n---\n".join([d.page_content for d in retrieved_docs])

        # Strict JSON enforced prompt
        prompt = f"""
You are an Insurance Claim Validator.

You MUST return ONLY strict VALID JSON.
Rules:
- NO text before JSON
- NO text after JSON
- NO comments
- NO trailing commas
- NO explanations outside JSON
- NO extra keys
- NO markdown fences
- MUST match schema EXACTLY

Return JSON:

{{
  "is_valid": true or false,
  "decision_reason": "string",
  "extracted_entities": {{
      "age": number or null,
      "procedure": "string" or "",
      "policy_duration_months": number or null,
      "hospitalization_days": number or null,
      "claim_amount": number or null
  }},
  "validation_checks": [
      {{
          "check": "string",
          "status": "PASS" or "FAIL" or "N/A",
          "detail": "string",
          "policy_reference": "string"
      }}
  ],
  "policy_clauses_used": ["string"]
}}

POLICY SUMMARY:
{self.policy_summary}

POLICY CONTEXT:
{policy_context}

CLAIM DETAILS:
{claim_text}

BEGIN STRICT JSON OUTPUT:
""".replace("{", "{{").replace("}", "}}")


        llm_chain = ChatPromptTemplate.from_messages([
            ("system", prompt)
        ]) | self.llm

        llm_output = llm_chain.invoke({})

        return self._parse_llm_response(llm_output.content, retrieved_docs, claim_text)

    # ----------------- Cleaning Helpers -----------------
    def _clean_text(self, text):
        t = text

        t = t.strip()
        t = re.sub(r"```json", "", t)
        t = re.sub(r"```", "", t)
        t = re.sub(r"#.*?$", "", t, flags=re.MULTILINE)
        t = re.sub(r"//.*?$", "", t, flags=re.MULTILINE)
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)
        t = re.sub(r"[^\x00-\x7F]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # ----------------- JSON Parsing + Repair -----------------
    def _parse_llm_response(self, response_text: str, retrieved_docs, claim_text="") -> dict:

        original = response_text
        cleaned = self._clean_text(response_text)

        try:
            # Extract main JSON block
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            candidate = m.group(0) if m else cleaned

            candidate = candidate.replace("True", "true").replace("False", "false").replace("None", "null")

            # Remove trailing commas
            candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

            # Fix quotes
            candidate = re.sub(r"(\w+)':", r'"\1":', candidate)
            candidate = re.sub(r"':\s*'([^']+)'", r'": "\1"', candidate)

            parsed = json.loads(candidate)

            return self._normalize(parsed, retrieved_docs)

        except Exception:
            # ---------- FINAL FALLBACK ----------
            return self._fallback_recover(original, retrieved_docs, claim_text)

    # ----------------- Normalization -----------------
    def _normalize(self, parsed, retrieved_docs):
        if not isinstance(parsed, dict):
            return self._fallback_total_failure(parsed, retrieved_docs)

        out = {
            "is_valid": bool(parsed.get("is_valid", False)),
            "decision_reason": parsed.get("decision_reason", ""),
            "validation_checks": parsed.get("validation_checks", []),
            "policy_clauses_used": parsed.get("policy_clauses_used", [])
        }

        ent = parsed.get("extracted_entities", {})

        out["extracted_entities"] = {
            "age": _safe_number_cast(ent.get("age")),
            "procedure": ent.get("procedure", ""),
            "policy_duration_months": _safe_number_cast(ent.get("policy_duration_months")),
            "hospitalization_days": _safe_number_cast(ent.get("hospitalization_days")),
            "claim_amount": _safe_number_cast(ent.get("claim_amount"))
        }

        # If model didn't return clauses, use retrieved context
        if not out["policy_clauses_used"]:
            out["policy_clauses_used"] = [doc.page_content for doc in retrieved_docs]

        return out

    # ----------------- Fallback Key-Value Extractor -----------------
    def _fallback_recover(self, text, retrieved_docs, claim_text):
        lines = text.split("\n")
        kv = {}

        for line in lines:
            m = re.match(r'\s*"?([A-Za-z0-9_ ]+)"?\s*[:= ]\s*"?([^",]+)"?', line)
            if m:
                key = m.group(1).strip().replace(" ", "_")
                val = m.group(2).strip()
                kv[key.lower()] = val

        if kv:
            # FIXED: is_valid comes from model, NOT hardcoded
            is_valid_raw = str(kv.get("is_valid", "")).lower()
            is_valid_bool = is_valid_raw in ("true", "1", "yes")

            extracted = {
                "age": _safe_number_cast(kv.get("age")),
                "procedure": kv.get("procedure", ""),
                "policy_duration_months": _safe_number_cast(kv.get("policy_duration_months")),
                "hospitalization_days": _safe_number_cast(kv.get("hospitalization_days")),
                "claim_amount": _safe_number_cast(kv.get("claim_amount"))
            }

            return {
                "is_valid": is_valid_bool,
                "decision_reason": kv.get("decision_reason", "Recovered from malformed JSON."),
                "extracted_entities": extracted,
                "validation_checks": [],
                "policy_clauses_used": [doc.page_content for doc in retrieved_docs],
                "raw_response": text[:500],
                "note": "Recovered using fallback parser"
            }

        return self._fallback_total_failure(text, retrieved_docs)

    # ----------------- Total Failure -----------------
    def _fallback_total_failure(self, raw, retrieved_docs):
        return {
            "is_valid": False,
            "decision_reason": "Parser failed.",
            "extracted_entities": {},
            "validation_checks": [],
            "policy_clauses_used": [doc.page_content for doc in retrieved_docs],
            "raw_response": raw[:700]
        }

    # ----------------- Public Summary Getter -----------------
    def get_policy_summary(self):
        return self.policy_summary or "Summary unavailable"
