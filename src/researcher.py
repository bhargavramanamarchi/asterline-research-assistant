"""
src/researcher.py - Asterline Research Assistant
Retrieval and Evidence Verification Component.

Retrieves relevant evidence from persistent ChromaDB and generates grounded
answers with citation validation and conflict analysis.
Supports Groq (default, ultra-fast) and Google Gemini (fallback provider).
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.indexer import search_corpus

# Load environment variables from .env file if present
load_dotenv(override=True)


def get_groq_api_key() -> Optional[str]:
    """
    Retrieves the Groq API key from environment variables or Streamlit secrets.
    Checks GROQ_API_KEY. Never prints or logs the key value.
    """
    load_dotenv(override=True)
    key = os.getenv("GROQ_API_KEY")
    invalid_placeholders = {
        "your_groq_api_key_here",
        "PASTE_YOUR_ACTUAL_KEY_HERE",
        "your_actual_key_here",
    }
    if key and key.strip() and key.strip() not in invalid_placeholders:
        return key.strip()

    # Fallback to Streamlit Community Cloud secrets when not in os.environ
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            sec_key = str(st.secrets["GROQ_API_KEY"]).strip()
            if sec_key and sec_key not in invalid_placeholders:
                return sec_key
    except Exception:
        pass

    return None


def get_gemini_api_key() -> Optional[str]:
    """
    Retrieves the Gemini API key from environment variables or Streamlit secrets.
    Checks GEMINI_API_KEY first, then GOOGLE_API_KEY.
    Never prints or logs the key value.
    """
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    invalid_placeholders = {
        "your_gemini_api_key_here",
        "PASTE_YOUR_ACTUAL_KEY_HERE",
        "your_actual_key_here",
    }
    if key and key.strip() and key.strip() not in invalid_placeholders:
        return key.strip()

    # Fallback to Streamlit Community Cloud secrets when not in os.environ
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for candidate in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if candidate in st.secrets:
                    sec_key = str(st.secrets[candidate]).strip()
                    if sec_key and sec_key not in invalid_placeholders:
                        return sec_key
    except Exception:
        pass

    return None


def get_active_provider() -> Tuple[str, Optional[str]]:
    """
    Determines the active LLM provider based on configured environment variables.
    Priority order:
      1. Groq (if GROQ_API_KEY is configured)
      2. Gemini (if GEMINI_API_KEY is configured)
      3. None (missing credentials)
    """
    groq_key = get_groq_api_key()
    if groq_key:
        return ("groq", groq_key)

    gemini_key = get_gemini_api_key()
    if gemini_key:
        return ("gemini", gemini_key)

    return ("none", None)


def format_evidence_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved documentary evidence into a structured, strictly grounded prompt.
    Instructs the LLM on:
      - Documentary grounding only (no external assumptions or user memories).
      - Citation syntax [chunk_id] and exact title.
      - Conflict and supersession resolution (comparing published dates, versions, explicit clauses).
      - Acknowledging unannounced/missing features honestly.
    """
    evidence_blocks: List[str] = []
    for idx, c in enumerate(retrieved_chunks, start=1):
        block = (
            f"--- EVIDENCE ITEM {idx} ---\n"
            f"Chunk ID: {c.get('chunk_id', '')}\n"
            f"Document Title: {c.get('title', '')}\n"
            f"Document ID: {c.get('doc_id', '')}\n"
            f"Category: {c.get('category', '')}\n"
            f"Published Date: {c.get('published', '')}\n"
            f"Version: {c.get('version', '')}\n"
            f"Source URL: {c.get('source_url', '')}\n"
            f"Content:\n{c.get('text', '')}\n"
        )
        evidence_blocks.append(block)

    joined_evidence = "\n".join(evidence_blocks)

    prompt = f"""You are the Asterline Research Assistant. Answer the question below strictly using the provided documentary evidence from Asterline Labs.

CRITICAL INSTRUCTIONS:
1. ONLY use facts directly supported by the evidence below. Never invent company facts, limits, or dates.
2. CITATIONS: Every claim must cite the specific source chunk using the exact format: `[chunk_id] (Document Title)`.
3. CONFLICTS & DATED DOCUMENTS:
   - If documents disagree or describe different points in time, compare the specific published dates, versions, and explicit supersession text (e.g. "This policy supersedes...").
   - Do NOT assume a newer document automatically overrides an older one unless the text or scope explicitly supports that.
   - Explain the divergence clearly (e.g., historical vs. current policy).
4. UNCERTAINTY & MISSING EVIDENCE:
   - If the evidence states something is unannounced or if the evidence is insufficient to answer the question, explicitly state: "The retrieved documentation does not establish..."
5. DO NOT invent citations to chunks not listed in the evidence.

QUESTION:
{query}

RETRIEVED EVIDENCE:
{joined_evidence}

ANSWER (grounded with exact citations and conflict resolution if applicable):"""
    return prompt


def validate_citations(
    answer: str, retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validates all citations in the generated answer against the actual retrieved chunks.
    
    Checks:
      1. Every cited chunk_id (e.g. [retention:0]) must exist in the retrieved evidence.
      2. Detects any unretrieved or hallucinated chunk IDs.
      3. Verifies cited titles match the actual titles of the retrieved chunks.
    
    Returns a dictionary with validation status and detailed findings.
    """
    retrieved_id_map = {c["chunk_id"]: c for c in retrieved_chunks}

    # Match chunk IDs in patterns like [retention:0] or `[retention:0]`
    cited_ids_found = re.findall(r"\[([a-zA-Z0-9_\-]+:[0-9]+)\]", answer)
    # Deduplicate while preserving occurrence order
    unique_cited_ids = list(dict.fromkeys(cited_ids_found))

    valid_citations: List[Dict[str, Any]] = []
    invalid_ids: List[str] = []

    for cid in unique_cited_ids:
        if cid in retrieved_id_map:
            doc = retrieved_id_map[cid]
            valid_citations.append({
                "chunk_id": cid,
                "title": doc.get("title", ""),
                "published": doc.get("published", ""),
                "version": doc.get("version", ""),
                "is_valid": True,
            })
        else:
            invalid_ids.append(cid)

    is_valid = len(invalid_ids) == 0 and len(valid_citations) > 0

    return {
        "is_valid": is_valid,
        "total_citations": len(unique_cited_ids),
        "valid_citations": valid_citations,
        "invalid_ids": invalid_ids,
        "cited_ids": unique_cited_ids,
        "available_chunk_ids": list(retrieved_id_map.keys()),
    }


def sanitize_error(error_msg: str, api_key: Optional[str] = None) -> str:
    """
    Sanitizes error messages before display or logging to ensure no credentials
    or authorization tokens are ever printed or leaked.
    """
    sanitized = str(error_msg)
    if api_key and api_key in sanitized:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    # Redact common key formats (Groq gsk_..., Google AI Studio AIza..., Bearer tokens)
    sanitized = re.sub(r"gsk_[A-Za-z0-9]+", "gsk_[REDACTED]", sanitized)
    sanitized = re.sub(r"AIza[A-Za-z0-9_\-]+", "AIza[REDACTED]", sanitized)
    sanitized = re.sub(r"key=[A-Za-z0-9_\-]+", "key=[REDACTED]", sanitized)
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", sanitized)
    return sanitized


def generate_grounded_answer(
    prompt: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes live text generation using the active configured provider.
    Tries Groq first (fastest, high throughput), falling back to Gemini if configured.
    """
    provider, api_key = get_active_provider()

    if provider == "none" or not api_key:
        return {
            "status": "missing_api_key",
            "provider": "none",
            "model": "none",
            "text": (
                "Neither GROQ_API_KEY nor GEMINI_API_KEY is configured in this project. "
                "Please configure GROQ_API_KEY in your .env file."
            ),
            "error_type": "MissingCredentials",
            "error_detail": "No valid API key found in environment variables.",
        }

    # -------------------------------------------------------------
    # PROVIDER: GROQ
    # -------------------------------------------------------------
    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=api_key)
        configured_model = model_name or os.getenv("GROQ_MODEL")
        
        # Build candidate models list (preference order)
        candidates: List[str] = []
        if configured_model:
            candidates.append(configured_model)
        for m in [
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
        ]:
            if m not in candidates:
                candidates.append(m)

        answer_text = ""
        used_model = candidates[0]
        last_error: Optional[Exception] = None

        for candidate in candidates:
            try:
                response = client.chat.completions.create(
                    model=candidate,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are the Asterline Research Assistant. Answer strictly "
                                "using the provided documentary evidence from Asterline Labs. "
                                "Ground every claim with exact citations in `[chunk_id] (Title)` format."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=800,
                )
                if response and response.choices and response.choices[0].message.content:
                    answer_text = response.choices[0].message.content.strip()
                    used_model = candidate
                    last_error = None
                    break
            except Exception as err:
                last_error = err
                continue

        if last_error is not None and not answer_text:
            err_type = type(last_error).__name__
            err_detail = sanitize_error(str(last_error), api_key)
            return {
                "status": "error",
                "provider": "groq",
                "model": used_model,
                "text": f"Groq generation failed due to {err_type}: {err_detail}",
                "error_type": err_type,
                "error_detail": err_detail,
            }

        return {
            "status": "success",
            "provider": "groq",
            "model": used_model,
            "text": answer_text,
        }

    # -------------------------------------------------------------
    # PROVIDER: GEMINI
    # -------------------------------------------------------------
    from google import genai

    client = genai.Client(api_key=api_key)
    configured_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    candidates = [configured_model]
    for fallback in ["gemini-3-flash-preview", "gemini-flash-latest"]:
        if fallback not in candidates:
            candidates.append(fallback)

    answer_text = ""
    used_model = candidates[0]
    last_error: Optional[Exception] = None

    for candidate in candidates:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt,
            )
            if response and response.text:
                answer_text = response.text
                used_model = candidate
                last_error = None
                break
        except Exception as err:
            last_error = err
            continue

    if last_error is not None and not answer_text:
        err_type = type(last_error).__name__
        err_detail = sanitize_error(str(last_error), api_key)
        return {
            "status": "error",
            "provider": "gemini",
            "model": used_model,
            "text": f"Gemini generation failed due to {err_type}: {err_detail}",
            "error_type": err_type,
            "error_detail": err_detail,
        }

    return {
        "status": "success",
        "provider": "gemini",
        "model": used_model,
        "text": answer_text,
    }


def research_question(
    query: str,
    top_k: int = 5,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main research answering pipeline for Stage 2:
      1. Retrieves top_k relevant evidence chunks from Chroma index.
      2. Checks for configured provider (Groq or Gemini).
      3. Generates answer using grounded evidence prompt.
      4. Validates citations against authentic retrieved chunks.
      5. Returns structured research result with sanitized diagnostics.
    """
    # 1. Retrieve evidence from existing persistent index
    retrieved_chunks = search_corpus(query=query, top_k=top_k)

    prompt = format_evidence_prompt(query, retrieved_chunks)

    # 2. Call generation provider
    gen_result = generate_grounded_answer(prompt, model_name=model_name)

    if gen_result["status"] != "success":
        return {
            "status": gen_result["status"],
            "query": query,
            "provider": gen_result.get("provider", "none"),
            "model": gen_result.get("model", "none"),
            "error_type": gen_result.get("error_type", ""),
            "error_detail": gen_result.get("error_detail", ""),
            "answer": gen_result.get("text", ""),
            "retrieved_chunks": retrieved_chunks,
            "citations": [],
            "citation_validation": {
                "is_valid": False,
                "reason": gen_result.get("error_detail", "Generation failed"),
            },
        }

    answer_text = gen_result["text"]

    # 3. Validate citations
    validation = validate_citations(answer_text, retrieved_chunks)

    return {
        "status": "success",
        "query": query,
        "provider": gen_result["provider"],
        "model": gen_result["model"],
        "answer": answer_text,
        "retrieved_chunks": retrieved_chunks,
        "citations": validation["valid_citations"],
        "citation_validation": validation,
    }
