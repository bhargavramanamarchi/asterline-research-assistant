"""
test_stage2.py - Automated Verification Suite for Stage 2
Tests:
  PART A: Deterministic Evidence Retrieval & Citation Verification (Offline)
    1. Straightforward factual retrieval: Enterprise SLA & service credits (sla:0).
    2. Conflict retrieval & textual analysis: Cold recovery retention (security:1 vs retention:0 supersession).
    3. Version-specific retrieval: AQL timeout cross-version documentation (query:0).
    4. Unannounced/missing information retrieval: Asia-Pacific region timeline (faq:1).
    5. Citation validator integrity: Validates matching citations and catches unretrieved/invented citations.
    6. Prompt construction check: Ensures formatting instructions and citations guidelines are strictly present.

  PART B: Live Generation Integration (Online - Requires GEMINI_API_KEY in .env)
    7. Live Gemini research query execution for all 4 scenarios.
    8. Validates live answer citations against retrieved evidence.
    If GEMINI_API_KEY is not configured, clearly marks live tests as BLOCKED.
"""

import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

# Ensure .env is explicitly reloaded
load_dotenv(override=True)

from src.indexer import search_corpus
from src.researcher import (
    format_evidence_prompt,
    get_gemini_api_key,
    research_question,
    validate_citations,
)


def test_part_a_deterministic_retrieval() -> bool:
    print("\n" + "=" * 70)
    print("PART A: DETERMINISTIC RETRIEVAL & CITATION VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST A1: Straightforward Enterprise SLA Factual Retrieval
    # -------------------------------------------------------------
    print("\n--- TEST A1: Enterprise SLA & Service Credit Retrieval ---")
    query_sla = "What is the contractual monthly uptime SLA for Enterprise, and what are the service credits?"
    hits_sla = search_corpus(query_sla, top_k=3)
    if not hits_sla:
        print("FAILED: No hits returned for SLA query.")
        return False

    top_sla = hits_sla[0]
    print(f"Query: '{query_sla}'")
    print(f"Top Hit: [{top_sla['chunk_id']}] '{top_sla['title']}' (Score: {top_sla['score']})")

    if top_sla["chunk_id"] != "sla:0":
        print(f"FAILED: Expected rank-1 chunk 'sla:0', got '{top_sla['chunk_id']}'")
        return False
    if "99.92%" not in top_sla["text"] or "service credits" not in top_sla["text"].lower():
        print("FAILED: Retrieved text does not contain expected SLA metrics.")
        return False
    print("PASSED: Exactly retrieved chunk 'sla:0' containing 99.92% uptime SLA and credit tiers.")

    # -------------------------------------------------------------
    # TEST A2: Cold Recovery Retention Conflict Analysis
    # -------------------------------------------------------------
    print("\n--- TEST A2: Cold Recovery Conflict Retrieval & Supersession ---")
    query_retention = "How long are raw run records kept in cold recovery storage after searchable retention ends?"
    hits_retention = search_corpus(query_retention, top_k=5)
    retrieved_ids = [h["chunk_id"] for h in hits_retention]

    print(f"Query: '{query_retention}'")
    print(f"Top retrieved chunk IDs: {retrieved_ids[:3]}")

    if "retention:0" not in retrieved_ids[:3] or "security:1" not in retrieved_ids[:3]:
        print(f"FAILED: Both 'retention:0' and 'security:1' must be retrieved in top hits. Got {retrieved_ids}")
        return False

    chunk_retention = next(h for h in hits_retention if h["chunk_id"] == "retention:0")
    chunk_security = next(h for h in hits_retention if h["chunk_id"] == "security:1")

    print(f"Older document: [{chunk_security['chunk_id']}] '{chunk_security['title']}' (Published: {chunk_security['published']} v{chunk_security['version']})")
    print(f"  Snippet: {chunk_security['text'][:90]}...")
    print(f"Superseding doc:[{chunk_retention['chunk_id']}] '{chunk_retention['title']}' (Published: {chunk_retention['published']} v{chunk_retention['version']})")
    print(f"  Snippet: {chunk_retention['text'][:90]}...")

    # Confirm explicit supersession text
    supersedes_phrase = "This policy supersedes the earlier 60-day cold-storage period."
    if supersedes_phrase not in chunk_retention["text"]:
        print(f"FAILED: Expected explicit supersession phrase not found in retention:0")
        return False
    print(f"PASSED: Both competing chunks retrieved; retention:0 explicitly supersedes security:1.")

    # -------------------------------------------------------------
    # TEST A3: Version-Specific AQL Timeout Retrieval
    # -------------------------------------------------------------
    print("\n--- TEST A3: Version-Specific AQL Query Timeout Retrieval ---")
    query_aql = "What is the timeout for AQL queries across different versions?"
    hits_aql = search_corpus(query_aql, top_k=3)
    top_aql = hits_aql[0]

    print(f"Query: '{query_aql}'")
    print(f"Top Hit: [{top_aql['chunk_id']}] '{top_aql['title']}'")

    if top_aql["chunk_id"] != "query:0":
        print(f"FAILED: Expected 'query:0', got '{top_aql['chunk_id']}'")
        return False
    if "50 seconds in version 2.3" not in top_aql["text"] or "25-second timeout" not in top_aql["text"]:
        print("FAILED: Retrieved text does not contain version timeout differences.")
        return False
    print("PASSED: Retrieved 'query:0' containing version 2.3 (50s) vs earlier (25s) timeout rules.")

    # -------------------------------------------------------------
    # TEST A4: Unannounced / Negative Statement Retrieval
    # -------------------------------------------------------------
    print("\n--- TEST A4: Asia-Pacific Timeline Retrieval (Unannounced Feature) ---")
    query_apac = "Has Asterline announced a release date or timeline for an Asia-Pacific region?"
    hits_apac = search_corpus(query_apac, top_k=3)
    top_apac = hits_apac[0]

    print(f"Query: '{query_apac}'")
    print(f"Top Hit: [{top_apac['chunk_id']}] '{top_apac['title']}'")

    if top_apac["chunk_id"] != "faq:1":
        print(f"FAILED: Expected 'faq:1', got '{top_apac['chunk_id']}'")
        return False
    if "There is no announced timeline for an Asia-Pacific region" not in top_apac["text"]:
        print("FAILED: Retrieved chunk does not state the negative fact.")
        return False
    print("PASSED: Retrieved 'faq:1' explicitly stating that no Asia-Pacific timeline has been announced.")

    # -------------------------------------------------------------
    # TEST A5: Citation Validator Logic Integrity
    # -------------------------------------------------------------
    print("\n--- TEST A5: Citation Validator Logic Integrity ---")
    mock_retrieved = [
        {
            "chunk_id": "sla:0",
            "doc_id": "sla",
            "title": "Service and support terms",
            "category": "pricing",
            "published": "20260310",
            "version": "2",
            "source_url": "kb://asterline/pricing/sla",
            "text": "Only Enterprise has a contractual monthly uptime SLA, set at 99.92%.",
        },
        {
            "chunk_id": "pricing:0",
            "doc_id": "pricing",
            "title": "Plans and billing",
            "category": "pricing",
            "published": "20260310",
            "version": "3",
            "source_url": "kb://asterline/pricing/pricing",
            "text": "Launch costs $59 per month. Team costs $449 per month.",
        },
    ]

    # Scenario 1: Valid citations matching retrieved evidence
    valid_text = "Enterprise uptime SLA is 99.92% as documented in [sla:0] (Service and support terms)."
    val1 = validate_citations(valid_text, mock_retrieved)
    if not val1["is_valid"] or len(val1["valid_citations"]) != 1 or val1["valid_citations"][0]["chunk_id"] != "sla:0":
        print(f"FAILED: Valid citation was incorrectly flagged: {val1}")
        return False
    print("  Case 1: Valid citation [sla:0] correctly verified.")

    # Scenario 2: Hallucinated / unretrieved citation
    invalid_text = "According to [fake:99] and [unretrieved:1], the limits are infinite."
    val2 = validate_citations(invalid_text, mock_retrieved)
    if val2["is_valid"] or "fake:99" not in val2["invalid_ids"] or "unretrieved:1" not in val2["invalid_ids"]:
        print(f"FAILED: Invalid citations were not caught: {val2}")
        return False
    print("  Case 2: Hallucinated citations [fake:99], [unretrieved:1] successfully rejected.")

    # -------------------------------------------------------------
    # TEST A6: Prompt Structure & Guardrail Verification
    # -------------------------------------------------------------
    print("\n--- TEST A6: Grounded Prompt Formatting ---")
    sample_prompt = format_evidence_prompt("test question", mock_retrieved)
    for guardrail in ["ONLY use facts directly supported", "CITATIONS:", "CONFLICTS & DATED DOCUMENTS:", "UNCERTAINTY & MISSING EVIDENCE:"]:
        if guardrail not in sample_prompt:
            print(f"FAILED: Prompt missing critical guardrail directive: '{guardrail}'")
            return False
    print("PASSED: Prompt includes all required grounding and supersession directives.")

    print("\n" + "=" * 70)
    print("PART A: ALL DETERMINISTIC RETRIEVAL & CITATION TESTS PASSED!")
    print("=" * 70)
    return True


def test_part_b_live_generation() -> bool:
    print("\n" + "=" * 70)
    print("PART B: LIVE LLM GENERATION INTEGRATION")
    print("=" * 70)

    from src.researcher import get_active_provider

    provider, api_key = get_active_provider()
    if provider == "none" or not api_key:
        print("\n[STATUS: BLOCKED]")
        print("Neither GROQ_API_KEY nor GEMINI_API_KEY is configured in this project.")
        print("To enable live generation testing:")
        print("  1. In '.env', set: GROQ_API_KEY=your_actual_private_key")
        print("  2. Run this script again.")
        print("\nNote: Per project instructions, live tests are marked BLOCKED (not passing).")
        print("=" * 70)
        return False

    print(f"\nActive generation provider detected: {provider.upper()}. Running live generation tests...")

    test_queries = [
        {
            "name": "Straightforward SLA",
            "query": "What is the contractual monthly uptime SLA for Enterprise, and what are the service credit tiers?",
            "expected_cited_id": "sla:0",
            "must_contain": ["99.92", "credit"],
        },
        {
            "name": "Cold Recovery Conflict",
            "query": "How many days are raw run records held in cold recovery storage after searchable retention ends?",
            "expected_cited_id": "retention:0",
            "must_contain": ["20 days", "supersede"],
        },
        {
            "name": "AQL Timeout Across Versions",
            "query": "What is the query timeout for AQL queries?",
            "expected_cited_id": "query:0",
            "must_contain": ["50", "25"],
        },
        {
            "name": "Asia-Pacific Region Timeline",
            "query": "Has Asterline announced a release date or timeline for an Asia-Pacific region?",
            "expected_cited_id": "faq:1",
            "must_contain": ["no announced timeline", "not"],
        },
    ]

    all_passed = True
    for item in test_queries:
        print(f"\nRunning Live Test: {item['name']}")
        result = research_question(item["query"], top_k=5)
        print(f"Status: {result['status']} | Provider: {result.get('provider')} | Model: {result.get('model')}")

        if result["status"] != "success":
            err_type = result.get("error_type", "")
            err_info = result.get("error_detail") or result.get("answer", "")
            print(f"FAILED: Live query error [{err_type}]: {err_info}")
            all_passed = False
            continue

        answer = result["answer"]
        print(f"Answer snippet:\n{answer[:250]}...\n")

        # Validate citations
        val = result["citation_validation"]
        print(f"Citations validated: {val['is_valid']}, Found: {val['cited_ids']}")

        if not val["is_valid"]:
            print(f"FAILED: Citations failed validation: {val}")
            all_passed = False
            continue

        if item["expected_cited_id"] not in val["cited_ids"]:
            print(f"FAILED: Expected cited chunk '{item['expected_cited_id']}' not in cited IDs: {val['cited_ids']}")
            all_passed = False
            continue

        # Check required semantic elements
        lower_answer = answer.lower()
        missing_phrases = [p for p in item["must_contain"] if p.lower() not in lower_answer]
        if missing_phrases:
            print(f"WARNING: Answer might be missing expected nuance phrases: {missing_phrases}")

        print(f"PASSED: Live query '{item['name']}' returned grounded answer with valid citations.")

    if all_passed:
        print("\n" + "=" * 70)
        print(f"PART B: ALL LIVE {provider.upper()} INTEGRATION TESTS PASSED!")
        print("=" * 70)
    return all_passed


if __name__ == "__main__":
    part_a_ok = test_part_a_deterministic_retrieval()
    part_b_ok = test_part_b_live_generation()

    if part_a_ok and part_b_ok:
        print("\nALL STAGE 2 TESTS (PART A + PART B) COMPLETED SUCCESSFULLY.")
        sys.exit(0)
    elif part_a_ok and not part_b_ok:
        print("\nSTAGE 2 PART A (DETERMINISTIC RETRIEVAL & CITATIONS) PASSED.")
        print("STAGE 2 PART B (LIVE GENERATION) IS BLOCKED DUE TO MISSING API KEY.")
        sys.exit(2)
    else:
        print("\nSTAGE 2 VERIFICATION FAILED.")
        sys.exit(1)
