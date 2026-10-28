"""
eval/eval_runner.py - Asterline Research Assistant
Automated Benchmark Evaluation Runner & Measured Improvement Experiment.

Executes 15 benchmark questions across:
  - Factual retrieval
  - Temporal supersession & conflict resolution
  - Version-specific behavior
  - Negative/unannounced feature grounding
  - Persistent SQLite memory CRUD & user isolation
  - Hybrid context-conditioned research

Measures:
  1. Routing Correctness (Planner Intent & Node Sequence)
  2. Retrieval Recall (Expected chunks in top-k)
  3. Citation Validity & Faithfulness (Zero hallucinated citations)
  4. Factual Grounding Accuracy (Expected semantic facts in answer)
  5. A/B Measured Improvement (Baseline RAG vs. Asterline Temporal Grounding)
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(override=True)

from src.agents.graph import run_agent
from src.agents.memory_agent import set_default_memory_store
from src.indexer import search_corpus
from src.memory import SQLiteMemoryStore
from src.researcher import format_evidence_prompt, get_active_provider, validate_citations

DATASET_PATH = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def load_benchmark_dataset() -> List[Dict[str, Any]]:
    """Loads the 15-question benchmark dataset."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline_rag(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Simulates a standard baseline RAG prompt without temporal conflict
    or document supersession directives.
    """
    provider, api_key = get_active_provider()
    if provider == "none" or not api_key:
        return "Baseline blocked: Missing API key."

    context_text = "\n\n".join(f"[{c['chunk_id']}] {c.get('title', '')}: {c['text']}" for c in chunks)
    baseline_prompt = f"""Answer the question based only on the context below:
Context:
{context_text}

Question: {query}
Answer:"""

    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=api_key)
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": baseline_prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    from google import genai

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    resp = client.models.generate_content(model=model, contents=baseline_prompt)
    return resp.text.strip() if resp and resp.text else ""


def evaluate_benchmark():
    """Runs all 15 benchmark questions and computes quantitative metrics."""
    print("=" * 75)
    print("ASTERLINE RESEARCH ASSISTANT: AUTOMATED BENCHMARK EVALUATION")
    print("=" * 75)

    dataset = load_benchmark_dataset()
    print(f"Loaded {len(dataset)} benchmark cases from: {DATASET_PATH}\n")

    # Use isolated test database for evaluation run
    with tempfile.TemporaryDirectory() as temp_dir:
        eval_db_path = str(Path(temp_dir) / "eval_memory.db")
        eval_store = SQLiteMemoryStore(db_path=eval_db_path)
        set_default_memory_store(eval_store)

        results: List[Dict[str, Any]] = []

        total_cases = len(dataset)
        routing_correct_count = 0
        retrieval_correct_count = 0
        citation_valid_count = 0
        factual_correct_count = 0

        research_cases = 0
        memory_cases = 0

        t_start = time.time()

        for idx, item in enumerate(dataset, start=1):
            cid = item["id"]
            category = item["category"]
            item_type = item["type"]
            query = item["query"]
            user_id = item.get("user_id", "eval_user_default")

            print(f"[{idx:02d}/{total_cases:02d}] Running {cid} ({category})...")

            # Setup any pre-existing memories if specified
            if "setup_memory" in item:
                for sk, sv in item["setup_memory"].items():
                    eval_store.set_memory(user_id, sk, sv)

            # Execute multi-agent graph
            agent_result = run_agent(query=query, user_id=user_id)

            trace = agent_result.get("execution_trace", [])
            intent = agent_result.get("intent", "")
            answer = agent_result.get("answer", "")
            raw_citations = agent_result.get("citations", [])
            cited_ids = [c["chunk_id"] if isinstance(c, dict) else str(c) for c in raw_citations]
            retrieved_chunks = agent_result.get("retrieved_chunks", [])
            retrieved_chunk_ids = [c["chunk_id"] for c in retrieved_chunks]

            # 1. Routing Correctness
            expected_trace = item.get("expected_node_trace", [])
            expected_intent = item.get("expected_intent")
            routing_ok = (trace == expected_trace) and (intent == expected_intent)
            if routing_ok:
                routing_correct_count += 1

            # 2. Retrieval Recall (for research & hybrid types)
            expected_chunks = item.get("expected_chunks", [])
            if expected_chunks:
                research_cases += 1
                retrieval_ok = all(ec in retrieved_chunk_ids for ec in expected_chunks)
                if retrieval_ok:
                    retrieval_correct_count += 1
            else:
                retrieval_ok = True
                memory_cases += 1

            # 3. Citation Validity
            if item_type in ("research", "hybrid"):
                val = validate_citations(answer, retrieved_chunks)
                citation_ok = val["is_valid"] and (len(val["invalid_ids"]) == 0) and (len(cited_ids) > 0)
                if citation_ok:
                    citation_valid_count += 1
            else:
                citation_ok = True  # Memory operations don't require citations

            # 4. Factual Grounding Accuracy
            expected_facts = item.get("expected_facts", [])
            answer_lower = answer.lower()
            if expected_facts:
                facts_matched = sum(1 for ef in expected_facts if ef.lower() in answer_lower)
                fact_score = facts_matched / len(expected_facts)
                factual_ok = fact_score >= 0.75  # Threshold 75%
            elif item_type == "memory" and item.get("expected_action") == "store":
                # Verify key was stored in SQLite
                mk = item.get("expected_memory_key")
                mv = item.get("expected_memory_val")
                db_val = eval_store.get_memory(user_id, mk) if mk else None
                factual_ok = (db_val == mv)
                fact_score = 1.0 if factual_ok else 0.0
            else:
                factual_ok = True
                fact_score = 1.0

            if factual_ok:
                factual_correct_count += 1

            case_status = "PASSED" if (routing_ok and retrieval_ok and citation_ok and factual_ok) else "FAILED"
            print(f"       Status: {case_status} | Routing: {routing_ok} | Retrieval: {retrieval_ok} | Citations: {citation_ok} | Facts: {factual_ok}")

            results.append({
                "id": cid,
                "category": category,
                "type": item_type,
                "query": query,
                "user_id": user_id,
                "case_status": case_status,
                "actual_trace": trace,
                "expected_trace": expected_trace,
                "actual_intent": intent,
                "expected_intent": expected_intent,
                "routing_ok": routing_ok,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "expected_chunks": expected_chunks,
                "retrieval_ok": retrieval_ok,
                "cited_ids": cited_ids,
                "citation_valid": citation_ok,
                "fact_score": fact_score,
                "factual_ok": factual_ok,
                "answer_snippet": answer[:200],
            })

        duration = time.time() - t_start

        # ------------------------------------------------------------------
        # MEASURED IMPROVEMENT EXPERIMENT: BASELINE VS TEMPORAL GROUNDING
        # ------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("MEASURED IMPROVEMENT: BASELINE RAG VS. TEMPORAL SUPERSEDED RAG")
        print("=" * 75)
        print("Evaluating on temporal conflict and supersession queries (BENCH-02 & BENCH-03)...\n")

        comparison_queries = [
            {
                "id": "BENCH-02",
                "name": "Cold Recovery Conflict (retention:0 vs security:1)",
                "query": "How many days are raw run records held in cold recovery storage after searchable retention ends?",
                "superseded_val": "60 days",
                "active_val": "20 days",
                "supersession_clause": "supersede",
            },
            {
                "id": "BENCH-03",
                "name": "AQL Timeout Across Versions (query:0 vs release notes)",
                "query": "What is the timeout for AQL queries across different product versions?",
                "superseded_val": "25 seconds",
                "active_val": "50 seconds",
                "supersession_clause": "version 2.3",
            },
        ]

        ab_results = []
        for cq in comparison_queries:
            q_text = cq["query"]
            chunks = search_corpus(q_text, top_k=5)

            # Condition A: Baseline RAG
            ans_baseline = run_baseline_rag(q_text, chunks)
            val_baseline = validate_citations(ans_baseline, chunks)
            baseline_has_citations = len(val_baseline["valid_citations"]) > 0
            baseline_resolves_conflict = (cq["supersession_clause"] in ans_baseline.lower())

            # Condition B: Asterline Grounded Prompt
            ans_improved = run_agent(query=q_text, user_id="eval_ab_user").get("answer", "")
            val_improved = validate_citations(ans_improved, chunks)
            improved_has_citations = len(val_improved["valid_citations"]) > 0
            improved_resolves_conflict = (cq["supersession_clause"] in ans_improved.lower()) and (cq["active_val"] in ans_improved.lower())

            ab_results.append({
                "case": cq["id"],
                "name": cq["name"],
                "baseline_citations_count": len(val_baseline["valid_citations"]),
                "baseline_has_citations": baseline_has_citations,
                "baseline_resolves_conflict": baseline_resolves_conflict,
                "improved_citations_count": len(val_improved["valid_citations"]),
                "improved_has_citations": improved_has_citations,
                "improved_resolves_conflict": improved_resolves_conflict,
            })

            print(f"Case: {cq['id']} ({cq['name']})")
            print(f"  [Baseline RAG] Citations: {len(val_baseline['valid_citations'])} | Conflict Resolution: {baseline_resolves_conflict}")
            print(f"  [Asterline RAG] Citations: {len(val_improved['valid_citations'])} | Conflict Resolution: {improved_resolves_conflict}")

        # Compute summary percentages
        routing_acc = (routing_correct_count / total_cases) * 100
        retrieval_acc = (retrieval_correct_count / research_cases) * 100 if research_cases else 100.0
        citation_acc = (citation_valid_count / research_cases) * 100 if research_cases else 100.0
        factual_acc = (factual_correct_count / total_cases) * 100

        baseline_cite_rate = sum(1 for r in ab_results if r["baseline_has_citations"]) / len(ab_results) * 100
        improved_cite_rate = sum(1 for r in ab_results if r["improved_has_citations"]) / len(ab_results) * 100

        baseline_conflict_rate = sum(1 for r in ab_results if r["baseline_resolves_conflict"]) / len(ab_results) * 100
        improved_conflict_rate = sum(1 for r in ab_results if r["improved_resolves_conflict"]) / len(ab_results) * 100

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "total_benchmark_cases": total_cases,
            "research_cases": research_cases,
            "memory_cases": memory_cases,
            "overall_metrics": {
                "routing_accuracy_pct": round(routing_acc, 2),
                "retrieval_recall_pct": round(retrieval_acc, 2),
                "citation_validity_pct": round(citation_acc, 2),
                "factual_grounding_accuracy_pct": round(factual_acc, 2),
            },
            "measured_improvement_ab_test": {
                "metric_1_citation_grounding_rate": {
                    "baseline_rag": f"{baseline_cite_rate:.1f}%",
                    "asterline_temporal_rag": f"{improved_cite_rate:.1f}%",
                    "absolute_gain": f"+{improved_cite_rate - baseline_cite_rate:.1f}%",
                },
                "metric_2_temporal_conflict_resolution": {
                    "baseline_rag": f"{baseline_conflict_rate:.1f}%",
                    "asterline_temporal_rag": f"{improved_conflict_rate:.1f}%",
                    "absolute_gain": f"+{improved_conflict_rate - baseline_conflict_rate:.1f}%",
                },
                "ab_cases_detail": ab_results,
            },
            "cases": results,
        }

        # Save to eval/results.json
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 75)
        print("FINAL EVALUATION METRICS SUMMARY")
        print("=" * 75)
        print(f"Total Cases Evaluated:       {total_cases}")
        print(f"Execution Duration:          {duration:.2f} seconds")
        print(f"Agent Routing Accuracy:      {routing_acc:.1f}% ({routing_correct_count}/{total_cases})")
        print(f"Retrieval Recall (Top-5):    {retrieval_acc:.1f}% ({retrieval_correct_count}/{research_cases})")
        print(f"Citation Validity Rate:      {citation_acc:.1f}% ({citation_valid_count}/{research_cases})")
        print(f"Factual Grounding Accuracy:  {factual_acc:.1f}% ({factual_correct_count}/{total_cases})")
        print("\nMEASURED IMPROVEMENT COMPARISON:")
        print(f"  - Citation Grounding Rate: Baseline {baseline_cite_rate:.1f}%  -->  Asterline {improved_cite_rate:.1f}%  (+{improved_cite_rate - baseline_cite_rate:.1f}%)")
        print(f"  - Conflict Resolution:     Baseline {baseline_conflict_rate:.1f}%  -->  Asterline {improved_conflict_rate:.1f}%  (+{improved_conflict_rate - baseline_conflict_rate:.1f}%)")
        print(f"\nResults successfully saved to: {RESULTS_PATH}")
        print("=" * 75)

        # Reset store injection
        set_default_memory_store(None)

        return summary


if __name__ == "__main__":
    evaluate_benchmark()
