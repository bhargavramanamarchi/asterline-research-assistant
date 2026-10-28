# LangSmith Trace Evidence

## Project Overview

These traces provide inspectable execution evidence for the **Asterline Research Assistant**, a multi-agent AI research system orchestrated with **LangGraph**.

By opening the public trace links below, evaluators and reviewers can inspect:
- The input user query.
- The multi-agent workflow execution trace and state progression (`PlannerNode` &rarr; `ResearcherNode`).
- Intermediate vector retrieval steps from ChromaDB.
- LLM prompt structure, runtime latency, and generated token outputs.
- Grounded citations and citation-validation metadata.

---

## 1. Successful Research Run — Enterprise SLA

- **Test question:** What is the Enterprise SLA?
- **Purpose:** Demonstrates the Researcher workflow retrieving documentary evidence and generating an answer with citations.
- **Expected evidence:** Successful run status, retrieved evidence references (`sla:0`), and citation-validation information.
- **Public LangSmith trace URL:** https://smith.langchain.com/public/56b1f482-3760-488d-9899-4a7af18abefa/r/01a1827d-e06e-71f7-a4ed-c0ad127beb6e?start_time=2026-10-28T12%3A30%3A49.966176Z

---

## 2. Difficult Research Run — Cold Recovery Retention Policy

- **Test question:** What is the current cold recovery retention period? Explain how the newer policy supersedes the older policy.
- **Purpose:** Demonstrates retrieval and handling of conflicting policy versions, including the newer retention policy superseding the older one.
- **Expected evidence:** The answer, the cited policy evidence (`retention:0` superseding `security:1`), and the execution information showing how the Researcher handled the query.
- **Public LangSmith trace URL:** https://smith.langchain.com/public/723b5949-0b25-44ed-ae6e-98510c6de4c9/r/01a1827e-259c-727b-86e9-bcd71b049f83?start_time=2026-10-28T12%3A31%3A07.676742Z

---

## Evaluator Access

The links provided above are individual public trace URLs that have been verified to open directly in a clean Incognito/private browser window without requiring a LangSmith login or account.

Reviewers are invited to open each trace link in any web browser to inspect the full step-by-step execution graph, node inputs, retrieved evidence, and token generation metadata. Access is provided via these individual public trace links rather than full project-level repository access.

---

## Security Note

In accordance with strict security and confidentiality guidelines, this repository does not contain `.env` files, API keys, private tokens, or access credentials. This document contains only public trace URLs and explanatory documentation.
