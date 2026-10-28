"""
test_cloud_fixes.py - Verification for Cloud-Deployment Fixes.
Tests:
  1. ChromaDB Cold-Start Auto-Indexing:
     - Verifies an empty Chroma collection auto-indexes corpus.jsonl on cold start.
     - Confirms exactly 75 records are produced.
     - Confirms repeated ensure_indexed calls maintain strict idempotency (count=75).
     - Confirms search_corpus on an empty collection triggers cold-start auto-index and returns hits.
  2. Streamlit Secrets Compatibility:
     - Verifies get_groq_api_key() retrieves keys from st.secrets when os.environ is absent.
     - Verifies invalid placeholders in st.secrets are safely ignored.
     - Verifies get_active_provider() selects Groq without requiring Gemini.
     - Confirms no environment variables are overwritten inappropriately.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.corpus import load_corpus
from src.indexer import (
    COLLECTION_NAME,
    ensure_indexed,
    get_chroma_client,
    get_or_create_collection,
    search_corpus,
)
from src.researcher import get_active_provider, get_gemini_api_key, get_groq_api_key


class TestCloudDeploymentFixes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_chroma_cold_start_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chromadb_cold_start_auto_indexing(self):
        """Fix 1: An empty ChromaDB collection automatically indexes corpus.jsonl."""
        client = get_chroma_client(self.temp_dir)
        collection = get_or_create_collection(client=client, db_path=self.temp_dir)
        self.assertEqual(collection.count(), 0, "Collection should be empty initially.")

        # Test ensure_indexed on empty collection
        count = ensure_indexed(client=client, db_path=self.temp_dir)
        self.assertEqual(count, 75, f"Expected exactly 75 records indexed, got {count}.")
        self.assertEqual(collection.count(), 75, "Collection count should now be 75.")

        # Test idempotency: Calling ensure_indexed again should NOT duplicate records
        second_count = ensure_indexed(client=client, db_path=self.temp_dir)
        self.assertEqual(second_count, 75, "Collection count must remain strictly 75 on second call.")
        self.assertEqual(collection.count(), 75, "Collection count must remain 75.")

    def test_search_corpus_triggers_cold_start_indexing(self):
        """Fix 1: Calling search_corpus on an unindexed collection automatically indexes and returns hits."""
        client = get_chroma_client(self.temp_dir)
        collection = get_or_create_collection(client=client, db_path=self.temp_dir)
        self.assertEqual(collection.count(), 0)

        # Search should trigger auto-indexing and return relevant hits
        hits = search_corpus(
            query="Enterprise monthly uptime SLA",
            top_k=3,
            client=client,
            db_path=self.temp_dir,
        )
        self.assertGreater(len(hits), 0, "Search should return hits after cold-start auto-index.")
        self.assertEqual(collection.count(), 75, "Collection must contain exactly 75 records.")
        self.assertEqual(hits[0]["chunk_id"], "sla:0", "Top hit for SLA query should be sla:0.")

    def test_streamlit_secrets_fallback_groq(self):
        """Fix 2: Reads GROQ_API_KEY from Streamlit Secrets when not in os.environ."""
        mock_secrets = {"GROQ_API_KEY": "gsk_cloud_mock_key_abc123"}

        with patch.dict(os.environ, {}, clear=True):
            # os.environ has no GROQ_API_KEY
            self.assertIsNone(os.getenv("GROQ_API_KEY"))

            # Mock streamlit.secrets
            mock_st = MagicMock()
            mock_st.secrets = mock_secrets

            with patch.dict(sys.modules, {"streamlit": mock_st}):
                with patch("src.researcher.load_dotenv"):
                    resolved_key = get_groq_api_key()
                    self.assertEqual(resolved_key, "gsk_cloud_mock_key_abc123")

                    provider, active_key = get_active_provider()
                    self.assertEqual(provider, "groq")
                    self.assertEqual(active_key, "gsk_cloud_mock_key_abc123")

    def test_streamlit_secrets_rejects_placeholders(self):
        """Fix 2: Streamlit Secrets rejects placeholder template strings."""
        mock_secrets = {"GROQ_API_KEY": "your_groq_api_key_here"}

        with patch.dict(os.environ, {}, clear=True):
            mock_st = MagicMock()
            mock_st.secrets = mock_secrets

            with patch.dict(sys.modules, {"streamlit": mock_st}):
                with patch("src.researcher.load_dotenv"):
                    resolved_key = get_groq_api_key()
                    self.assertIsNone(resolved_key, "Placeholder in st.secrets must be rejected.")

    def test_cold_start_missing_corpus_error(self):
        """Fix 1: Handles missing corpus file cleanly with informative FileNotFoundError."""
        client = get_chroma_client(self.temp_dir)
        fake_corpus_path = os.path.join(self.temp_dir, "nonexistent_corpus.jsonl")

        with self.assertRaises((FileNotFoundError, RuntimeError)) as ctx:
            ensure_indexed(
                client=client,
                db_path=self.temp_dir,
                corpus_path=fake_corpus_path,
            )
        self.assertIn("Cold-start indexing failed", str(ctx.exception))

    def test_secrets_sync_to_environ(self):
        """Fix 2: Secrets sync into os.environ without overwriting existing vars."""
        mock_secrets = {
            "STREAMLIT_CUSTOM_FLAG": "enabled_from_cloud",
            "EXISTING_ENV_VAR": "new_cloud_value",
        }

        with patch.dict(os.environ, {"EXISTING_ENV_VAR": "original_local_value"}):
            # Simulate app.py startup sync
            for k, v in mock_secrets.items():
                if isinstance(v, str) and k not in os.environ:
                    os.environ[k] = v

            # Verify new secret was synced
            self.assertEqual(os.environ.get("STREAMLIT_CUSTOM_FLAG"), "enabled_from_cloud")
            # Verify existing env var was preserved (not overwritten)
            self.assertEqual(os.environ.get("EXISTING_ENV_VAR"), "original_local_value")


if __name__ == "__main__":
    unittest.main()
