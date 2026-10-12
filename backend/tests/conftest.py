"""
Test isolation. pytest imports this file before it collects (and therefore
imports) any test module, so the environment variables below are set before
app.config is first imported. That keeps every test's database, uploads and
result images in a throw-away temp directory instead of the real
backend/storage/ (which would otherwise pollute a developer's Inspection
History and Dashboard with test records).

The trained model in backend/models/ is intentionally NOT redirected: the
real-mode tests exercise the shipped model artifacts.
"""
import atexit
import os
import shutil
import tempfile

_TEST_ROOT = tempfile.mkdtemp(prefix="defectscope_tests_")
os.environ["STORAGE_DIR"] = os.path.join(_TEST_ROOT, "storage")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TEST_ROOT, "test.db")
atexit.register(shutil.rmtree, _TEST_ROOT, ignore_errors=True)
