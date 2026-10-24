"""Test session setup: keep the explanation call log out of the repository's reports/ folder."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("FATIGUE_CALL_LOG", str(Path(tempfile.gettempdir()) / "fatigue_tests_llm_call_log.jsonl"))
