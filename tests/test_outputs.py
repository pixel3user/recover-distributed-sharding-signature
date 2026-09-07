"""Behavioral verifier for the PyTorch DistributedSampler compatibility extension."""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET = Path("/app/torch/utils/data/distributed.py")
TESTS = Path(__file__).parent
RUNNER = TESTS / "agent_runner.py"
CHILD_TIMEOUT_SEC = 120
HELDOUT = [json.loads(line) for line in (TESTS / "heldout.jsonl").read_text().splitlines() if line.strip()]
_golden = json.loads((TESTS / "golden.json").read_text())
_SALT = b"distributed-sampler-signature-v1"


def _digest(index, value):
    return hashlib.sha256(_SALT + b"%d:%d" % (index, value)).hexdigest()


GOLDEN_DIGESTS = [_digest(index, value) for index, value in enumerate(_golden)]
del _golden
_SENSITIVE = [TESTS / "heldout.jsonl", TESTS / "golden.json"]
_LEFT = []
for path in _SENSITIVE:
    try:
        path.unlink()
    except OSError:
        _LEFT.append(path.name)
_CACHE = None


def _answers():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    scratch = tempfile.mkdtemp(prefix="distributed-signature-")
    runner = Path(scratch) / "agent_runner.py"
    shutil.copy(RUNNER, runner)
    payload = json.dumps({"queries": [row["query"] for row in HELDOUT]})
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    try:
        result = subprocess.run([sys.executable, "-I", "-S", str(runner)], input=payload, text=True,
                                cwd=scratch, env=env, capture_output=True, timeout=CHILD_TIMEOUT_SEC)
        _CACHE = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, ValueError) as exc:
        _CACHE = {"error": str(exc)}
    return _CACHE


def _require_answers():
    result = _answers()
    assert "error" not in result, result.get("error")
    answers = result["answers"]
    assert len(answers) == len(HELDOUT)
    return answers


def _bad(indices):
    answers = _require_answers()
    return [index for index in indices if answers[index]["t"] != "int" or
            _digest(index, answers[index]["v"]) != GOLDEN_DIGESTS[index]]


def test_extension_exists_in_the_requested_pytorch_module():
    """The requested function must be added to the existing PyTorch distributed sampler module."""
    assert TARGET.is_file(), f"missing {TARGET}"
    assert "def sharding_signature" in TARGET.read_text(), "missing sharding_signature(query)"


def test_extension_uses_no_torch_at_call_time_or_dynamic_code_loading():
    """The added function and its private helpers must be self-contained standard-library Python."""
    tree = ast.parse(TARGET.read_text())
    extensions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and (node.name == "sharding_signature" or node.name.startswith("_sharding_signature_"))]
    assert any(node.name == "sharding_signature" for node in extensions)
    forbidden = {"exec", "eval", "compile", "__import__", "torch", "subprocess", "ctypes", "importlib"}
    for node in extensions:
        for child in ast.walk(node):
            assert not isinstance(child, (ast.Import, ast.ImportFrom)), "extension helpers may not import modules"
            if isinstance(child, ast.Name):
                assert child.id not in forbidden, f"extension references forbidden name {child.id}"


def test_ground_truth_is_removed_before_extension_execution():
    """Hidden query/signature files are removed before the child imports agent-modified code."""
    assert not _LEFT
    assert all(not path.exists() for path in _SENSITIVE)


def test_returns_plain_nonnegative_integers():
    """The API must return plain non-negative int values for all valid blind queries."""
    for answer in _require_answers():
        assert answer["t"] == "int"
        assert answer["v"] >= 0


def test_equivalent_queries_share_signatures():
    """Different raw queries with the same resolved rank-code sequence must agree."""
    answers = _require_answers()
    groups = {}
    for index, row in enumerate(HELDOUT):
        groups.setdefault(row["grp"], []).append(index)
    for indices in groups.values():
        assert len({(answers[index]["t"], answers[index]["v"]) for index in indices}) == 1


def test_reproduces_tail_mode_and_shuffle_cases():
    """Uneven tails, contiguous/interleaved assignment, and deterministic shuffling are load-bearing."""
    for field in ("tail", "mode", "shuf"):
        indices = [index for index, row in enumerate(HELDOUT) if row[field]]
        assert indices, f"fixture lacks {field} coverage"
        assert not _bad(indices), f"{field} cases did not match"


def test_reproduces_every_blind_signature():
    """Every held-out assignment signature must match exactly with no tolerance."""
    assert not _bad(range(len(HELDOUT)))
