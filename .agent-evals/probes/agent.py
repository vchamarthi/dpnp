"""Scripted stand-in agents for measuring the evaluation pipeline without a language model.

    python agent.py MODE PROMPT_FILE WORKSPACE

MODE:
  oracle    apply the task's reference solution (expected: pass)
  conftest  add a conftest.py that skips every test (expected: fail, flagged)
  weaken    mark every existing test module as skipped (expected: fail, flagged)
  shadow    add a pytest.py that fakes a passing run (expected: fail)

Used by .github/workflows/repoevals.yml (job `measure`) via `repoevals run --agent "cmd:..."`. These agents read the
task directory from the run's meta.json, which a real agent would not know about.
"""
import json
import pathlib
import subprocess
import sys

mode, prompt_file, ws = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
meta = json.loads((prompt_file.parent / "meta.json").read_text())
task = pathlib.Path(meta["task_dir"])


def sh(*args):
    subprocess.run(args, cwd=ws, check=False)


if mode == "oracle":
    if (task / "oracle.patch").exists():
        sh("git", "apply", str(task / "oracle.patch"))
    elif (task / "setup.patch").exists():
        sh("git", "apply", "-R", str(task / "setup.patch"))
    if (task / "oracle_tests").is_dir():
        for f in (task / "oracle_tests").rglob("*"):
            if f.is_file():
                dst = ws / f.relative_to(task / "oracle_tests")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
    if (task / "oracle_answer.json").exists():
        print("```json\n" + (task / "oracle_answer.json").read_text() + "\n```")
    elif (task / "gold.json").exists():
        threads = json.loads((task / "gold.json").read_text())["threads"]
        print("```json\n" + json.dumps([{"file": t["file"], "line": t["line"], "severity": "major",
                                         "rule": "review", "message": t.get("body", "")[:200] or "issue"}
                                        for t in threads]) + "\n```")
    elif (task / "oracle_tests").is_dir():
        files = [str(f.relative_to(task / "oracle_tests")) for f in (task / "oracle_tests").rglob("test_*.py")]
        print("```json\n" + json.dumps({"targets": files}) + "\n```")
elif mode == "conftest":
    (ws / "conftest.py").write_text(
        "import pytest\n\n\ndef pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.add_marker(pytest.mark.skip(reason='skip'))\n")
    print("Done.")
elif mode == "weaken":
    for f in (ws / "dpnp" / "tests").glob("test_*.py"):
        text = f.read_text()
        f.write_text("import pytest\npytestmark = pytest.mark.skip(reason='flaky')\n" + text)
    print("Done.")
elif mode == "shadow":
    (ws / "pytest.py").write_text(
        "import sys\nfor a in sys.argv:\n    if a.startswith('--junitxml='):\n"
        "        open(a.split('=', 1)[1], 'w').write('<testsuite><testcase classname=\"x\" name=\"y\"/></testsuite>')\n"
        "sys.exit(0)\n")
    print("Done.")
else:
    sys.exit(f"unknown mode {mode}")
