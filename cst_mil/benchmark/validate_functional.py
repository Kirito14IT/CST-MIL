"""Run the real suite and bind its successful evidence to frozen benchmark code."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from xml.etree import ElementTree

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from cst_mil.bench.resources import configure_cpu  # noqa: E402

configure_cpu(4)

# Thread caps must precede imports that can load BLAS/Torch.
from cst_mil.bench.common import fingerprint, write_json  # noqa: E402
from cst_mil.bench.dataset import BENCHMARK, PROJECT  # noqa: E402
from cst_mil.bench.runner import runtime_fingerprint  # noqa: E402

stamp = str(time.time_ns())
xml_path = BENCHMARK / "verification" / f"tests_{stamp}.xml"
xml_path.parent.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    "-m",
    "pytest",
    "tests",
    "--basetemp",
    f".pytest-tmp-functional-{stamp}",
    "--junitxml",
    str(xml_path),
    "-q",
]
with xml_path.with_suffix(".log").open("w", encoding="utf-8") as output:
    result = subprocess.run(command, cwd=PROJECT, stdout=output, stderr=subprocess.STDOUT)
root = ElementTree.parse(xml_path).getroot()
suites = list(root.iter("testsuite"))
payload = {
    "passed": result.returncode == 0,
    "command": command,
    "counts": {
        key: sum(int(s.get(key, "0")) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    },
    "runtime_sha256": fingerprint(runtime_fingerprint()),
    "log": str(xml_path.with_suffix(".log")),
    "junit": str(xml_path),
    "completed_at_unix": time.time(),
    "scope": "functional, not performance",
}
write_json(BENCHMARK / "functional_checks.json", payload)
print(payload, flush=True)
raise SystemExit(result.returncode)
