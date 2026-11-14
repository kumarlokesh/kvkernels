"""Run scripts/gpu_session.py on a Compute Engine GPU VM, from a development machine.

Needs the gcloud CLI, logged in, and a running VM with an NVIDIA driver.

    python scripts/vm_session.py setup  --vm NAME --zone ZONE --project PROJECT [--iap]
    python scripts/vm_session.py run    --vm ... [--steps env,test-gpu,...]
    python scripts/vm_session.py status --vm ...
    python scripts/vm_session.py fetch  --vm ... [--session STAMP]
    python scripts/gpu_session.py collect gpu-sessions/<vm>/<stamp>/session

    this machine                              VM
    setup  bundle, HF token ──scp──▶  ~/kvkernels, ~/.cache/huggingface/token
                                      uv sync (the same extras as on Kaggle)
    run    ──ssh──▶  gpu_session.py run --out ~/sessions/<stamp>, detached
                     from the SSH connection, log in ~/sessions/<stamp>.log
    fetch  ◀──scp──  ~/sessions/<stamp>  ──▶  gpu-sessions/<vm>/<stamp>/session

The session keeps running when the SSH connection closes, so a dropped tunnel
does not stop a run of several hours. --iap sends SSH through Google's
Identity-Aware Proxy over HTTPS, for networks that block port 22.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.kaggle_session import bundle_bytes, git_sha

ROOT = Path(__file__).resolve().parent.parent
HF_TOKEN = Path.home() / ".cache" / "huggingface" / "token"
UV_VERSION = "0.13.5"
REMOTE_REPO = "kvkernels"  # under the remote home directory
REMOTE_SESSIONS = "sessions"
CUDA = "12.6"  # torch's CUDA build (the cu126 extra)
CUDA_PKG = CUDA.replace(".", "-")
CUDA_KEYRING = (
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
    "cuda-keyring_1.1-1_all.deb"
)
INSTALL = "uv sync --locked --extra cu126 --extra bench --extra baselines"

# Runs on the VM after the bundle is copied. FlashInfer links the kernels it
# compiles with -lcuda; the driver packages install only libcuda.so.1, so a
# link named libcuda.so goes in ~/cuda-link, which the run adds to LIBRARY_PATH.
SETUP_SCRIPT = f"""set -e
# A C compiler and Python's headers: Triton builds its launcher against both.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential python3-dev >/dev/null
# The CUDA toolkit matching torch's build: FlashInfer compiles its kernels with
# nvcc on first use, and the profiling steps use its Nsight Compute.
if [ ! -x /usr/local/cuda-{CUDA}/bin/nvcc ]; then
  curl -fsSLo /tmp/cuda-keyring.deb {CUDA_KEYRING}
  sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-{CUDA_PKG} >/dev/null
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
rm -rf ~/{REMOTE_REPO} && tar -xzf ~/kvkernels-bundle.tar.gz -C ~ && rm ~/kvkernels-bundle.tar.gz
mkdir -p ~/cuda-link
ln -sf "$(ldconfig -p | awk '/libcuda.so.1 /{{print $NF; exit}}')" ~/cuda-link/libcuda.so
sudo nvidia-smi -pm 1 >/dev/null
cd ~/{REMOTE_REPO} && {INSTALL}
"""


@dataclass(frozen=True)
class Vm:
    name: str
    zone: str
    project: str
    iap: bool

    def flags(self) -> list[str]:
        return [
            f"--project={self.project}",
            f"--zone={self.zone}",
            "--quiet",
            *(["--tunnel-through-iap"] if self.iap else []),
        ]

    def ssh_args(self, command: str) -> list[str]:
        return ["gcloud", "compute", "ssh", self.name, *self.flags(), f"--command={command}"]

    def scp_args(self, sources: Sequence[str], dest: str, recurse: bool = False) -> list[str]:
        extra = ["--recurse"] if recurse else []
        return ["gcloud", "compute", "scp", *extra, *self.flags(), *sources, dest]

    def remote(self, path: str) -> str:
        return f"{self.name}:{path}"


def run_command(stamp: str, steps: str | None, sha: str) -> str:
    """The shell command that starts a session on the VM, detached from SSH."""
    out = f"~/{REMOTE_SESSIONS}/{stamp}"
    session = f"uv run python scripts/gpu_session.py run --out {out}"
    if steps:
        session += f" --steps {shlex.quote(steps)}"
    env = (
        f'export PATH="$HOME/.local/bin:/usr/local/cuda-{CUDA}/bin:$PATH"'
        f" CUDA_HOME=/usr/local/cuda-{CUDA}"
        ' LIBRARY_PATH="$HOME/cuda-link:$LIBRARY_PATH"'
        f" KVKERNELS_GIT_SHA={shlex.quote(sha)} UV_NO_SYNC=1 PYTHONUNBUFFERED=1"
    )
    return (
        f"mkdir -p ~/{REMOTE_SESSIONS} && cd ~/{REMOTE_REPO} && {env} && "
        f"setsid nohup sh -c {shlex.quote(session)} > {out}.log 2>&1 < /dev/null & "
        f"echo {stamp}"
    )


STATUS_COMMAND = f"""latest=$(ls -t ~/{REMOTE_SESSIONS}/*.log 2>/dev/null | head -1)
if pgrep -f "gpu_session.py run" >/dev/null; then echo RUNNING; else echo IDLE; fi
[ -n "$latest" ] && echo "$latest" && tail -n 15 "$latest"
"""


def call(args: list[str]) -> int:
    print("$", " ".join(a if not a.startswith("--command=") else "--command=…" for a in args))
    return subprocess.run(args, check=False).returncode


def setup(vm: Vm) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "kvkernels-bundle.tar.gz"
        bundle.write_bytes(bundle_bytes())
        if call(vm.scp_args([str(bundle)], vm.remote("~/"))):
            return 1
    if HF_TOKEN.exists():
        if call(vm.ssh_args("mkdir -p ~/.cache/huggingface && chmod 700 ~/.cache/huggingface")):
            return 1
        if call(vm.scp_args([str(HF_TOKEN)], vm.remote("~/.cache/huggingface/token"))):
            return 1
        if call(vm.ssh_args("chmod 600 ~/.cache/huggingface/token")):
            return 1
    else:
        print(f"no Hugging Face token at {HF_TOKEN}; gated models will not download")
    return call(vm.ssh_args(SETUP_SCRIPT))


def latest_stamp(vm: Vm) -> str | None:
    found = subprocess.run(
        vm.ssh_args(f"ls -t ~/{REMOTE_SESSIONS} | grep -v '\\.log$' | head -1"),
        check=False,
        capture_output=True,
        text=True,
    )
    lines = found.stdout.split()
    return lines[-1] if found.returncode == 0 and lines else None


def fetch(vm: Vm, stamp: str | None) -> int:
    stamp = stamp or latest_stamp(vm)
    if stamp is None:
        print("no session on the VM")
        return 1
    dest = ROOT / "gpu-sessions" / vm.name / stamp
    dest.mkdir(parents=True, exist_ok=True)
    remote = f"~/{REMOTE_SESSIONS}/{stamp}"
    code = call(vm.scp_args([vm.remote(remote)], str(dest / "session"), recurse=True))
    code = code or call(vm.scp_args([vm.remote(f"{remote}.log")], str(dest / "session.log")))
    if not code:
        print(f"fetched to {dest}; then: python scripts/gpu_session.py collect {dest}/session")
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("setup", "run", "status", "fetch"):
        p = sub.add_parser(name)
        p.add_argument("--vm", required=True, help="instance name")
        p.add_argument("--zone", required=True)
        p.add_argument("--project", required=True)
        p.add_argument("--iap", action="store_true", help="SSH through an IAP tunnel")
        if name == "run":
            p.add_argument("--steps", help="passed to gpu_session.py run --steps")
        if name == "fetch":
            p.add_argument("--session", help="the session's stamp; the newest by default")
    args = parser.parse_args(argv)
    vm = Vm(args.vm, args.zone, args.project, args.iap)
    if args.action == "setup":
        return setup(vm)
    if args.action == "run":
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return call(vm.ssh_args(run_command(stamp, args.steps, git_sha())))
    if args.action == "status":
        return call(vm.ssh_args(STATUS_COMMAND))
    return fetch(vm, args.session)


if __name__ == "__main__":
    raise SystemExit(main())
