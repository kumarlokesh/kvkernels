"""The VM session driver's commands, without gcloud or a VM."""

from __future__ import annotations

from scripts import vm_session

VM = vm_session.Vm("gpu-vm", "us-central1-c", "proj", iap=True)


def test_every_gcloud_call_names_the_project_and_zone() -> None:
    ssh = VM.ssh_args("echo hi")
    scp = VM.scp_args(["a.tar.gz"], VM.remote("~/"), recurse=True)
    for args in (ssh, scp):
        assert "--project=proj" in args
        assert "--zone=us-central1-c" in args
        assert "--tunnel-through-iap" in args
    assert ssh[:4] == ["gcloud", "compute", "ssh", "gpu-vm"]
    assert ssh[-1] == "--command=echo hi"
    assert scp[-2:] == ["a.tar.gz", "gpu-vm:~/"]
    assert "--recurse" in scp


def test_without_iap_ssh_goes_straight_to_port_22() -> None:
    vm = vm_session.Vm("gpu-vm", "z", "p", iap=False)
    assert "--tunnel-through-iap" not in vm.ssh_args("true")


def test_a_run_outlives_the_ssh_connection_and_logs_next_to_its_results() -> None:
    command = vm_session.run_command("20260101T000000Z", "env,test-gpu", "abc123")
    assert "setsid nohup" in command
    assert "--out ~/sessions/20260101T000000Z" in command
    assert "> ~/sessions/20260101T000000Z.log" in command
    assert "--steps env,test-gpu" in command
    assert "KVKERNELS_GIT_SHA=abc123" in command
    assert "UV_NO_SYNC=1" in command
    assert command.endswith("echo 20260101T000000Z")


def test_a_run_without_steps_runs_the_whole_session() -> None:
    assert "--steps" not in vm_session.run_command("s", None, "abc")


def test_setup_installs_what_triton_and_flashinfer_compile_with() -> None:
    script = vm_session.SETUP_SCRIPT
    assert "build-essential" in script
    assert "python3-dev" in script
    assert "cuda-link/libcuda.so" in script
    assert vm_session.INSTALL in script


def test_the_token_is_copied_as_a_file_and_never_echoed() -> None:
    assert "token" not in vm_session.SETUP_SCRIPT
    assert "token" not in vm_session.run_command("s", None, "abc")
