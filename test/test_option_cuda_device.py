import os
from pathlib import Path
import subprocess
import sys

from conftest import cuda_device_from_argv

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCudaDeviceFromArgv:
    """Command-line parsing of the --cuda-device option."""

    def test_default_is_zero_when_option_absent(self):
        assert cuda_device_from_argv(["pytest", "test/"]) == "0"

    def test_equals_form(self):
        argv = ["pytest", "--cuda-device=5", "test/"]
        assert cuda_device_from_argv(argv) == "5"

    def test_space_form(self):
        argv = ["pytest", "--cuda-device", "5", "test/"]
        assert cuda_device_from_argv(argv) == "5"

    def test_comma_separated_list_passes_through(self):
        argv = ["pytest", "--cuda-device=1,3", "test/"]
        assert cuda_device_from_argv(argv) == "1,3"

    def test_trailing_option_value_missing_is_ignored_here(self):
        # A bare trailing --cuda-device is left to pytest's own parser, which
        # reports the missing-argument error; the scan must not crash on it.
        assert cuda_device_from_argv(["pytest", "--cuda-device"]) == "0"

    def test_option_like_value_is_not_taken_as_the_device(self):
        argv = ["pytest", "--cuda-device", "--quiet", "5"]
        assert cuda_device_from_argv(argv) == "0"


class TestCudaDeviceWiring:
    """The selected device reaches the environment before jax is imported."""

    def test_env_matches_registered_option(self, request):
        requested = request.config.getoption("--cuda-device")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == requested

    def test_non_default_device_selected_in_subprocess(self):
        # The wiring only proves itself in a fresh interpreter: a child pytest
        # run must see CUDA_VISIBLE_DEVICES set from its own --cuda-device
        # before conftest imports jax.
        probe = (
            "test/test_option_cuda_device.py::TestCudaDeviceWiring"
            "::test_env_matches_registered_option"
        )
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", probe, "--cuda-device=1"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )  # the returncode assert reports stdout+stderr
        assert res.returncode == 0, res.stdout + res.stderr
