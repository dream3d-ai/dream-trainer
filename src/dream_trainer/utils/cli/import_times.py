"""Helpers for the `importtime` CLI subcommand.

The `importtime` subcommand re-launches the current process with
`python -X importtime` under a single-rank `torch.distributed.launcher`
harness, captures the timing dump on stderr, and renders it with the
`tuna` visualiser.
"""

import os
import sys
from typing import Union

from torch.distributed.elastic.multiprocessing import LogsDest, LogsSpecs, Std
from torch.distributed.launcher.api import LaunchConfig, launch_agent

LOG_DIR = "/tmp/importtime"
"""Scratch directory where stderr/stdout/error logs for the import-time run are written."""


class BaseLogSpecs(LogsSpecs):
    """`LogsSpecs` implementation that redirects all streams to a single file.

    Torchelastic expects a `LogsSpecs` that can produce per-rank log
    destinations; since the `importtime` subcommand is intentionally
    single-rank we only ever emit one mapping.
    """

    def __init__(
        self,
        log_dir: str,
        redirects: Union[Std, dict[int, Std]] = Std.ALL,
        tee: Union[Std, dict[int, Std]] = Std.NONE,
    ) -> None:
        """Initialise the log spec, creating `log_dir` if needed.

        Args:
            log_dir: Directory to write stdout/stderr/error files to.
            redirects: Which streams torchelastic should capture.
            tee: Which streams to also forward to the parent console.

        Raises:
            NotADirectoryError: If `log_dir` exists and is a regular file.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        elif os.path.isfile(log_dir):
            raise NotADirectoryError(f"log_dir: {log_dir} is a file")

        super().__init__(log_dir, redirects, tee)
        self._log_dir = log_dir

    def reify(self, envs: dict[int, dict[str, str]]) -> LogsDest:
        """Return the log destinations for the single-rank process.

        Args:
            envs: Per-rank environment dicts, as passed by torchelastic.
                Asserts that exactly one rank is being launched.

        Returns:
            A `LogsDest` with stdout, stderr, and error-file paths under
            `self._log_dir`. `tee_*` mappings are empty because `tee`
            defaults to `NONE`.
        """
        assert len(envs) == 1, "Run --importtime with a single process"

        return LogsDest(
            stdouts={0: os.path.join(self._log_dir, "stdout.log")},
            stderrs={0: os.path.join(self._log_dir, "stderr.log")},
            tee_stdouts={},
            tee_stderrs={},
            error_files={0: os.path.join(self._log_dir, "error.json")},
        )

    @property
    def root_log_dir(self) -> str:
        """Root directory for logs (matches the `log_dir` passed at construction)."""
        return self._log_dir


def get_importtime() -> None:
    """Re-launch the current script with `-X importtime` and visualise the result.

    Adds `-X importtime` to the original argv, pins the run to a single
    CUDA device, launches under torchelastic with rdzv on `localhost:29500`,
    and finally shells out to `tuna` on the captured stderr to render the
    import waterfall.

    The process exits early (with code 0) if `-X` is already present in the
    original argv, to avoid infinite relaunch loops.
    """
    args = sys.orig_argv[1:]
    if "-X" in args:
        exit(0)

    args = ["-X", "importtime"] + args
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    launch_agent(
        config=LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:29500",
            run_id="import-times",
            max_restarts=0,
            logs_specs=BaseLogSpecs(log_dir=LOG_DIR),
        ),
        entrypoint=sys.executable,
        args=args,
    )

    os.system(f"tuna {os.path.join(LOG_DIR, 'stderr.log')}")
