from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from .types import CHECKPOINT_REGEX, Checkpoint
from .utils import sort_checkpoints


def is_s3_uri(path: str | Path) -> bool:
    return str(path).startswith("s3://")


def join_s3_uri(root: str, *parts: str) -> str:
    suffix = "/".join(part.strip("/") for part in parts if part)
    if not suffix:
        return root.rstrip("/")
    return f"{root.rstrip('/')}/{suffix}"


def delete_s3_checkpoint_prefix(
    checkpoint_path: str,
    storage_options: dict[str, Any] | None = None,
) -> None:
    storage = S3CheckpointStorage.from_checkpoint_path(
        checkpoint_path,
        storage_options,
    )
    storage.delete_prefix(checkpoint_path)


class S3CheckpointStorage:
    def __init__(
        self,
        root_dir: str,
        storage_options: dict[str, Any] | None = None,
        *,
        fs: Any | None = None,
    ) -> None:
        self.root_dir = root_dir.rstrip("/")
        self.storage_options = dict(storage_options or {})
        self.fs = fs or _build_s3_filesystem(self.storage_options)

    @classmethod
    def from_checkpoint_path(
        cls,
        checkpoint_path: str,
        storage_options: dict[str, Any] | None = None,
    ) -> "S3CheckpointStorage":
        root_dir = checkpoint_path.rstrip("/").rsplit("/", 1)[0]
        return cls(root_dir, storage_options)

    def checkpoint_path(self, checkpoint_id: str) -> str:
        return join_s3_uri(self.root_dir, checkpoint_id)

    def exists(self, checkpoint_id: str) -> bool:
        return self.fs.exists(join_s3_uri(self.checkpoint_path(checkpoint_id), ".metadata"))

    def reader(self, checkpoint_id: str):
        S3StorageReader = _s3_storage_reader()
        return S3StorageReader(
            path=self.checkpoint_path(checkpoint_id),
            **_normalize_storage_options(self.storage_options),
        )

    def writer(self, checkpoint_id: str):
        S3StorageWriter = _s3_storage_writer()
        return S3StorageWriter(
            path=self.checkpoint_path(checkpoint_id),
            **_normalize_storage_options(self.storage_options),
        )

    def find_checkpoints(
        self,
        mode: Literal["min", "max", "last"] = "last",
    ) -> list[Checkpoint]:
        prefix = _s3_prefix(self.root_dir)
        checkpoints: dict[str, Checkpoint] = {}
        for key in self._list_keys(prefix):
            relative_key = key[len(prefix) :]
            checkpoint_id = relative_key.split("/", 1)[0]
            if not relative_key.endswith("/.metadata"):
                continue
            if CHECKPOINT_REGEX.search(checkpoint_id) is None:
                continue
            checkpoints[checkpoint_id] = Checkpoint.from_path(Path(checkpoint_id))

        return sort_checkpoints(list(checkpoints.values()), mode)

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        self.delete_prefix(self.checkpoint_path(checkpoint_id))

    def delete_prefix(self, path: str) -> None:
        for key in self._list_keys(_s3_prefix(path)):
            bucket, _ = _parse_s3_uri(path)
            self.fs.rm_file(f"s3://{bucket}/{key}")

    def _list_keys(self, prefix: str) -> list[str]:
        bucket, _ = _parse_s3_uri(self.root_dir)
        keys: list[str] = []
        for result in self.fs._client.list_objects(bucket, prefix):
            keys.extend(obj.key for obj in result.object_info)
        return keys


def _s3_prefix(path: str) -> str:
    _, key = _parse_s3_uri(path)
    if key and not key.endswith("/"):
        key = f"{key}/"
    return key


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc == "":
        raise ValueError(f"Expected an s3:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _build_s3_filesystem(storage_options: dict[str, Any]):
    S3FileSystem = _s3_file_system()
    options = _normalize_storage_options(storage_options)
    try:
        region = options.pop("region")
    except KeyError as exc:
        raise ValueError(
            "S3 checkpoint storage_options must include 'region' unless 'endpoint_url' is set"
        ) from exc

    fs_options = {
        key: options[key]
        for key in (
            "s3client_config",
            "endpoint_url",
            "access_key_id",
            "secret_access_key",
        )
        if key in options
    }
    return S3FileSystem(region, **fs_options)


def _normalize_storage_options(
    storage_options: dict[str, Any] | None,
) -> dict[str, Any]:
    options = dict(storage_options or {})
    force_path_style = options.pop("force_path_style", None)
    if options.get("endpoint_url") is not None:
        options.setdefault("region", "auto")
        options.setdefault(
            "s3client_config",
            _s3_client_config()(
                force_path_style=True
                if force_path_style is None
                else bool(force_path_style)
            ),
        )
    return options


def _s3_client_config():
    try:
        from s3torchconnector import S3ClientConfig
    except ImportError as exc:
        raise ImportError("Install dream-trainer[s3] to use S3 checkpoints") from exc
    return S3ClientConfig


def _s3_file_system():
    try:
        from s3torchconnector.dcp import S3FileSystem
    except ImportError as exc:
        raise ImportError("Install dream-trainer[s3] to use S3 checkpoints") from exc
    return S3FileSystem


def _s3_storage_reader():
    try:
        from s3torchconnector.dcp import S3StorageReader
    except ImportError as exc:
        raise ImportError("Install dream-trainer[s3] to use S3 checkpoints") from exc
    return S3StorageReader


def _s3_storage_writer():
    try:
        from s3torchconnector.dcp import S3StorageWriter
    except ImportError as exc:
        raise ImportError("Install dream-trainer[s3] to use S3 checkpoints") from exc
    return S3StorageWriter
