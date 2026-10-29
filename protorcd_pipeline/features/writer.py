"""Chunked embedding writer with JSONL metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ChunkedEmbeddingWriter:
    def __init__(
        self,
        output_dir: Path,
        chunk_size: int,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.embeddings_dir = output_dir / "embeddings"
        self.metadata_path = output_dir / "metadata.jsonl"
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("--chunk-size must be positive")
        if overwrite and resume:
            raise ValueError("--overwrite and --resume are mutually exclusive")

        if self.metadata_path.exists() and not overwrite and not resume:
            raise FileExistsError(
                f"{self.metadata_path} already exists. Use --overwrite, --resume, or a new output dir."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for old in self.embeddings_dir.glob("embeddings_*.npz"):
                old.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()

        existing_total_rows = 0
        existing_chunk_files: list[str] = []
        existing_embedding_dim: int | None = None
        if resume and self.metadata_path.exists():
            existing_total_rows, existing_chunk_files, existing_embedding_dim = self._read_existing_state()

        self._metadata_f = self.metadata_path.open("a")
        self._buffer: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._chunk_index = self._next_chunk_index(existing_chunk_files)
        self.total_rows = existing_total_rows
        self.embedding_dim: int | None = existing_embedding_dim
        self.chunk_files: list[str] = list(existing_chunk_files)

    def _read_existing_state(self) -> tuple[int, list[str], int | None]:
        chunk_files: list[str] = []
        seen_chunks: set[str] = set()
        total_rows = 0
        embedding_dim: int | None = None

        with self.metadata_path.open("r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Cannot resume from invalid metadata JSON at {self.metadata_path}:{line_no}"
                    ) from exc
                chunk_name = row.get("chunk")
                if not chunk_name:
                    raise ValueError(
                        f"Cannot resume from metadata row without chunk at {self.metadata_path}:{line_no}"
                    )
                if chunk_name not in seen_chunks:
                    chunk_path = self.embeddings_dir / chunk_name
                    if not chunk_path.exists():
                        raise FileNotFoundError(
                            f"Cannot resume because referenced chunk is missing: {chunk_path}"
                        )
                    if embedding_dim is None:
                        with np.load(chunk_path) as data:
                            arr = data["embeddings"]
                            embedding_dim = int(arr.shape[1]) if arr.ndim == 2 else int(arr.reshape(arr.shape[0], -1).shape[1])
                    seen_chunks.add(chunk_name)
                    chunk_files.append(str(Path("embeddings") / chunk_name))
                total_rows += 1
        return total_rows, chunk_files, embedding_dim

    @staticmethod
    def _next_chunk_index(chunk_files: list[str]) -> int:
        max_idx = -1
        for chunk_file in chunk_files:
            stem = Path(chunk_file).stem
            prefix = "embeddings_"
            if stem.startswith(prefix):
                try:
                    max_idx = max(max_idx, int(stem[len(prefix):]))
                except ValueError:
                    pass
        return max_idx + 1

    def add(self, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if self.embedding_dim is None:
            self.embedding_dim = int(emb.shape[0])
        elif emb.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dim changed from {self.embedding_dim} to {emb.shape[0]}")
        self._buffer.append(emb)
        self._metadata.append(metadata)
        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        chunk_name = f"embeddings_{self._chunk_index:06d}.npz"
        chunk_path = self.embeddings_dir / chunk_name
        arr = np.stack(self._buffer, axis=0).astype(np.float32)
        np.savez_compressed(chunk_path, embeddings=arr)

        for row, metadata in enumerate(self._metadata):
            out = dict(metadata)
            out["chunk"] = chunk_name
            out["row"] = row
            out["global_index"] = self.total_rows + row
            self._metadata_f.write(json.dumps(out, sort_keys=True) + "\n")
        self._metadata_f.flush()

        self.chunk_files.append(str(Path("embeddings") / chunk_name))
        self.total_rows += len(self._buffer)
        self._chunk_index += 1
        self._buffer = []
        self._metadata = []

    def close(self) -> None:
        self.flush()
        self._metadata_f.close()

    def write_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        out = dict(manifest)
        out.update(
            {
                "embedding_dim": self.embedding_dim,
                "total_embeddings": self.total_rows,
                "num_chunks": len(self.chunk_files),
                "chunk_files": self.chunk_files,
            }
        )
        with (self.output_dir / "manifest.json").open("w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        return out
