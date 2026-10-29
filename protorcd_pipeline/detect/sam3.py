"""Adapter for launching the local SAM3 text-prompt detector."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INFERENCE_SCRIPT = Path(__file__).resolve().parent / "run_sam3_text_detection.py"


@dataclass(frozen=True)
class SAM3Command:
    coco_json: Path
    images_dir: Path
    output_base: Path
    use_sahi: bool | None = None
    inference_script: Path = DEFAULT_INFERENCE_SCRIPT
    config: Path | None = None
    sam3_source_root: Path | None = None
    text_prompts: tuple[str, ...] = ("visual",)
    confidence_threshold: float | None = None
    batch_size: int | None = None
    vis_save_every_n: int | None = None
    return_masks: bool = False

    def to_argv(self) -> list[str]:
        argv = ["python", str(self.inference_script), "--batch"]
        if self.config is not None:
            argv.extend(["--config", str(self.config)])
        if self.sam3_source_root is not None:
            argv.extend(["--sam3-source-root", str(self.sam3_source_root)])
        argv.extend(["--coco-json", str(self.coco_json)])
        argv.extend(["--images-dir", str(self.images_dir)])
        argv.extend(["--output-base", str(self.output_base)])
        for prompt in self.text_prompts:
            argv.extend(["--text-prompt", str(prompt)])
        if self.use_sahi is True:
            argv.append("--use-sahi")
        elif self.use_sahi is False:
            argv.append("--no-sahi")
        if self.confidence_threshold is not None:
            argv.extend(["--confidence-threshold", str(self.confidence_threshold)])
        if self.batch_size is not None:
            argv.extend(["--batch-size", str(self.batch_size)])
        if self.vis_save_every_n is not None:
            argv.extend(["--vis-save-every-n", str(self.vis_save_every_n)])
        if self.return_masks:
            argv.append("--return-masks")
        return argv


def run_sam3(command: SAM3Command, dry_run: bool = False) -> int:
    argv = command.to_argv()
    print(" ".join(argv))
    if dry_run:
        return 0
    return int(subprocess.run(argv, check=False).returncode)
