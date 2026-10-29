"""ProtoRCD production pipeline: SAM3 detection -> DINOv3/BioCLIP-2 features -> active triage.

Three stages, importable independently:
  protorcd_pipeline.detect    - SAM3 text-prompt animal detection (stage 1)
  protorcd_pipeline.features  - DINOv3 / BioCLIP-2 proposal embeddings (stage 2)
  protorcd_pipeline.triage    - density-moderated acquisition + guarded Resolve/Clear/Defer triage (stage 3)

Stage 3 is decoupled from ground-truth boxes: labels arrive through a LabelProvider
(a UI callback, a JSON file, or an offline COCO/GT provider for reproducing paper results).
"""
__version__ = "0.1.0"
