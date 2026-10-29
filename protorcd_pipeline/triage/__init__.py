"""Stage 3: density-moderated acquisition + guarded Resolve/Clear/Defer triage.

Decoupled from ground-truth boxes. Public API::

    from protorcd_pipeline.triage import (
        ProposalPool, load_pool, labels_from_coco,
        CallbackLabelProvider, FileLabelProvider, GTLabelProvider,
        TriageRule, calibrate_rule, guarded_triage,
        TriageSession,
    )
"""
from .acquisition import ACQUISITION_METHODS, acquire, compute_density, l2_normalize
from .calibration import (
    CALIBRATION_MODES,
    DEFAULT_K_VALUES,
    calibrate_rule,
    loo_score,
)
from .labels import (
    CallbackLabelProvider,
    FileLabelProvider,
    GTLabelProvider,
    LabelProvider,
)
from .pool import (
    ProposalPool,
    labels_from_coco,
    labels_from_coco_multi,
    load_feature_store,
    load_pool,
    load_pool_multi,
)
from .session import PROTOCOL_PRESETS, RoundInfo, TriageResult, TriageSession
from .triage import TriageRule, class_support_thresholds, guarded_triage

__all__ = [
    # acquisition
    "acquire", "compute_density", "l2_normalize", "ACQUISITION_METHODS",
    # calibration
    "calibrate_rule", "loo_score", "DEFAULT_K_VALUES", "CALIBRATION_MODES",
    # triage rule
    "TriageRule", "guarded_triage", "class_support_thresholds",
    # labels
    "LabelProvider", "CallbackLabelProvider", "FileLabelProvider", "GTLabelProvider",
    # pool
    "ProposalPool", "load_pool", "load_pool_multi", "load_feature_store",
    "labels_from_coco", "labels_from_coco_multi",
    # session
    "TriageSession", "PROTOCOL_PRESETS", "RoundInfo", "TriageResult",
]
