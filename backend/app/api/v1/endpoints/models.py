"""
Models API Endpoints: Active Production Models, Versions, and Calibration.
"""

from fastapi import APIRouter
from backend.app.schemas.jobs import ModelInfoResponse
from backend.app.core.config import settings

router = APIRouter()


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get active production model information and calibration thresholds"
)
def get_model_info():
    """
    Returns active detection and verification model architectures, parameters,
    licenses, and operating decision thresholds.
    """
    return ModelInfoResponse(
        detector_model="RTMDet-Tiny (OpenMMLab)",
        detector_license="Apache License 2.0",
        detector_parameters="4.8 Million Parameters (CSPNeXt Backbone)",
        verifier_model="Siamese CNN with Contrastive Metric Projection (ResNet-18)",
        verifier_license="BSD-3-Clause / Apache 2.0",
        verifier_parameters="11.2 Million Parameters (256-dimensional Unit Hypersphere)",
        threshold_genuine=settings.THRESHOLD_GENUINE,
        threshold_forged=settings.THRESHOLD_FORGED,
        compliance_notice="100% Permissive Open Source (Strictly Non-Ultralytics, Zero AGPL)."
    )
