"""
Model Registry & Deployment Gatekeeper.
Evaluates candidate models against active champions.
Strict condition: Candidate must exceed current champion accuracy/mAP before promotion.
"""

from typing import Dict, Any, Tuple


class ModelRegistryGate:
    """
    Quality gate enforcing automated promotion criteria before new model deployment.
    """
    def __init__(self, min_map50: float = 0.85, max_eer: float = 0.08):
        self.min_map50 = min_map50
        self.max_eer = max_eer

    def evaluate_detection_candidate(
        self,
        candidate_metrics: Dict[str, float],
        champion_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Evaluates detection candidate (mAP@50).
        """
        cand_map = candidate_metrics.get("mAP@50", 0.0)
        champ_map = champion_metrics.get("mAP@50", 0.0)

        if cand_map < self.min_map50:
            return False, f"Candidate mAP@50 ({cand_map:.4f}) failed minimum threshold ({self.min_map50:.4f})."

        if cand_map <= champ_map:
            return False, f"Candidate mAP@50 ({cand_map:.4f}) did not exceed active champion ({champ_map:.4f}). Promotion rejected."

        return True, f"Candidate mAP@50 ({cand_map:.4f}) successfully exceeded champion ({champ_map:.4f}). Approved for promotion."

    def evaluate_verification_candidate(
        self,
        candidate_metrics: Dict[str, float],
        champion_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Evaluates verification candidate (Equal Error Rate — lower is better).
        """
        cand_eer = candidate_metrics.get("eer", 1.0)
        champ_eer = champion_metrics.get("eer", 1.0)

        if cand_eer > self.max_eer:
            return False, f"Candidate EER ({cand_eer:.4f}) exceeded maximum allowable error ({self.max_eer:.4f})."

        if cand_eer >= champ_eer:
            return False, f"Candidate EER ({cand_eer:.4f}) is not superior to active champion ({champ_eer:.4f}). Promotion rejected."

        return True, f"Candidate EER ({cand_eer:.4f}) successfully reduced error below champion ({champ_eer:.4f}). Approved for promotion."
