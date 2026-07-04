"""Face detection via InsightFace (RetinaFace-based detector, buffalo_l model pack) and
primary-face selection for images that contain more than one face (e.g. bystanders)."""

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    """Wraps insightface's FaceAnalysis: detection + 5-point landmarks + ArcFace
    embedding, all in one ONNX Runtime CPU pipeline (no CUDA required)."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        ctx_id: int = -1,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ):
        self.app = FaceAnalysis(name=model_name, providers=list(providers))
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)

    def detect(self, image_bgr: np.ndarray) -> list:
        """Returns a list of insightface Face objects (.bbox, .kps, .det_score,
        .normed_embedding), one per detected face in the image."""
        return self.app.get(image_bgr)


def _bbox_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _center_weighted_score(face, image_shape: tuple[int, int]) -> float:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = face.bbox
    face_cx, face_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    img_cx, img_cy = w / 2.0, h / 2.0
    dist_to_center = np.hypot(face_cx - img_cx, face_cy - img_cy)
    scale = 0.5 * min(h, w)
    return _bbox_area(face) * float(np.exp(-dist_to_center / max(scale, 1e-6)))


def select_primary_face(
    faces: list,
    policy: str = "largest_area",
    image_shape: tuple[int, int] | None = None,
):
    """Pick the single face that represents "the subject" of a photo containing
    multiple detected faces (e.g. bystanders in the background).

    Policies:
      - largest_area (default): the subject typically fills more of the frame than
        smaller/partially-occluded bystanders. Ties broken by higher det_score.
      - center_weighted: rewards faces that are both large AND centered, guarding
        against a large-but-off-frame bystander outscoring a smaller centered subject.
        Requires image_shape.
      - det_score: pure detector-confidence argmax, useful when bbox size is a poor
        proxy for prominence (e.g. subject partially cropped by the frame edge).

    Known limitation: none of these heuristics can distinguish "the subject" from a
    bystander who happens to be closer to the camera / larger / more centered than the
    actual subject. A production system would need subject-tracking across a burst of
    frames or a manual click-to-select UI; out of scope for single stills here.
    """
    if not faces:
        return None

    if policy == "largest_area":
        return max(faces, key=lambda f: (_bbox_area(f), f.det_score))
    elif policy == "det_score":
        return max(faces, key=lambda f: f.det_score)
    elif policy == "center_weighted":
        if image_shape is None:
            raise ValueError("center_weighted policy requires image_shape")
        return max(faces, key=lambda f: _center_weighted_score(f, image_shape))
    else:
        raise ValueError(f"Unknown primary-face policy: {policy}")
