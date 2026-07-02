from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from navigation.route_config import NavigationRoute, RouteKeyframe
from vision import RecognitionSize, Rect, crop_roi, resize_to_recognition


@dataclass(frozen=True)
class LocalizationCandidate:
    found: bool
    keyframe_name: str | None
    index: int | None
    score: float
    inliers: int
    inlier_ratio: float
    correlation: float


@dataclass(frozen=True)
class LocalizationResult:
    found: bool
    keyframe_name: str | None
    index: int | None
    score: float
    inliers: int
    inlier_ratio: float
    correlation: float
    candidates: tuple[LocalizationCandidate, ...] = ()


@dataclass
class _PreparedKeyframe:
    config: RouteKeyframe
    image: np.ndarray
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None
    thumb: np.ndarray


class FeatureLocalizer:
    def __init__(self, route: NavigationRoute, recognition_size: RecognitionSize) -> None:
        self.route = route
        self.recognition_size = recognition_size
        self._orb = cv2.ORB_create(nfeatures=1200)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._prepared = [self._prepare(keyframe) for keyframe in route.keyframes]

    def locate(self, screenshot: np.ndarray) -> LocalizationResult:
        if not self._prepared:
            return LocalizationResult(False, None, None, 0.0, 0, 0.0, -1.0)

        image = resize_to_recognition(screenshot, self.recognition_size)
        candidates: list[LocalizationCandidate] = []
        for index, prepared in enumerate(self._prepared):
            query = self._crop_scene(image, prepared.config)
            keypoints, descriptors = self._orb.detectAndCompute(query, None)
            inliers, inlier_ratio = self._feature_score(prepared, keypoints, descriptors)
            correlation = self._correlation_score(prepared.thumb, query)
            score = float(inliers) + max(correlation, 0.0) * 20.0 + inlier_ratio * 10.0
            candidates.append(
                LocalizationCandidate(
                    found=score >= prepared.config.min_score,
                    keyframe_name=prepared.config.name,
                    index=index,
                    score=round(score, 4),
                    inliers=inliers,
                    inlier_ratio=round(inlier_ratio, 4),
                    correlation=round(correlation, 4),
                )
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        best = candidates[0]
        return LocalizationResult(
            found=best.found,
            keyframe_name=best.keyframe_name,
            index=best.index,
            score=best.score,
            inliers=best.inliers,
            inlier_ratio=best.inlier_ratio,
            correlation=best.correlation,
            candidates=tuple(candidates[:3]),
        )

    def _prepare(self, keyframe: RouteKeyframe) -> _PreparedKeyframe:
        path = self.route.resolve_path(keyframe.image)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Route keyframe image is unreadable: {path}")
        image = resize_to_recognition(image, self.recognition_size)
        scene = self._crop_scene(image, keyframe)
        keypoints, descriptors = self._orb.detectAndCompute(scene, None)
        thumb = self._thumbnail(scene)
        return _PreparedKeyframe(
            config=keyframe,
            image=scene,
            keypoints=tuple(keypoints or []),
            descriptors=descriptors,
            thumb=thumb,
        )

    def _crop_scene(self, image: np.ndarray, keyframe: RouteKeyframe) -> np.ndarray:
        roi = Rect.from_sequence(keyframe.roi or self.route.scene_roi)
        cropped, _ = crop_roi(image, roi)
        return cropped

    def _feature_score(
        self,
        prepared: _PreparedKeyframe,
        keypoints: tuple[cv2.KeyPoint, ...],
        descriptors: np.ndarray | None,
    ) -> tuple[int, float]:
        if prepared.descriptors is None or descriptors is None:
            return 0, 0.0
        if len(prepared.keypoints) < 4 or len(keypoints) < 4:
            return 0, 0.0
        matches = self._matcher.knnMatch(prepared.descriptors, descriptors, k=2)
        good = []
        for pair in matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < 0.78 * second.distance:
                good.append(first)
        if len(good) < 4:
            return len(good), 0.0
        source = np.float32([prepared.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if mask is None:
            return len(good), 0.0
        inliers = int(mask.ravel().sum())
        return inliers, inliers / max(len(good), 1)

    @staticmethod
    def _thumbnail(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

    def _correlation_score(self, prepared_thumb: np.ndarray, query: np.ndarray) -> float:
        query_thumb = self._thumbnail(query)
        if prepared_thumb.shape != query_thumb.shape:
            return -1.0
        score = cv2.matchTemplate(query_thumb, prepared_thumb, cv2.TM_CCOEFF_NORMED)
        return float(score[0, 0])
