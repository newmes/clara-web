"""Sampler — 확률 분포에서 값을 추출하는 순수 Python 모듈

LLM이 확률을 정하면, 이 모듈이 주사위를 굴린다.
LLM 호출 없음. 재현 가능(seed).
"""

import math
import random
from typing import Any


class Sampler:
    """시드 기반 난수 생성기. 동일 시드 → 동일 결과."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.seed = seed

    # ── 범주형 ─────────────────────────────────────────

    def categorical(self, options: dict[str, float]) -> str:
        """카테고리별 확률에서 하나를 뽑는다.

        Args:
            options: {"M": 0.75, "F": 0.25} 형태

        Returns:
            선택된 카테고리 이름
        """
        if not options:
            raise ValueError("options가 비어 있음")
        names = list(options.keys())
        weights = [max(0, w) for w in options.values()]
        total = sum(weights)
        if total <= 0:
            # 모든 가중치가 0이면 균등 분포
            return self.rng.choice(names)
        return self.rng.choices(names, weights=weights, k=1)[0]

    def boolean(self, probability: float) -> bool:
        """주어진 확률로 True/False를 뽑는다."""
        return self.rng.random() < probability

    def multi_boolean(self, options: dict[str, float]) -> dict[str, bool]:
        """각 항목에 대해 독립적으로 True/False를 뽑는다.

        Args:
            options: {"hypertension": 0.6, "diabetes": 0.3, ...}

        Returns:
            {"hypertension": True, "diabetes": False, ...}
        """
        return {name: self.boolean(prob) for name, prob in options.items()}

    # ── 수치형 ─────────────────────────────────────────

    def numeric(
        self,
        distribution: str,
        params: dict[str, float],
    ) -> float:
        """명명된 분포에서 수치를 뽑는다.

        Args:
            distribution: "normal", "uniform", "lognormal", "triangular"
            params: 분포 파라미터. min/max는 clamp용.

        Returns:
            샘플된 값
        """
        if distribution == "normal":
            val = self.rng.gauss(params["mean"], params.get("std", 1.0))
        elif distribution == "uniform":
            val = self.rng.uniform(params.get("min", 0), params.get("max", 1))
        elif distribution == "lognormal":
            if "mu" in params:
                mu = params["mu"]
                sigma = params.get("sigma", 1)
            elif "mean" in params:
                import math
                mean = max(params["mean"], 1)
                std = max(params.get("std", mean * 0.5), 0.1)
                sigma_sq = math.log(1 + (std / mean) ** 2)
                mu = math.log(mean) - sigma_sq / 2
                sigma = math.sqrt(sigma_sq)
            else:
                mu, sigma = 0, 1
            val = self.rng.lognormvariate(mu, sigma)
        elif distribution == "triangular":
            val = self.rng.triangular(
                params.get("min", 0), params.get("max", 1), params.get("mode", 0.5)
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Clamp
        if "min" in params:
            val = max(val, params["min"])
        if "max" in params:
            val = min(val, params["max"])

        return val

    def integer(self, distribution: str, params: dict[str, float]) -> int:
        """수치 분포에서 정수를 뽑는다."""
        return round(self.numeric(distribution, params))

    # ── 복합형 ─────────────────────────────────────────

    def sample_from_spec(self, spec: dict) -> Any:
        """LLM이 출력한 확률 스펙에서 자동으로 샘플링.

        spec 형식:
            {"type": "categorical", "options": {"M": 0.75, "F": 0.25}}
            {"type": "boolean", "probability": 0.3}
            {"type": "numeric", "distribution": "normal", "params": {"mean": 68, "std": 9}}
            {"type": "integer", "distribution": "normal", "params": {"mean": 68, "std": 9}}
            {"type": "fixed", "value": "something"}

        Returns:
            샘플된 값
        """
        spec_type = spec.get("type", "fixed")

        if spec_type == "categorical":
            return self.categorical(spec["options"])
        elif spec_type == "boolean":
            return self.boolean(spec["probability"])
        elif spec_type == "numeric":
            return self.numeric(spec["distribution"], spec.get("params", {}))
        elif spec_type == "integer":
            return self.integer(spec["distribution"], spec.get("params", {}))
        elif spec_type == "fixed":
            return spec["value"]
        elif spec_type == "multi_boolean":
            return self.multi_boolean(spec["options"])
        else:
            raise ValueError(f"Unknown spec type: {spec_type}")

    def sample_decision_nodes(self, nodes: list[dict]) -> dict[str, Any]:
        """여러 결정 노드를 순서대로 샘플링.

        Args:
            nodes: [{"id": "sex", "spec": {...}}, {"id": "age", "spec": {...}}, ...]

        Returns:
            {"sex": "M", "age": 72, ...}
        """
        results = {}
        for node in nodes:
            node_id = node["id"]
            spec = node["spec"]
            results[node_id] = self.sample_from_spec(spec)
        return results
