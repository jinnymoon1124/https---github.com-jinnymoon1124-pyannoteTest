"""
데이터 처리 관련 유틸리티 함수들
JSON 직렬화, 타입 변환 등의 공통 기능을 제공
"""
import numpy as np
from typing import Any


def convert_numpy_types(obj: Any) -> Any:
    """numpy 타입을 JSON 직렬화 가능한 Python 기본 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
