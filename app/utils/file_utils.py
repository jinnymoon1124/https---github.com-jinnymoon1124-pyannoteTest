"""
파일 처리 관련 유틸리티 함수들
파일 검증, 변환, 정리 등의 공통 기능을 제공
"""
import os
from werkzeug.utils import secure_filename


def get_safe_filename(original_filename: str, timestamp: str) -> str:
    """안전한 파일명 생성"""
    filename = secure_filename(original_filename)
    return f"{timestamp}_{filename}"
