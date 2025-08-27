"""
파일 처리 관련 유틸리티 함수들
파일 검증, 변환, 정리 등의 공통 기능을 제공
"""
import os
from werkzeug.utils import secure_filename


# 허용된 오디오 파일 확장자
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}


def allowed_file(filename: str) -> bool:
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_safe_filename(original_filename: str, timestamp: str) -> str:
    """안전한 파일명 생성"""
    filename = secure_filename(original_filename)
    return f"{timestamp}_{filename}"
