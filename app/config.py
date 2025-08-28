"""
환경별 설정 관리 모듈
각 환경(local/dev/prod)에 따라 다른 설정값을 제공합니다.
"""

import os
from dotenv import load_dotenv


class ConstantConfig:
    """환경별 설정을 관리하는 클래스"""
    
    def __init__(self):
        load_dotenv()
        self._init_common_config()
        self._init_aws_config()

    def _init_common_config(self):
        """공통 설정 초기화"""
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'local').lower()


    def _init_aws_config(self):
        """AWS 일반 설정 초기화"""
        self.S3_BUCKET = os.getenv("AWS_S3_VOICE_BUCKET")
        self.AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
        self.DYNAMODB_SPEAKER_TABLE = os.getenv("AWS_DYNAMODB_SPEAKER_TABLE", "speakers")
        
