"""
Speaker DynamoDB 모델
음성 화자 정보를 관리하는 모델입니다.
"""

import os
import datetime
import uuid
from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, UTCDateTimeAttribute, NumberAttribute, JSONAttribute
from pynamodb.indexes import GlobalSecondaryIndex, AllProjection

KST = datetime.timezone(datetime.timedelta(hours=9))


class SpeakerNameIndex(GlobalSecondaryIndex):
    """
    화자 이름 기준 검색을 위한 GSI
    """
    class Meta:
        index_name = 'speakerName-createdAt-index'
        read_capacity_units = 5
        write_capacity_units = 5
        projection = AllProjection()
        region = 'ap-northeast-2'
    
    speakerName = UnicodeAttribute(hash_key=True)
    createdAt = UTCDateTimeAttribute(range_key=True)


class SpeakerIdIndex(GlobalSecondaryIndex):
    """
    화자 ID 기준 검색을 위한 GSI
    """
    class Meta:
        index_name = 'speakerId-createdAt-index'
        read_capacity_units = 5
        write_capacity_units = 5
        projection = AllProjection()
        region = 'ap-northeast-2'
    
    speakerId = UnicodeAttribute(hash_key=True)
    createdAt = UTCDateTimeAttribute(range_key=True)


class SpeakerModel(Model):
    """화자 정보 DynamoDB 모델"""
    
    class Meta:
        table_name = "todo-speaker"
        region = "ap-northeast-2"
    
    # 인덱스 정의
    speakerName_index = SpeakerNameIndex()
    speakerId_index = SpeakerIdIndex()
    
    @staticmethod
    def generate_speaker_id():
        """화자 ID 생성 (yyyymmdd_임의의값 형식)"""
        today = datetime.datetime.now().strftime('%Y%m%d')
        random_part = str(uuid.uuid4())[:8]
        return f"{today}_{random_part}"
    
    # 속성 정의
    id = UnicodeAttribute(hash_key=True, default=generate_speaker_id)  # PK: yyyymmdd_임의의값
    speakerId = UnicodeAttribute()  # 화자 ID (명확성을 위한 중복)
    speakerName = UnicodeAttribute()  # 매칭된 이름 (설정된 이름)
    profileS3Path = UnicodeAttribute(null=True)  # 프로필 파일 S3 경로
    embeddingsS3Path = UnicodeAttribute(null=True)  # 임베딩 파일(.pkl) S3 경로
    createdAt = UTCDateTimeAttribute(default=datetime.datetime.utcnow)  # 생성일시
    updatedAt = UTCDateTimeAttribute(null=True)  # 수정일시
    deletedAt = UTCDateTimeAttribute(null=True)  # 삭제일시 (소프트 삭제용)
    
    @classmethod
    def get_meta_table(cls):
        """테이블 메타데이터 반환"""
        return {
            'table_name': cls.Meta.table_name,
            'region': cls.Meta.region,
            'hash_key_attr': cls.id
        }
