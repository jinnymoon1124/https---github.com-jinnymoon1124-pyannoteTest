"""
AWS S3 파일 업로드 서비스
다양한 파일 타입의 업로드, 다운로드, 삭제 기능을 제공합니다.
"""

import os
import mimetypes
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import boto3

from botocore.exceptions import ClientError

from app.config import ConstantConfig


class S3Service:
    """AWS S3 파일 관리 서비스 클래스"""
    
    def __init__(self):
        self.config = ConstantConfig()
        # S3 클라이언트 초기화
        self.s3 = boto3.client(
            's3',
            region_name=self.config.AWS_REGION
        )
    
    def upload_file(self, 
                   file_path: str, 
                   object_key: Optional[str] = None,
                   file_type: str = "audio",
                   ) -> Dict[str, Any]:
        """
        파일을 S3에 업로드합니다.
        
        Args:
            file_path: 업로드할 파일의 로컬 경로
            object_key: S3 객체 키 (None이면 자동 생성)
            file_type: 파일 타입 (audio, image, document 등)
            
        Returns:
            업로드 결과 정보 딕셔너리
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            # 객체 키 생성
            if object_key is None:
                object_key = self._generate_object_key(file_path, file_type)
            
            # 파일 크기 확인
            file_size = os.path.getsize(file_path)
            
            # MIME 타입 결정
            content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            
            # S3에 업로드 (메타데이터 제거)
            print(f"파일 업로드 시작: {file_path} -> s3://{self.config.S3_BUCKET}/{object_key}")
            
            self.s3.upload_file(
                file_path,
                self.config.S3_BUCKET,
                object_key,
                ExtraArgs={
                    'ContentType': content_type
                }
            )
            
            # 업로드된 파일의 URL 생성
            file_url = f"https://{self.config.S3_BUCKET}.s3.{self.config.AWS_REGION}.amazonaws.com/{object_key}"
            
            result = {
                'success': True,
                'object_key': object_key,
                'file_url': file_url,
                'file_size': file_size,
                'content_type': content_type,
                'bucket': self.config.S3_BUCKET,
                'environment': self.config.ENVIRONMENT,
                'upload_time': datetime.now().isoformat()
            }
            
            print(f"파일 업로드 완료: {object_key}")
            return result
            
        except ClientError as e:
            error_msg = f"S3 업로드 오류: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_code': e.response.get('Error', {}).get('Code', 'Unknown')
            }
        except Exception as e:
            error_msg = f"파일 업로드 실패: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def upload_file_content(self,
                           file_content: bytes,
                           filename: str,
                           file_type: str = "audio",
                           ) -> Dict[str, Any]:
        """
        파일 내용을 직접 S3에 업로드합니다.
        
        Args:
            file_content: 업로드할 파일의 바이트 내용
            filename: 파일명
            file_type: 파일 타입
            
        Returns:
            업로드 결과 정보 딕셔너리
        """
        try:
            # 객체 키 생성
            object_key = self._generate_object_key(filename, file_type)
            
            # MIME 타입 결정
            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            
            # S3에 업로드 (메타데이터 제거)
            print(f"파일 내용 업로드 시작: {filename} -> s3://{self.config.S3_BUCKET}/{object_key}")
            
            self.s3.put_object(
                Bucket=self.config.S3_BUCKET,
                Key=object_key,
                Body=file_content,
                ContentType=content_type
            )
            
            # 업로드된 파일의 URL 생성
            file_url = f"https://{self.config.S3_BUCKET}.s3.{self.config.AWS_REGION}.amazonaws.com/{object_key}"
            
            result = {
                'success': True,
                'object_key': object_key,
                'file_url': file_url,
                'file_size': len(file_content),
                'content_type': content_type,
                'bucket': self.config.S3_BUCKET,
                'environment': self.config.ENVIRONMENT,
                'upload_time': datetime.now().isoformat()
            }
            
            print(f"파일 내용 업로드 완료: {object_key}")
            return result
            
        except ClientError as e:
            error_msg = f"S3 업로드 오류: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_code': e.response.get('Error', {}).get('Code', 'Unknown')
            }
        except Exception as e:
            error_msg = f"파일 내용 업로드 실패: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def download_file(self, object_key: str, local_path: str) -> Dict[str, Any]:
        """
        S3에서 파일을 다운로드합니다.
        
        Args:
            object_key: S3 객체 키
            local_path: 다운로드할 로컬 경로
            
        Returns:
            다운로드 결과 정보 딕셔너리
        """
        try:
            print(f"파일 다운로드 시작: s3://{self.config.S3_BUCKET}/{object_key} -> {local_path}")
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3.download_file(self.config.S3_BUCKET, object_key, local_path)
            
            file_size = os.path.getsize(local_path)
            
            result = {
                'success': True,
                'object_key': object_key,
                'local_path': local_path,
                'file_size': file_size,
                'download_time': datetime.now().isoformat()
            }
            
            print(f"파일 다운로드 완료: {local_path}")
            return result
            
        except ClientError as e:
            error_msg = f"S3 다운로드 오류: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_code': e.response.get('Error', {}).get('Code', 'Unknown')
            }
        except Exception as e:
            error_msg = f"파일 다운로드 실패: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def delete_file(self, object_key: str) -> Dict[str, Any]:
        """
        S3에서 파일을 삭제합니다.
        
        Args:
            object_key: 삭제할 S3 객체 키
            
        Returns:
            삭제 결과 정보 딕셔너리
        """
        try:
            print(f"파일 삭제 시작: s3://{self.config.S3_BUCKET}/{object_key}")
            
            self.s3.delete_object(Bucket=self.config.S3_BUCKET, Key=object_key)
            
            result = {
                'success': True,
                'object_key': object_key,
                'delete_time': datetime.now().isoformat()
            }
            
            print(f"파일 삭제 완료: {object_key}")
            return result
            
        except ClientError as e:
            error_msg = f"S3 삭제 오류: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_code': e.response.get('Error', {}).get('Code', 'Unknown')
            }
        except Exception as e:
            error_msg = f"파일 삭제 실패: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def list_files(self, prefix: str = "") -> Dict[str, Any]:
        """
        S3 버킷의 파일 목록을 조회합니다.
        
        Args:
            prefix: 파일 경로 접두사
            
        Returns:
            파일 목록 정보 딕셔너리
        """
        try:
            # S3 객체 목록 조회 파라미터 설정
            params = {
                'Bucket': self.config.S3_BUCKET
            }
            
            # prefix가 있으면 추가
            if prefix:
                params['Prefix'] = prefix
            
            response = self.s3.list_objects_v2(**params)
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag'].strip('"')
                    })
            
            result = {
                'success': True,
                'files': files,
                'count': len(files),
                'is_truncated': response.get('IsTruncated', False)
            }
            
            print(f"파일 목록 조회 완료: {len(files)}개 파일")
            return result
            
        except ClientError as e:
            error_msg = f"S3 목록 조회 오류: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_code': e.response.get('Error', {}).get('Code', 'Unknown')
            }
        except Exception as e:
            error_msg = f"파일 목록 조회 실패: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def _generate_object_key(self, filename: str, file_type: str) -> str:
        """
        S3 객체 키를 생성합니다.
        
        Args:
            filename: 파일명
            file_type: 파일 타입
            
        Returns:
            생성된 객체 키
        """
        # 파일 확장자 추출
        file_ext = os.path.splitext(filename)[1].lower()
        
        # 고유 ID 생성
        unique_id = str(uuid.uuid4())[:8]
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 객체 키 생성 (환경별 접두사 포함)
        object_key = f"{file_type}/{timestamp}_{unique_id}{file_ext}"
        
        return object_key

    def get_file_info(self, object_key: str) -> Dict[str, Any]:
        """
        S3 파일의 정보를 조회합니다.
        
        Args:
            object_key: S3 객체 키
            
        Returns:
            파일 정보 딕셔너리
        """
        try:
            response = self.s3.head_object(Bucket=self.config.S3_BUCKET, Key=object_key)
            
            result = {
                'success': True,
                'object_key': object_key,
                'size': response['ContentLength'],
                'content_type': response.get('ContentType', 'unknown'),
                'last_modified': response['LastModified'].isoformat(),
                'etag': response['ETag'].strip('"'),
            }
            
            return result
            
        except ClientError as e:
            error_msg = f"파일 정보 조회 오류: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_code': e.response.get('Error', {}).get('Code', 'Unknown')
            }
        except Exception as e:
            error_msg = f"파일 정보 조회 실패: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
