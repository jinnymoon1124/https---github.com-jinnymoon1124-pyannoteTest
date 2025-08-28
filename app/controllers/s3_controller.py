"""
S3 파일 업로드/관리 컨트롤러
Flask API 엔드포인트를 제공합니다.
"""

import os
import tempfile
from typing import Dict, Any, Optional
from flask import Blueprint, request

from app.services.s3_service import S3Service
from app.controllers.base_controller import BaseController

# S3 관련 API Blueprint 생성
s3_bp = Blueprint('s3', __name__, url_prefix='/api/s3')


class S3Controller(BaseController):
    """S3 파일 관리 컨트롤러 클래스"""
    
    def __init__(self):
        super().__init__()
        self.s3_service = S3Service()
        

    def upload_file(self) -> Dict[str, Any]:
        """
        파일을 S3에 업로드합니다.
        
        Form Data:
            file: 업로드할 파일
            
        Returns:
            업로드 결과 JSON 응답
        """
        try:
            # 파일 존재 확인만 진행
            if 'file' not in request.files:
                return self.error_response(
                    data="업로드할 파일이 없습니다.",
                    status_code=400
                )
            
            file = request.files['file']
            if file.filename == '':
                return self.error_response(
                    data="파일이 선택되지 않았습니다.",
                    status_code=400
                )
            
            # 파일 타입은 기본값으로 설정
            file_type = request.form.get('file_type', 'audio')
            
            # 파일명은 원본 그대로 사용 (보안 처리 제거)
            filename = file.filename
            
            # 파일 내용 읽기
            file_content = file.read()
            file_size = len(file_content)
            
            
            # S3에 업로드
            result = self.s3_service.upload_file_content(
                file_content=file_content,
                filename=filename,
                file_type=file_type,
            )
            
            if result['success']:
                return self.success_response(
                    data=result,
                    message="파일이 성공적으로 업로드되었습니다."
                )
            else:
                return self.error_response(
                    data=f"파일 업로드에 실패했습니다: {result.get('error', 'Unknown error')}",
                    status_code=500
                )
                
        except Exception as e:
            print(f"파일 업로드 처리 중 오류 발생: {str(e)}")
            return self.error_response(
                data="파일 업로드 처리 중 오류가 발생했습니다.",
                status_code=500
            )

    def download_file(self, object_key: str) -> Dict[str, Any]:
        """
        S3에서 파일을 다운로드합니다.
        
        Args:
            object_key: 다운로드할 S3 객체 키
            
        Returns:
            다운로드 결과 JSON 응답
        """
        try:
            if not object_key:
                return self.error_response(
                    data="객체 키가 제공되지 않았습니다.",
                    status_code=400
                )
            
            # 임시 파일 경로 생성
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                local_path = tmp_file.name
            
            # S3에서 다운로드
            result = self.s3_service.download_file(object_key, local_path)
            
            if result['success']:
                return self.success_response(
                    data=result,
                    message="파일이 성공적으로 다운로드되었습니다."
                )
            else:
                return self.error_response(
                    data=f"파일 다운로드에 실패했습니다: {result.get('error', 'Unknown error')}",
                    status_code=500
                )
                
        except Exception as e:
            print(f"파일 다운로드 처리 중 오류 발생: {str(e)}")
            return self.error_response(
                data="파일 다운로드 처리 중 오류가 발생했습니다.",
                status_code=500
            )

    def delete_file(self, object_key: str) -> Dict[str, Any]:
        """
        S3에서 파일을 삭제합니다.
        
        Args:
            object_key: 삭제할 S3 객체 키
            
        Returns:
            삭제 결과 JSON 응답
        """
        try:
            if not object_key:
                return self.error_response(
                    data="객체 키가 제공되지 않았습니다.",
                    status_code=400
                )
            
            # S3에서 삭제
            result = self.s3_service.delete_file(object_key)
            
            if result['success']:
                return self.success_response(
                    data=result,
                    message="파일이 성공적으로 삭제되었습니다."
                )
            else:
                return self.error_response(
                    data=f"파일 삭제에 실패했습니다: {result.get('error', 'Unknown error')}",
                    status_code=500
                )
                
        except Exception as e:
            print(f"파일 삭제 처리 중 오류 발생: {str(e)}")
            return self.error_response(
                data="파일 삭제 처리 중 오류가 발생했습니다.",
                status_code=500
            )

    def list_files(self) -> Dict[str, Any]:
        """
        S3 버킷의 파일 목록을 조회합니다.
        
        Query Parameters:
            prefix: 파일 경로 접두사 (선택사항)
            max_keys: 최대 조회 개수 (선택사항, 기본값: 100)
            
        Returns:
            파일 목록 JSON 응답
        """
        try:
            prefix = request.args.get('prefix', '')
            max_keys = int(request.args.get('max_keys', 100))
            
            # 최대 조회 개수 제한
            if max_keys > 1000:
                max_keys = 1000
            
            # S3에서 파일 목록 조회
            result = self.s3_service.list_files(prefix=prefix, max_keys=max_keys)
            
            if result['success']:
                return self.success_response(
                    data=result,
                    message=f"{result['count']}개의 파일을 조회했습니다."
                )
            else:
                return self.error_response(
                    data=f"파일 목록 조회에 실패했습니다: {result.get('error', 'Unknown error')}",
                    status_code=500
                )
                
        except ValueError:
            return self.error_response(
                data="max_keys 값이 올바르지 않습니다.",
                status_code=400
            )
        except Exception as e:
            print(f"파일 목록 조회 처리 중 오류 발생: {str(e)}")
            return self.error_response(
                data="파일 목록 조회 처리 중 오류가 발생했습니다.",
                status_code=500
            )

    def get_file_info(self, object_key: str) -> Dict[str, Any]:
        """
        S3 파일의 정보를 조회합니다.
        
        Args:
            object_key: 조회할 S3 객체 키
            
        Returns:
            파일 정보 JSON 응답
        """
        try:
            if not object_key:
                return self.error_response(
                    data="객체 키가 제공되지 않았습니다.",
                    status_code=400
                )
            
            # S3에서 파일 정보 조회
            result = self.s3_service.get_file_info(object_key)
            
            if result['success']:
                return self.success_response(
                    data=result,
                    message="파일 정보를 성공적으로 조회했습니다."
                )
            else:
                return self.error_response(
                    data=f"파일 정보 조회에 실패했습니다: {result.get('error', 'Unknown error')}",
                    status_code=404 if result.get('error_code') == 'NoSuchKey' else 500
                )
                
        except Exception as e:
            print(f"파일 정보 조회 처리 중 오류 발생: {str(e)}")
            return self.error_response(
                data="파일 정보 조회 처리 중 오류가 발생했습니다.",
                status_code=500
            )




# S3Controller 인스턴스 생성
s3_controller = S3Controller()


# API 엔드포인트 등록
@s3_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    파일을 S3에 업로드합니다.
    
    Form Data:
        file: 업로드할 파일
        file_type: 파일 타입 (audio, image, document, video)
    """
    return s3_controller.upload_file()



# @s3_bp.route('/download/<path:object_key>', methods=['GET'])
# def download_file(object_key):
#     """
#     S3에서 파일을 다운로드합니다.
    
#     Args:
#         object_key: 다운로드할 S3 객체 키
#     """
#     return s3_controller.download_file(object_key)


# @s3_bp.route('/delete/<path:object_key>', methods=['DELETE'])
# def delete_file(object_key):
#     """
#     S3에서 파일을 삭제합니다.
    
#     Args:
#         object_key: 삭제할 S3 객체 키
#     """
#     return s3_controller.delete_file(object_key)


# @s3_bp.route('/list', methods=['GET'])
# def list_files():
#     """
#     S3 버킷의 파일 목록을 조회합니다.
    
#     Query Parameters:
#         prefix: 파일 경로 접두사 (선택사항)
#         max_keys: 최대 조회 개수 (선택사항, 기본값: 100)
#     """
#     return s3_controller.list_files()


# @s3_bp.route('/info/<path:object_key>', methods=['GET'])
# def get_file_info(object_key):
#     """
#     S3 파일의 정보를 조회합니다.
    
#     Args:
#         object_key: 조회할 S3 객체 키
#     """
#     return s3_controller.get_file_info(object_key)



@s3_bp.route('/health', methods=['GET'])
def health_check():
    """S3 서비스 상태를 확인합니다."""
    try:
        s3_service = s3_controller.s3_service
        
        # 환경 변수 확인
        env_status = {
            'bucket': bool(s3_service.config.S3_BUCKET),
            'region': bool(s3_service.config.AWS_REGION)
        }
        
        # S3 연결 테스트
        s3_tests = {}
        try:
            # 1. 버킷 존재 확인
            s3_service.s3.head_bucket(Bucket=s3_service.config.S3_BUCKET)
            s3_tests['bucket_accessible'] = True
            s3_message = "S3 연결이 정상입니다."
            s3_status = "healthy"
            
            # 2. 권한 테스트 (리스트 권한)
            try:
                s3_service.s3.list_objects_v2(
                    Bucket=s3_service.config.S3_BUCKET, 
                    MaxKeys=1
                )
                s3_tests['list_permission'] = True
            except Exception as e: # Changed from ClientError to Exception to catch more potential errors
                s3_tests['list_permission'] = False
                s3_tests['list_error'] = str(e)
            
        except Exception as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            s3_tests['bucket_accessible'] = False
            s3_tests['error_code'] = error_code
            s3_tests['error_message'] = str(e)
            s3_status = "unhealthy"
            s3_message = f"S3 연결 실패: {error_code} - {str(e)}"
        
        health_info = {
            'service': 'S3 File Upload Service',
            'environment': s3_service.config.ENVIRONMENT,
            'status': s3_status,
            'message': s3_message,
            'bucket': s3_service.config.S3_BUCKET,
            'region': s3_service.config.AWS_REGION,
            'env_variables': env_status,
            's3_tests': s3_tests
        }
        
        status_code = 200 if s3_status == "healthy" else 503
        
        return s3_controller.success_response(
            data=health_info,
            message="S3 서비스 상태 확인 완료"
        ), status_code
        
    except Exception as e:
        return s3_controller.error_response(
            data=f"S3 서비스 상태 확인 실패: {str(e)}",
            status_code=503
        )
