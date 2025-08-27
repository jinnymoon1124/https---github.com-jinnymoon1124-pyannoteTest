"""
화자 관리 관련 컨트롤러
화자 조회, 이름 변경, 프로파일 관리 등의 HTTP 요청을 처리
"""
from flask import Blueprint, request, jsonify
from flask_cors import CORS
from typing import Dict, Any

from app.controllers.base_controller import BaseController
from app.services.speaker_service import SpeakerService
from app.utils.data_utils import convert_numpy_types

# Blueprint 정의
speaker_bp = Blueprint('speaker_bp', __name__, url_prefix='/speakers')
CORS(speaker_bp)


class SpeakerController(BaseController):
    """화자 관리 컨트롤러 클래스"""
    
    def __init__(self):
        """컨트롤러 초기화"""
        super().__init__()
        self.speaker_service = SpeakerService()
    
    def get_speakers(self):
        """등록된 화자 목록 조회"""
        result = self.speaker_service.get_all_speakers()
        
        if result['success']:
            # numpy 타입을 JSON 직렬화 가능한 타입으로 변환
            response_data = convert_numpy_types(result)
            return jsonify(response_data)
        else:
            return jsonify(result), 500
    
    def update_speaker_name(self, speaker_id: str):
        """화자 이름 변경"""
        # 요청 데이터 검증
        data = request.get_json()
        if not data or 'name' not in data:
            return self.invalid_param_response('name')
        
        new_name = data['name']
        
        # 서비스 레이어에서 처리
        result = self.speaker_service.update_speaker_name(speaker_id, new_name)
        
        if result['success']:
            return jsonify(result)
        else:
            status_code = 404 if '찾을 수 없습니다' in result['error'] else 400
            return jsonify(result), status_code
    
    def reset_speaker_name(self, speaker_id: str):
        """화자 이름 초기화 (원래 ID로 되돌리기)"""
        result = self.speaker_service.reset_speaker_name(speaker_id)
        
        if result['success']:
            return jsonify(result)
        else:
            status_code = 404 if '없습니다' in result['error'] else 500
            return jsonify(result), status_code
    
    def get_speaker_names(self):
        """모든 화자 이름 매핑 조회"""
        result = self.speaker_service.get_all_speaker_names()
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
    
    def delete_speaker(self, speaker_id: str):
        """화자 프로파일 삭제 (필요시 사용)"""
        result = self.speaker_service.delete_speaker_profile(speaker_id)
        
        if result['success']:
            return jsonify(result)
        else:
            status_code = 404 if '찾을 수 없습니다' in result['error'] else 500
            return jsonify(result), status_code


# 컨트롤러 인스턴스 생성
speaker_controller = SpeakerController()


# ============================================================================
# API 라우트 정의 (Blueprint에 등록)
# ============================================================================

@speaker_bp.route('', methods=['GET'])
@BaseController.handle_exceptions
def get_speakers():
    """등록된 화자 목록 조회 - 컨트롤러에 위임"""
    return speaker_controller.get_speakers()


@speaker_bp.route('/<speaker_id>/name', methods=['PUT'])
@BaseController.handle_exceptions
def update_speaker_name(speaker_id: str):
    """화자 이름 변경 - 컨트롤러에 위임"""
    return speaker_controller.update_speaker_name(speaker_id)


@speaker_bp.route('/<speaker_id>/name', methods=['DELETE'])
@BaseController.handle_exceptions
def reset_speaker_name(speaker_id: str):
    """화자 이름 초기화 - 컨트롤러에 위임"""
    return speaker_controller.reset_speaker_name(speaker_id)


@speaker_bp.route('/names', methods=['GET'])
@BaseController.handle_exceptions
def get_speaker_names():
    """모든 화자 이름 매핑 조회 - 컨트롤러에 위임"""
    return speaker_controller.get_speaker_names()


@speaker_bp.route('/<speaker_id>', methods=['DELETE'])
@BaseController.handle_exceptions
def delete_speaker(speaker_id: str):
    """화자 프로파일 삭제 - 컨트롤러에 위임"""
    return speaker_controller.delete_speaker(speaker_id)
