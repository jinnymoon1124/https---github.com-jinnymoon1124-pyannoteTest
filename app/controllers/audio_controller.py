"""
오디오 처리 관련 컨트롤러
오디오 업로드, 화자 분리, STT 처리 등의 HTTP 요청을 처리
"""
import os
import time
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_cors import CORS
from typing import Dict, Any

from app.controllers.base_controller import BaseController
from app.services.audio_service import AudioProcessingService
from app.utils.file_utils import allowed_file, get_safe_filename
from app.utils.data_utils import convert_numpy_types

# Blueprint 정의
audio_bp = Blueprint('audio_bp', __name__, url_prefix='/audio')
CORS(audio_bp)


class AudioController(BaseController):
    """오디오 처리 컨트롤러 클래스"""
    
    def __init__(self):
        """컨트롤러 초기화"""
        super().__init__()
        self.audio_service = AudioProcessingService()
        self.upload_folder = 'temp/uploads'
    
    def process_audio(self):
        """오디오 파일을 받아서 화자 분리 및 STT 처리"""
        # 파일 검증
        validation_result = self._validate_uploaded_file()
        if validation_result:
            return validation_result
        # 모델 초기화 (처음 호출 시에만)
        self.audio_service.initialize_models()
        
        file = request.files['audio_file']
        
        # 파일 저장 및 변환
        original_file_path, converted_file_path = self._save_and_convert_file(file)
        
        try:
            # 화자 분리 처리
            print("화자 분리 처리 시작...")
            diarization, diarization_time = self.audio_service.perform_speaker_diarization(converted_file_path)
            
            # STT 처리
            print("STT 처리 시작...")
            results, stt_time = self.audio_service.perform_speech_to_text(converted_file_path, diarization)
            
            # 화자 검증
            print("화자 검증 시작...")
            verify_start_time = time.time()
            verified_speakers = self.audio_service.verify_speakers_against_profiles(converted_file_path, results)
            verify_time = time.time() - verify_start_time
            
            # 검증 결과로 화자명 업데이트
            self._update_results_with_verification(results, verified_speakers)
            
            # 화자별 발화 요약 생성
            speaker_summary = self.audio_service.generate_speaker_summary(results)
            
            # 응답 데이터 구성
            total_time = diarization_time + stt_time + verify_time
            response_data = self._build_response_data(
                results, speaker_summary, verified_speakers,
                total_time, diarization_time, stt_time, verify_time
            )
            
            return jsonify(response_data)
            
        finally:
            # 파일 정리
            self.audio_service.cleanup_files(original_file_path, converted_file_path)
    
    def _validate_uploaded_file(self):
        """업로드된 파일 검증"""
        print("업로드된 파일 검증 시작...")
        print(f"Request files: {list(request.files.keys())}")  # 업로드된 파일 키 확인
        
        if 'audio_file' not in request.files:
            print("오류: audio_file 키가 request.files에 없음")
            return jsonify({'error': '오디오 파일이 없습니다.'}), 400
        
        file = request.files['audio_file']
        print(f"파일명: {file.filename}")
        print(f"파일 크기: {file.content_length if hasattr(file, 'content_length') else 'Unknown'}")
        
        if file.filename == '':
            print("오류: 빈 파일명")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        if not allowed_file(file.filename):
            from app.utils.file_utils import ALLOWED_EXTENSIONS
            print(f"오류: 지원되지 않는 파일 형식 - {file.filename}")
            return jsonify({
                'error': f'지원되지 않는 파일 형식입니다. 지원 형식: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        print("파일 검증 통과!")
        return None  # 검증 통과
    
    def _save_and_convert_file(self, file):
        """파일 저장 및 WAV 변환"""
        # 안전한 파일명으로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = get_safe_filename(file.filename, timestamp)
        original_file_path = os.path.join(self.upload_folder, safe_filename)
        
        # 업로드 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(original_file_path), exist_ok=True)
        
        file.save(original_file_path)
        print(f"원본 파일 저장됨: {original_file_path}")
        
        # 오디오 파일을 WAV 형식으로 변환
        converted_file_path = self.audio_service.convert_audio_to_wav(original_file_path, timestamp)
        
        return original_file_path, converted_file_path
    
    def _update_results_with_verification(self, results, verified_speakers):
        """검증 결과로 화자명 업데이트"""
        for result in results:
            original_speaker = result["speaker"]
            if original_speaker in verified_speakers:
                verified_info = verified_speakers[original_speaker]
                result["verified_speaker"] = verified_info['identified_as']
                result["verification_confidence"] = verified_info['confidence']
                result["similarity_score"] = float(verified_info['similarity'])
                result["is_known_speaker"] = verified_info['is_known']
    
    def _build_response_data(self, results, speaker_summary, verified_speakers,
                           total_time, diarization_time, stt_time, verify_time):
        """응답 데이터 구성"""
        response_data = {
            "success": True,
            "message": "처리 완료",
            "processing_info": {
                "total_time": round(total_time, 2),
                "diarization_time": round(diarization_time, 2),
                "stt_time": round(stt_time, 2),
                "verification_time": round(verify_time, 2),
                "total_segments": len(results),
                "unique_speakers": len(speaker_summary)
            },
            "segments": results,
            "speaker_summary": speaker_summary,
            "verified_speakers": verified_speakers
        }
        
        # numpy 타입을 JSON 직렬화 가능한 타입으로 변환
        return convert_numpy_types(response_data)


# 컨트롤러 인스턴스 생성
audio_controller = AudioController()


# ============================================================================
# API 라우트 정의 (Blueprint에 등록)
# ============================================================================

@audio_bp.route('/process', methods=['POST'])
@BaseController.handle_exceptions
def process_audio():
    """오디오 파일을 받아서 화자 분리 및 STT 처리 - 컨트롤러에 위임"""
    return audio_controller.process_audio()
