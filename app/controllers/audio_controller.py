"""
오디오 처리 관련 컨트롤러
오디오 업로드, 화자 분리, STT 처리 등의 HTTP 요청을 처리
"""
import os
import time
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_cors import CORS

from app.controllers.base_controller import BaseController
from app.services.audio_service import AudioProcessingService
from app.services.s3_service import S3Service
from app.services.speaker_service import SpeakerService
from app.utils.file_utils import get_safe_filename
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
        self.s3_service = S3Service()
        self.speaker_service = SpeakerService()
        self.upload_folder = 'temp/uploads'
    
    def process_audio(self):
        """오디오 파일을 받아서 화자 분리 및 STT 처리"""

        # 모델 초기화 (처음 호출 시에만)
        self.audio_service.initialize_models()
        
        file = request.files['audio_file']
        
        # 파일 저장 및 변환
        original_file_path, converted_file_path = self._save_and_convert_file(file)
        
        # 원본 파일을 S3에 업로드
        self._upload_original_file_to_s3(original_file_path, file.filename)
        
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
            
            # 검증된 화자들을 DynamoDB에 저장
            self._save_verified_speakers_to_dynamodb(verified_speakers)
            
            # 검증 결과로 화자명 업데이트 (DynamoDB에서 이름 조회)
            self._update_results_with_verification(results, verified_speakers)
            
            # 화자별 발화 요약 생성
            speaker_summary = self.audio_service.generate_speaker_summary(results)
            
            # 응답 데이터 구성
            total_time = diarization_time + stt_time + verify_time
            response_data = self._build_response_data(
                results, speaker_summary, verified_speakers,
                total_time, diarization_time, stt_time, verify_time
            )
            
            # 대화록 파일 저장
            print("대화록 파일 저장 중...")
            transcript_path = self.audio_service.save_transcript_to_file(
                results, speaker_summary, verified_speakers, response_data["processing_info"]
            )
            
            # 응답 데이터에 대화록 파일 경로 추가
            if transcript_path:
                response_data["transcript_file"] = {
                    "path": transcript_path,
                    "filename": os.path.basename(transcript_path),
                    "message": "대화록이 파일로 저장되었습니다."
                }
            
            return jsonify(response_data)
            
        finally:
            # 파일 정리
            self.audio_service.cleanup_files(original_file_path, converted_file_path)
    
    
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
    
    def _upload_original_file_to_s3(self, file_path, original_filename):
        """원본 오디오 파일을 S3에 업로드"""
        try:
            now = datetime.now()
            
            # 파일 확장자 추출
            file_ext = os.path.splitext(original_filename)[1].lower()
            filename_without_ext = os.path.splitext(original_filename)[0]
            
            # 임의값 생성 (8자리)
            import uuid
            random_value = str(uuid.uuid4())[:8]
            
            # 파일명 생성: yyyymmdd_원본파일명_임의값.확장자
            new_filename = f"{now.strftime('%Y%m%d')}_{filename_without_ext}_{random_value}{file_ext}"
            
            # S3 키 생성: audio/yyyy/mm/파일명
            s3_key = f"audio/{now.strftime('%Y')}/{now.strftime('%m')}/{new_filename}"
            
            print(f"📤 원본 오디오 파일 S3 업로드: {s3_key}")
            
            # S3에 업로드
            result = self.s3_service.upload_file(
                file_path=file_path,
                object_key=s3_key,
                file_type="audio"
            )
            
            if result['success']:
                print(f"✅ 원본 파일 S3 업로드 완료: {s3_key}")
                return s3_key
            else:
                print(f"❌ 원본 파일 S3 업로드 실패: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ 원본 파일 S3 업로드 중 오류: {e}")
            return None
    
    def _save_verified_speakers_to_dynamodb(self, verified_speakers):
        """검증된 화자들을 DynamoDB에 저장"""
        try:
            for speaker_id, speaker_info in verified_speakers.items():
                # 새로운 화자이거나 기존 화자로 매칭된 경우 DynamoDB에 저장
                if speaker_info.get('new_speaker_id') or speaker_info.get('matched_speaker_id'):
                    # 화자 ID 결정
                    final_speaker_id = speaker_info.get('matched_speaker_id') or speaker_info.get('new_speaker_id') or speaker_id
                    
                    # DynamoDB에 저장 (기본 이름은 speaker_id) - SpeakerService를 통해 저장
                    result = self.speaker_service.create_or_update_speaker_name(final_speaker_id, final_speaker_id)
                    
                    if result['success']:
                        print(f"✅ 화자 {final_speaker_id} DynamoDB 저장 완료: {result['action']}")
                    else:
                        print(f"❌ 화자 {final_speaker_id} DynamoDB 저장 실패: {result.get('error')}")
                        
        except Exception as e:
            print(f"❌ 검증된 화자 DynamoDB 저장 중 오류: {e}")
    
    def _update_results_with_verification(self, results, verified_speakers):
        """검증 결과로 화자명 업데이트 (DynamoDB에서 실제 이름 조회)"""
        for result in results:
            original_speaker = result["speaker"]
            if original_speaker in verified_speakers:
                verified_info = verified_speakers[original_speaker]
                
                # 화자 ID 정보 정리
                if verified_info.get('new_speaker_id'):
                    # 새로운 화자인 경우
                    final_speaker_id = verified_info['new_speaker_id']
                elif verified_info.get('matched_speaker_id'):
                    # 기존 화자로 매칭된 경우
                    final_speaker_id = verified_info['matched_speaker_id']
                else:
                    # fallback: 원본 화자 라벨 사용
                    final_speaker_id = original_speaker
                
                # DynamoDB에서 화자의 실제 이름 조회
                display_name = self.speaker_service.get_display_name(final_speaker_id)
                
                # 검증 정보와 함께 화자 이름 설정
                result["verified_speaker"] = display_name  # DynamoDB에서 조회한 실제 이름
                result["speaker_id"] = final_speaker_id
                result["verification_confidence"] = verified_info['confidence']
                result["similarity_score"] = float(verified_info['similarity'])
                result["is_known_speaker"] = verified_info['is_known']
                
                # 원본 화자 라벨은 별도 필드로 보관 (디버깅용)
                result["original_speaker_label"] = original_speaker
                
                # 기존 speaker 필드 제거 (혼동 방지)
                del result["speaker"]
            else:
                # 검증되지 않은 화자의 경우 기본 처리
                result["verified_speaker"] = original_speaker
                result["speaker_id"] = original_speaker
                result["verification_confidence"] = 0.0
                result["similarity_score"] = 0.0
                result["is_known_speaker"] = False
                result["original_speaker_label"] = original_speaker
                del result["speaker"]
    
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
