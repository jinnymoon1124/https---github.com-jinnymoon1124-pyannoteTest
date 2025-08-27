"""
오디오 처리 관련 서비스 클래스
화자 분리, STT, 화자 검증 등의 핵심 비즈니스 로직을 담당
"""
import os
import tempfile
import time
import torch
import whisper
import librosa
import soundfile as sf
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from pyannote.audio import Pipeline
import glob


class AudioProcessingService:
    """오디오 처리 서비스 클래스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.pipeline = None
        self.whisper_model = None
        self.device = None
        
    def initialize_models(self):
        """AI 모델들을 초기화하는 함수"""
        if self.pipeline is None:
            print("모델 로딩 중...")
            
            # GPU 설정
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("CPU 모드로 실행")
            
            # 화자 분리 파이프라인 로드
            self.pipeline = Pipeline.from_pretrained("models/pyannote_diarization_config.yaml")
            self.pipeline.to(self.device)
            print("화자 분리 파이프라인 로드 완료")
            
            # Whisper 모델 로드
            self.whisper_model = whisper.load_model("large-v3")
            self.whisper_model = self.whisper_model.to(self.device)
            print("Whisper STT 모델 로드 완료")
    
    def convert_audio_to_wav(self, original_file_path, timestamp):
        """오디오 파일을 WAV 형식으로 변환"""
        print("오디오 파일을 WAV로 변환 중...")
        try:
            # librosa로 오디오 로드 및 정규화 (16kHz, mono)
            audio_data, original_sr = librosa.load(original_file_path, sr=16000, mono=True)
            
            # WAV 파일로 저장
            wav_filename = f"{timestamp}_converted.wav"
            wav_file_path = os.path.join('temp/uploads', wav_filename)
            
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(wav_file_path), exist_ok=True)
            
            sf.write(wav_file_path, audio_data, 16000)
            
            print(f"WAV 변환 완료: {wav_file_path}")
            return wav_file_path
            
        except Exception as e:
            print(f"오디오 변환 중 오류: {e}")
            # 변환 실패 시 원본 파일 사용
            return original_file_path
    
    def perform_speaker_diarization(self, file_path):
        """화자 분리 처리를 수행"""
        print("화자 분리 처리 시작...")
        start_time = time.time()
        
        try:
            # 파이프라인 배치 크기를 1로 설정하여 텐서 크기 불일치 문제 방지
            original_batch_size = getattr(self.pipeline._segmentation, 'batch_size', None)
            if hasattr(self.pipeline._segmentation, 'batch_size'):
                self.pipeline._segmentation.batch_size = 1
            
            if hasattr(self.pipeline._embedding, 'batch_size'):
                self.pipeline._embedding.batch_size = 1
                
            diarization = self.pipeline(file_path)
            
            # 원래 배치 크기로 복원
            if original_batch_size is not None and hasattr(self.pipeline._segmentation, 'batch_size'):
                self.pipeline._segmentation.batch_size = original_batch_size
                
        except Exception as e:
            print(f"화자 분리 처리 중 오류: {e}")
            # 오류 발생 시 더 안전한 방식으로 재시도
            try:
                print("안전 모드로 재시도 중...")
                # 더 작은 청크 단위로 처리
                if hasattr(self.pipeline, '_segmentation') and hasattr(self.pipeline._segmentation, 'step'):
                    original_step = self.pipeline._segmentation.step
                    self.pipeline._segmentation.step = 0.25  # 0.25초 단위로 처리
                    
                diarization = self.pipeline(file_path)
                
                # 원래 설정 복원
                if 'original_step' in locals():
                    self.pipeline._segmentation.step = original_step
                    
            except Exception as e2:
                print(f"안전 모드 재시도도 실패: {e2}")
                raise Exception(f"화자 분리 처리 실패: {str(e)} / 재시도 실패: {str(e2)}")
        
        diarization_time = time.time() - start_time
        return diarization, diarization_time
    
    def perform_speech_to_text(self, file_path, diarization):
        """각 화자별 구간에 대해 STT 처리를 수행"""
        stt_start_time = time.time()
        
        # 오디오 데이터 로드
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        results = []
        
        for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)
            
            # 해당 구간의 오디오 추출
            segment_audio = audio_data[start_sample:end_sample]
            
            # 임시 파일로 저장하여 Whisper STT 수행
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, segment_audio, sample_rate)
                
                # Whisper를 사용하여 STT 수행
                try:
                    result = self.whisper_model.transcribe(temp_file.name, language="ko")
                    text = result["text"].strip()
                    
                    results.append({
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker": speaker,
                        "text": text,
                        "duration": float(turn.end - turn.start)
                    })
                    
                except Exception as e:
                    print(f"STT 처리 오류: {e}")
                    results.append({
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker": speaker,
                        "text": "[STT 처리 실패]",
                        "duration": float(turn.end - turn.start)
                    })
                finally:
                    # 임시 파일 삭제
                    os.unlink(temp_file.name)
        
        stt_time = time.time() - stt_start_time
        return results, stt_time
    
    def extract_single_segment_embedding(self, audio_file, segment_result):
        """단일 세그먼트에 대한 임베딩 추출 (빠른 검증용)"""
        try:
            # 임베딩 모델 추출
            embedding_model = self.pipeline._embedding
            
            # 오디오 데이터 로드
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            
            start_time = segment_result["start"]
            end_time = segment_result["end"]
            
            # 해당 구간의 오디오 추출
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # 너무 짧은 세그먼트는 제외
            if len(segment_audio) < sample_rate * 0.5:
                return None
            
            # 임베딩 추출
            if hasattr(embedding_model, 'model_'):
                direct_model = embedding_model.model_
            else:
                from pyannote.audio import Model
                model_path = "models/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin"
                direct_model = Model.from_pretrained(model_path)
                if torch.cuda.is_available():
                    direct_model = direct_model.cuda()
            
            # 텐서 변환
            audio_tensor = torch.from_numpy(segment_audio).float()
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # 임베딩 추출
            with torch.no_grad():
                embedding = direct_model(audio_tensor)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"❌ 단일 세그먼트 임베딩 추출 실패: {e}")
            return None
    
    def verify_speakers_against_profiles(self, audio_file, results):
        """기존 화자 프로파일과 비교하여 화자 검증"""
        try:
            # 기존 프로파일 로드
            existing_profiles = {}
            # 여러 경로에서 프로파일 파일 검색
            profile_files = glob.glob("embeddings/*_profile.pkl") + glob.glob("temp/embeddings/*_profile.pkl")
            print(profile_files)
            for profile_file in profile_files:
                try:
                    with open(profile_file, 'rb') as f:
                        profile = pickle.load(f)
                        speaker_id = profile['speaker_id']
                        existing_profiles[speaker_id] = profile
                except Exception as e:
                    continue
            
            if not existing_profiles:
                # 기존 화자가 없으면 모든 화자를 새로운 화자로 처리
                verified_speakers = {}
                for result in results:
                    speaker = result["speaker"]
                    if speaker not in verified_speakers:
                        verified_speakers[speaker] = {
                            'identified_as': f"새로운_화자_{speaker}",
                            'confidence': '신규등록',
                            'similarity': 0.0,
                            'is_known': False
                        }
                return verified_speakers
            
            # 각 화자별 대표 세그먼트로 검증
            current_speakers = {}
            for result in results:
                speaker = result["speaker"]
                if speaker not in current_speakers:
                    current_speakers[speaker] = []
                current_speakers[speaker].append(result)
            
            verified_speakers = {}
            
            for current_speaker, speaker_segments in current_speakers.items():
                # 가장 긴 세그먼트 선택
                longest_segment = max(speaker_segments, key=lambda x: x['end'] - x['start'])
                
                # 대표 임베딩 추출
                representative_embedding = self.extract_single_segment_embedding(audio_file, longest_segment)
                
                if representative_embedding is None:
                    verified_speakers[current_speaker] = {
                        'identified_as': f"새로운_화자_{current_speaker}",
                        'confidence': '임베딩추출실패',
                        'similarity': 0.0,
                        'is_known': False
                    }
                    continue
                
                # 기존 화자들과 비교
                best_match = None
                best_similarity = 0
                
                for existing_speaker, profile in existing_profiles.items():
                    existing_mean = profile['mean_embedding']
                    
                    # 코사인 유사도 계산
                    similarity = cosine_similarity(
                        representative_embedding.reshape(1, -1), 
                        existing_mean.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = existing_speaker
                
                # 임계값 기준으로 판단
                threshold = 0.6
                
                if best_similarity >= threshold:
                    # 기존 화자로 인식된 경우
                    from app.services.speaker_service import SpeakerService
                    speaker_service = SpeakerService()
                    identified_as = speaker_service.get_display_name(best_match)
                    confidence_level = "높음" if best_similarity >= 0.8 else "보통"
                    is_known = True
                    matched_speaker_id = best_match
                else:
                    identified_as = f"새로운_화자_{current_speaker}"
                    confidence_level = "새로운화자"
                    is_known = False
                    matched_speaker_id = None
                
                verified_speakers[current_speaker] = {
                    'identified_as': identified_as,
                    'confidence': confidence_level,
                    'similarity': best_similarity,
                    'is_known': is_known,
                    'matched_speaker_id': matched_speaker_id
                }
            
            return verified_speakers
            
        except Exception as e:
            print(f"화자 검증 중 오류: {e}")
            return {}
    
    def generate_speaker_summary(self, results):
        """화자별 발화 요약 생성"""
        speaker_summary = {}
        for result in results:
            verified_speaker = result.get("verified_speaker", result["speaker"])
            if verified_speaker not in speaker_summary:
                speaker_summary[verified_speaker] = {
                    "total_duration": 0,
                    "segment_count": 0,
                    "texts": []
                }
            
            speaker_summary[verified_speaker]["total_duration"] += float(result["duration"])
            speaker_summary[verified_speaker]["segment_count"] += 1
            speaker_summary[verified_speaker]["texts"].append(result["text"])
        
        return speaker_summary
    
    def cleanup_files(self, original_file_path, converted_file_path):
        """업로드된 임시 파일들 정리"""
        try:
            # 원본 파일 삭제
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
                print(f"원본 파일 삭제됨: {original_file_path}")
            
            # 변환된 WAV 파일 삭제 (원본과 다른 경우에만)
            if converted_file_path != original_file_path and os.path.exists(converted_file_path):
                os.remove(converted_file_path)
                print(f"변환된 파일 삭제됨: {converted_file_path}")
                
        except Exception as e:
            print(f"파일 정리 중 오류: {e}")
