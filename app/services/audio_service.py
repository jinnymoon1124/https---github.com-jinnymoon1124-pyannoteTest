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
import json
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from pyannote.audio import Pipeline
import glob

from app.services.s3_service import S3Service
from app.services.speaker_service import SpeakerService
from app.models.speaker_model import SpeakerModel


class AudioProcessingService:
    """오디오 처리 서비스 클래스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.pipeline = None
        self.whisper_model = None
        self.device = None
        self.s3_service = S3Service()
        self.speaker_service = SpeakerService()
        
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
    
    def get_next_speaker_id(self, embeddings_dir="temp/embeddings"):
        """다음 사용 가능한 화자 ID를 반환 (SPEAKER_XX 형식)"""
        try:
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # 기존 프로파일 파일들에서 사용된 ID 추출
            profile_files = glob.glob(f"{embeddings_dir}/*_profile.pkl")
            used_ids = set()
            
            for profile_file in profile_files:
                try:
                    # 파일명에서 SPEAKER_XX 패턴 추출
                    filename = os.path.basename(profile_file)
                    if filename.startswith('SPEAKER_') and '_profile.pkl' in filename:
                        speaker_part = filename.split('_profile.pkl')[0]
                        used_ids.add(speaker_part)
                except:
                    continue
            
            # 다음 사용 가능한 ID 찾기
            counter = 0
            while True:
                new_id = f"SPEAKER_{counter:02d}"
                if new_id not in used_ids:
                    return new_id
                counter += 1
                
        except Exception as e:
            print(f"화자 ID 생성 중 오류: {e}")
            return f"SPEAKER_{int(time.time()) % 1000:03d}"  # 타임스탬프 기반 fallback
    
    def extract_and_save_embeddings(self, audio_file, results, target_speakers=None):
        """
        화자별 임베딩 벡터를 추출하고 저장하는 함수
        target_speakers: 저장할 대상 화자 리스트 (None이면 모든 화자)
        """
        if target_speakers:
            # 새로운 화자 ID들을 미리 계산하여 로그에 표시
            embeddings_dir = "temp/embeddings"
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # 기존 사용된 ID들 확인
            profile_files = glob.glob(f"{embeddings_dir}/*_profile.pkl")
            used_ids = set()
            for profile_file in profile_files:
                try:
                    filename = os.path.basename(profile_file)
                    if filename.startswith('SPEAKER_') and '_profile.pkl' in filename:
                        speaker_part = filename.split('_profile.pkl')[0]
                        used_ids.add(speaker_part)
                except:
                    continue
            
            # 새로운 화자들에게 할당될 ID들 계산
            new_speaker_ids = []
            counter = 0
            for _ in target_speakers:
                while True:
                    new_id = f"SPEAKER_{counter:02d}"
                    if new_id not in used_ids:
                        new_speaker_ids.append(new_id)
                        used_ids.add(new_id)  # 중복 방지
                        break
                    counter += 1
                counter += 1
            
            print(f"\n=== 새로운 화자 임베딩 추출 및 저장 ({', '.join(new_speaker_ids)}) ===")
        else:
            print("\n=== 화자 임베딩 추출 및 저장 ===")
        
        # 임베딩 저장 디렉토리 생성
        embeddings_dir = "temp/embeddings"
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # 화자별 임베딩 저장소
        speaker_embeddings = defaultdict(list)
        
        try:
            # 오디오 데이터 로드
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"임베딩 추출 시작... (총 {len(results)}개 세그먼트)")
            
            for idx, result in enumerate(results):
                speaker = result["speaker"]
                start_time = result["start"]
                end_time = result["end"]
                
                # 대상 화자 필터링 (target_speakers가 지정된 경우)
                if target_speakers and speaker not in target_speakers:
                    continue
                
                # 해당 구간의 오디오 추출
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                segment_audio = audio_data[start_sample:end_sample]
                
                # 너무 짧은 세그먼트는 건너뛰기 (0.5초 미만)
                if len(segment_audio) < sample_rate * 0.5:
                    print(f"세그먼트 너무 짧음, 건너뛰기: {speaker} ({start_time:.1f}s-{end_time:.1f}s)")
                    continue
                
                try:
                    # 임베딩 추출
                    embedding_vector = self.extract_single_segment_embedding(audio_file, result)
                    
                    if embedding_vector is None:
                        continue
                    
                    # 화자별 임베딩 저장
                    speaker_embeddings[speaker].append({
                        'embedding': embedding_vector,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'text': result['text'],
                        'timestamp': current_time
                    })
                    
                    print(f"[{idx+1:2d}/{len(results)}] 임베딩 추출: {speaker} ({start_time:.1f}s-{end_time:.1f}s) - 차원: {embedding_vector.shape}")
                    
                except Exception as e:
                    print(f"임베딩 추출 실패 ({speaker}, {idx}): {e}")
            
            # 화자별 임베딩 데이터 저장
            print(f"\n임베딩 데이터 저장 중...")
            total_embeddings = 0
            saved_speaker_mapping = {}  # 원래 ID -> 새 ID 매핑
            
            for speaker, embeddings in speaker_embeddings.items():
                if not embeddings:
                    continue
                
                # 새로운 화자 ID 할당 (DynamoDB 기반)
                new_speaker_id = self.speaker_service.get_next_available_speaker_id()
                saved_speaker_mapping[speaker] = new_speaker_id
                
                # 개별 임베딩 파일 저장 (로컬)
                embedding_file = os.path.join(embeddings_dir, f"{new_speaker_id}_embeddings.pkl")
                with open(embedding_file, 'wb') as f:
                    pickle.dump(embeddings, f)
                
                # 평균 임베딩 계산 (화자 대표 벡터)
                all_embeddings = np.array([emb['embedding'] for emb in embeddings])
                mean_embedding = np.mean(all_embeddings, axis=0)
                std_embedding = np.std(all_embeddings, axis=0)
                
                # 화자 프로파일 저장 (로컬)
                speaker_profile = {
                    'speaker_id': new_speaker_id,
                    'original_label': speaker,
                    'mean_embedding': mean_embedding,
                    'std_embedding': std_embedding,
                    'num_segments': len(embeddings),
                    'total_duration': sum([emb['duration'] for emb in embeddings]),
                    'timestamp': current_time,
                    'audio_file': audio_file,
                    'embedding_dim': mean_embedding.shape[0],
                    'sample_embeddings': embeddings[:3] if len(embeddings) > 3 else embeddings  # 샘플 저장
                }
                
                profile_file = os.path.join(embeddings_dir, f"{new_speaker_id}_profile.pkl")
                with open(profile_file, 'wb') as f:
                    pickle.dump(speaker_profile, f)
                
                # S3에 화자별 폴더로 업로드
                self._upload_speaker_files_to_s3(new_speaker_id, embedding_file, profile_file)
                
                # DynamoDB에 화자 정보 저장 (기본 이름은 speaker_id)
                result = self.speaker_service.create_or_update_speaker_name(new_speaker_id, new_speaker_id)
                if not result['success']:
                    print(f"❌ 화자 {new_speaker_id} DynamoDB 저장 실패: {result['error']}")
                else:
                    print(f"✅ 화자 {new_speaker_id} DynamoDB 저장 완료: {result['action']}")
                
                total_embeddings += len(embeddings)
                print(f"✅ {new_speaker_id} 프로파일 저장: {profile_file}")
                print(f"   - 원본 라벨: {speaker}")
                print(f"   - 세그먼트 수: {len(embeddings)}")
                print(f"   - 총 발화 시간: {speaker_profile['total_duration']:.2f}초")
                print(f"   - 임베딩 차원: {mean_embedding.shape[0]}")
            
            # 전체 세션 메타데이터 저장
            session_metadata = {
                'timestamp': current_time,
                'audio_file': audio_file,
                'speakers': list(speaker_embeddings.keys()),
                'saved_speaker_mapping': saved_speaker_mapping,
                'total_segments': len(results),
                'total_embeddings': total_embeddings,
                'embedding_dim': mean_embedding.shape[0] if speaker_embeddings else 0,
                'sample_rate': sample_rate
            }
            
            metadata_file = os.path.join(embeddings_dir, f"{current_time}_session_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(session_metadata, f, ensure_ascii=False, indent=2)
            
            print(f"\n🎉 임베딩 데이터 저장 완료!")
            print(f"📁 저장 위치: {embeddings_dir}/")
            print(f"📊 총 임베딩 수: {total_embeddings}개")
            print(f"👥 새로 저장된 화자 수: {len(speaker_embeddings)}명")
            print(f"🔄 화자 ID 매핑: {saved_speaker_mapping}")
            
            return speaker_embeddings, session_metadata, saved_speaker_mapping
            
        except Exception as e:
            print(f"❌ 임베딩 추출 중 오류 발생: {e}")
            return {}, {}, {}
    
    def _upload_speaker_files_to_s3(self, speaker_id, embedding_file_path, profile_file_path):
        """화자별 임베딩 및 프로파일 파일을 S3에 업로드"""
        try:
            print(f"📤 S3 업로드 시작: {speaker_id}")
            
            # S3 경로 구조: speakers/{speaker_id}/embeddings.pkl, speakers/{speaker_id}/profile.pkl
            embedding_s3_key = f"speakers/{speaker_id}/embeddings.pkl"
            profile_s3_key = f"speakers/{speaker_id}/profile.pkl"
            
            # 임베딩 파일 업로드
            embedding_result = self.s3_service.upload_file(
                file_path=embedding_file_path,
                object_key=embedding_s3_key,
                file_type="speaker_data"
            )
            
            if embedding_result['success']:
                print(f"   ✅ 임베딩 파일 S3 업로드 완료: {embedding_s3_key}")
            else:
                print(f"   ❌ 임베딩 파일 S3 업로드 실패: {embedding_result.get('error', 'Unknown error')}")
            
            # 프로파일 파일 업로드
            profile_result = self.s3_service.upload_file(
                file_path=profile_file_path,
                object_key=profile_s3_key,
                file_type="speaker_data"
            )
            
            if profile_result['success']:
                print(f"   ✅ 프로파일 파일 S3 업로드 완료: {profile_s3_key}")
            else:
                print(f"   ❌ 프로파일 파일 S3 업로드 실패: {profile_result.get('error', 'Unknown error')}")
            
            # 업로드 성공 여부 반환
            return embedding_result['success'] and profile_result['success']
            
        except Exception as e:
            print(f"❌ S3 업로드 중 오류 발생 ({speaker_id}): {e}")
            return False
    
    def _load_speaker_profiles_from_s3(self):
        """S3에서 모든 화자 프로파일을 로드"""
        try:
            print("🔍 S3에서 화자 프로파일 로드 중...")
            
            # S3에서 speakers/ 경로의 파일 목록 조회
            files_result = self.s3_service.list_files(prefix="speakers/")
            
            if not files_result['success']:
                print(f"❌ S3 파일 목록 조회 실패: {files_result.get('error', 'Unknown error')}")
                return {}
            
            # profile.pkl 파일들만 필터링
            profile_files = [f for f in files_result['files'] if f['key'].endswith('/profile.pkl')]
            print(f"📁 S3에서 {len(profile_files)}개의 프로파일 파일 발견")
            
            existing_profiles = {}
            temp_dir = "temp/s3_cache"
            os.makedirs(temp_dir, exist_ok=True)
            
            for file_info in profile_files:
                try:
                    s3_key = file_info['key']
                    # speakers/SPEAKER_XX/profile.pkl에서 SPEAKER_XX 추출
                    speaker_id = s3_key.split('/')[1]
                    
                    # 임시 파일로 다운로드
                    local_temp_path = os.path.join(temp_dir, f"{speaker_id}_profile.pkl")
                    download_result = self.s3_service.download_file(s3_key, local_temp_path)
                    
                    if download_result['success']:
                        # 프로파일 로드
                        with open(local_temp_path, 'rb') as f:
                            profile = pickle.load(f)
                            existing_profiles[speaker_id] = profile
                            print(f"✅ S3에서 화자 프로파일 로드: {speaker_id} ({profile['num_segments']}개 세그먼트)")
                        
                        # 임시 파일 삭제
                        os.remove(local_temp_path)
                    else:
                        print(f"❌ 프로파일 다운로드 실패 ({s3_key}): {download_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"❌ 프로파일 처리 실패 ({s3_key}): {e}")
                    continue
            
            # 임시 디렉토리 정리
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
            print(f"🎉 S3에서 총 {len(existing_profiles)}명의 화자 프로파일 로드 완료")
            return existing_profiles
            
        except Exception as e:
            print(f"❌ S3 프로파일 로드 중 오류 발생: {e}")
            return {}
    
    def verify_speakers_against_profiles(self, audio_file, results):
        """기존 화자 프로파일과 비교하여 화자 검증 및 새로운 화자 저장"""
        try:
            # S3에서 기존 프로파일 로드
            existing_profiles = self._load_speaker_profiles_from_s3()
            
            # S3에서 로드된 프로파일이 없으면 로컬에서도 확인 (fallback)
            if not existing_profiles:
                print("📂 S3에서 프로파일을 찾을 수 없어 로컬 fallback 시도...")
                embeddings_dir = "temp/embeddings"
                os.makedirs(embeddings_dir, exist_ok=True)
                
                # 프로파일 파일 검색
                profile_files = glob.glob(f"{embeddings_dir}/*_profile.pkl")
                print(f"🔍 로컬 프로파일 파일 검색: {len(profile_files)}개 발견")
                
                for profile_file in profile_files:
                    try:
                        with open(profile_file, 'rb') as f:
                            profile = pickle.load(f)
                            speaker_id = profile['speaker_id']
                            existing_profiles[speaker_id] = profile
                            print(f"✅ 로컬 화자 로드: {speaker_id} ({profile['num_segments']}개 세그먼트)")
                    except Exception as e:
                        print(f"❌ 로컬 프로파일 로드 실패 ({profile_file}): {e}")
                        continue
            
            if not existing_profiles:
                print("📝 기존 등록된 화자가 없습니다. 모든 화자를 새로 등록합니다.")
                # 모든 화자에 대해 임베딩 저장
                speaker_embeddings, session_metadata, saved_speaker_mapping = self.extract_and_save_embeddings(audio_file, results)
                
                # 새로운 화자들에 대한 기본 정보 반환
                verified_speakers = {}
                for original_speaker, new_speaker_id in saved_speaker_mapping.items():
                    verified_speakers[original_speaker] = {
                        'identified_as': f"새로운_화자_{new_speaker_id}",
                        'confidence': '신규등록',
                        'similarity': 0.0,
                        'is_known': False,
                        'new_speaker_id': new_speaker_id
                    }
                
                return verified_speakers
            
            print(f"🔍 총 {len(existing_profiles)}명의 기존 화자와 사전 비교 중...")
            
            # 각 화자별 대표 세그먼트로 검증
            current_speakers = {}
            for result in results:
                speaker = result["speaker"]
                if speaker not in current_speakers:
                    current_speakers[speaker] = []
                current_speakers[speaker].append(result)
            
            verified_speakers = {}
            speakers_to_save = []  # 새로운 화자들만 저장할 리스트
            
            for current_speaker, speaker_segments in current_speakers.items():
                print(f"\n🎯 {current_speaker} 사전 검증 중...")
                
                # 가장 긴 세그먼트 선택
                longest_segment = max(speaker_segments, key=lambda x: x['end'] - x['start'])
                
                # 대표 임베딩 추출
                representative_embedding = self.extract_single_segment_embedding(audio_file, longest_segment)
                
                if representative_embedding is None:
                    print(f"   ❌ 대표 임베딩 추출 실패")
                    speakers_to_save.append(current_speaker)
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
                all_similarities = {}
                
                for existing_speaker, profile in existing_profiles.items():
                    existing_mean = profile['mean_embedding']
                    
                    # 코사인 유사도 계산
                    similarity = cosine_similarity(
                        representative_embedding.reshape(1, -1), 
                        existing_mean.reshape(1, -1)
                    )[0][0]
                    
                    all_similarities[existing_speaker] = similarity
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = existing_speaker
                
                # 임계값 기준으로 판단 (0.6 사용)
                threshold = 0.6
                
                if best_similarity >= threshold:
                    # 기존 화자로 인식된 경우 - DynamoDB에서 실제 이름 가져오기
                    display_name = self.speaker_service.get_display_name(best_match)
                    identified_as = display_name
                    confidence_level = "높음" if best_similarity >= 0.8 else "보통"
                    is_known = True
                    matched_speaker_id = best_match
                    print(f"   ✅ 기존 화자 매칭: {best_match} -> {display_name} (유사도: {best_similarity:.4f}, 신뢰도: {confidence_level})")
                    print(f"   💾 임베딩 저장 생략 - 기존 프로필 재사용")
                else:
                    identified_as = f"새로운_화자_{current_speaker}"
                    confidence_level = "새로운화자"
                    is_known = False
                    matched_speaker_id = None
                    speakers_to_save.append(current_speaker)
                    print(f"   ❓ 새로운 화자: {identified_as} (최고 유사도: {best_similarity:.4f})")
                    print(f"   💾 임베딩 저장 예정")
                
                # 상위 3개 유사도 출력
                sorted_similarities = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)
                print(f"   📊 유사도 순위:")
                for i, (speaker, sim) in enumerate(sorted_similarities[:3], 1):
                    status = "✅" if sim >= threshold else "❌"
                    print(f"      {status} {i}위: {speaker} ({sim:.4f})")
                
                verified_speakers[current_speaker] = {
                    'identified_as': identified_as,
                    'confidence': confidence_level,
                    'similarity': best_similarity,
                    'is_known': is_known,
                    'matched_speaker_id': matched_speaker_id,
                    'all_similarities': all_similarities
                }
            
            print(f"\n🎉 사전 검증 완료!")
            print(f"📊 검증 결과 요약:")
            total_speakers = len(verified_speakers)
            known_count = sum(1 for v in verified_speakers.values() if v['is_known'])
            new_count = len(verified_speakers) - known_count
            print(f"   - 전체 화자: {total_speakers}명")
            print(f"   - 기존 화자: {known_count}명 (임베딩 저장 생략)")
            print(f"   - 새로운 화자: {new_count}명 (임베딩 저장 진행)")
            
            # 새로운 화자들에 대해서만 전체 임베딩 저장
            if speakers_to_save:
                print(f"\n💾 새로운 화자들에 대한 임베딩 저장 시작...")
                
                # 새로운 화자들의 결과만 필터링
                new_speaker_results = [r for r in results if r['speaker'] in speakers_to_save]
                
                # 임베딩 추출 및 저장 (새로운 화자들만)
                new_speaker_embeddings, session_metadata, saved_speaker_mapping = self.extract_and_save_embeddings(
                    audio_file, new_speaker_results, speakers_to_save
                )
                
                # 새로 저장된 화자들의 정보 업데이트
                for original_speaker, new_speaker_id in saved_speaker_mapping.items():
                    if original_speaker in verified_speakers:
                        verified_speakers[original_speaker]['identified_as'] = f"새로운_화자_{new_speaker_id}"
                        verified_speakers[original_speaker]['new_speaker_id'] = new_speaker_id
            
            return verified_speakers
            
        except Exception as e:
            print(f"❌ 화자 검증 중 오류 발생: {e}")
            return {}
    
    def generate_speaker_summary(self, results):
        """화자별 발화 요약 생성"""
        speaker_summary = {}
        for result in results:
            # 화자 키 결정 (기존 화자면 실제 이름, 새로운 화자면 speaker_id 사용)
            if result.get('is_known_speaker'):
                # 기존 화자인 경우 실제 이름 사용
                speaker_key = result.get("verified_speaker") or result.get("speaker_id", "UNKNOWN")
            else:
                # 새로운 화자인 경우 speaker_id 사용
                speaker_key = result.get("speaker_id") or result.get("verified_speaker") or result.get("speaker", "UNKNOWN")
            
            if speaker_key not in speaker_summary:
                speaker_summary[speaker_key] = {
                    "total_duration": 0,
                    "segment_count": 0,
                    "texts": []
                }
            
            speaker_summary[speaker_key]["total_duration"] += float(result["duration"])
            speaker_summary[speaker_key]["segment_count"] += 1
            speaker_summary[speaker_key]["texts"].append(result["text"])
        
        return speaker_summary
    
    # 결과값 확인용 / 실제 서비스 시에는 필요하지 않은 기능
    def save_transcript_to_file(self, results, speaker_summary, verified_speakers, processing_info):
        """STT 결과를 가독성 좋은 대화록 파일로 저장"""
        try:
            # 결과 디렉토리 생성
            result_dir = "temp/result"
            os.makedirs(result_dir, exist_ok=True)
            
            # 파일명 생성 (타임스탬프 포함)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            transcript_filename = f"transcript_{timestamp}.txt"
            transcript_path = os.path.join(result_dir, transcript_filename)
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                # 헤더 정보 작성
                f.write("=" * 80 + "\n")
                f.write("                          음성 인식 대화록\n")
                f.write("=" * 80 + "\n")
                f.write(f"생성 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}\n")
                f.write(f"총 처리 시간: {processing_info.get('total_time', 0):.2f}초\n")
                f.write(f"총 세그먼트 수: {processing_info.get('total_segments', 0)}개\n")
                f.write(f"인식된 화자 수: {processing_info.get('unique_speakers', 0)}명\n")
                f.write("\n")
                
                # 화자 정보 요약
                f.write("-" * 50 + "\n")
                f.write("화자 정보 요약\n")
                f.write("-" * 50 + "\n")
                for speaker, info in speaker_summary.items():
                    f.write(f"• {speaker}\n")
                    f.write(f"  - 총 발화 시간: {info['total_duration']:.1f}초\n")
                    f.write(f"  - 발화 횟수: {info['segment_count']}회\n")
                    
                    # 검증 정보 추가
                    original_speaker = None
                    for orig, verified_info in verified_speakers.items():
                        if verified_info['identified_as'] == speaker:
                            original_speaker = orig
                            break
                    
                    if original_speaker and verified_speakers.get(original_speaker):
                        verify_info = verified_speakers[original_speaker]
                        f.write(f"  - 화자 인식: {verify_info['confidence']}")
                        if verify_info['is_known']:
                            f.write(f" (유사도: {verify_info['similarity']:.2f})")
                        elif verify_info.get('new_speaker_id'):
                            f.write(f" (새 ID: {verify_info['new_speaker_id']})")
                        f.write("\n")
                    f.write("\n")
                
                # 대화록 본문
                f.write("=" * 80 + "\n")
                f.write("                            대화록 본문\n")
                f.write("=" * 80 + "\n\n")
                
                # 시간순으로 정렬된 세그먼트들을 대화록 형태로 작성
                sorted_results = sorted(results, key=lambda x: x['start'])
                
                for i, result in enumerate(sorted_results, 1):
                    # 시간 정보를 분:초 형태로 변환
                    start_min = int(result['start'] // 60)
                    start_sec = int(result['start'] % 60)
                    end_min = int(result['end'] // 60)
                    end_sec = int(result['end'] % 60)
                    
                    # 화자명 결정 (기존 화자면 실제 이름, 새로운 화자면 speaker_id 사용)
                    if result.get('is_known_speaker'):
                        # 기존 화자인 경우 실제 이름 사용
                        speaker_name = result.get('verified_speaker', result.get('speaker_id', 'UNKNOWN'))
                    else:
                        # 새로운 화자인 경우 speaker_id 사용
                        speaker_name = result.get('speaker_id') or result.get('verified_speaker', result.get('speaker', 'UNKNOWN'))
                    
                    # 대화록 형태로 작성
                    f.write(f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}] {speaker_name}\n")
                    f.write(f"{result['text']}\n")
                    
                    # 화자 검증 정보 (상세 모드)
                    if result.get('is_known_speaker') is not None:
                        confidence = result.get('verification_confidence', 'N/A')
                        similarity = result.get('similarity_score', 0)
                        original_label = result.get('original_speaker_label', '')
                        
                        f.write(f"(검증: {confidence}")
                        if similarity > 0:
                            f.write(f", 유사도: {similarity:.2f}")
                        if original_label:
                            f.write(f", 원본: {original_label}")
                        f.write(")\n")
                    
                    f.write("\n")
                
                # 푸터
                f.write("=" * 80 + "\n")
                f.write("                          대화록 종료\n")
                f.write("=" * 80 + "\n")
            
            print(f"대화록 파일 저장 완료: {transcript_path}")
            
            # S3에 대화록 파일 업로드 (오디오 파일과 동일한 경로 구조)
            s3_transcript_path = self._upload_transcript_to_s3(transcript_path, timestamp)
            
            return transcript_path
            
        except Exception as e:
            print(f"대화록 파일 저장 중 오류: {e}")
            return None
    
    def _upload_transcript_to_s3(self, transcript_path, timestamp):
        """대화록 파일을 S3에 업로드 (오디오 파일과 동일한 경로 구조)"""
        try:
            from datetime import datetime
            now = datetime.now()
            
            # 파일명 생성: transcript_yyyymmdd_HHMMSS.txt
            transcript_filename = f"transcript_{timestamp}.txt"
            
            # S3 키 생성: audio/yyyy/mm/transcript_파일명 (오디오와 동일한 경로)
            s3_key = f"audio/{now.strftime('%Y')}/{now.strftime('%m')}/{transcript_filename}"
            
            print(f"📤 대화록 파일 S3 업로드: {s3_key}")
            
            # S3에 업로드
            result = self.s3_service.upload_file(
                file_path=transcript_path,
                object_key=s3_key,
                file_type="transcript"
            )
            
            if result['success']:
                print(f"✅ 대화록 파일 S3 업로드 완료: {s3_key}")
                return s3_key
            else:
                print(f"❌ 대화록 파일 S3 업로드 실패: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ 대화록 파일 S3 업로드 중 오류: {e}")
            return None
    
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
