from pyannote.audio import Pipeline
import psutil
import GPUtil
import time
import threading
import os
import whisper
import librosa
import soundfile as sf
import pickle
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def monitor_resources(stop_event):
    """시스템 리소스를 모니터링하는 함수"""
    process = psutil.Process(os.getpid())
    
    while not stop_event.is_set():
        # 메모리 사용량 (MB 단위)
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # CPU 사용률
        cpu_percent = process.cpu_percent()
        
        # GPU 사용량 확인
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # 첫 번째 GPU 사용
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_total = gpu.memoryTotal
            gpu_utilization = gpu.load * 100
            gpu_temp = gpu.temperature if hasattr(gpu, 'temperature') else 'N/A'
            print(f"[리소스] 메모리: {memory_mb:.1f}MB | CPU: {cpu_percent:.1f}% | GPU 메모리: {gpu_memory_used}MB/{gpu_memory_total}MB | GPU 사용률: {gpu_utilization:.1f}% | GPU 온도: {gpu_temp}°C")
        else:
            print(f"[리소스] 메모리: {memory_mb:.1f}MB | CPU: {cpu_percent:.1f}% | GPU: 감지되지 않음")
        
        time.sleep(2)  # 2초마다 모니터링

def extract_and_save_embeddings(pipeline, diarization, audio_file, results):
    """
    화자별 임베딩 벡터를 추출하고 저장하는 함수
    """
    print("\n=== 화자 임베딩 추출 및 저장 ===")
    
    # 임베딩 저장 디렉토리 생성
    embeddings_dir = "embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # 화자별 임베딩 저장소
    speaker_embeddings = defaultdict(list)
    
    try:
        # 임베딩 모델 추출 (파이프라인 내부의 임베딩 모델 접근)
        embedding_model = pipeline._embedding
        print(f"임베딩 모델: {type(embedding_model)}")
        
        # 오디오 데이터 로드
        audio_data, sample_rate = librosa.load(audio_file, sr=16000)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"임베딩 추출 시작... (총 {len(results)}개 세그먼트)")
        
        for idx, result in enumerate(results):
            speaker = result["speaker"]
            start_time = result["start"]
            end_time = result["end"]
            
            # 해당 구간의 오디오 추출
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # 너무 짧은 세그먼트는 건너뛰기 (0.5초 미만)
            if len(segment_audio) < sample_rate * 0.5:
                print(f"세그먼트 너무 짧음, 건너뛰기: {speaker} ({start_time:.1f}s-{end_time:.1f}s)")
                continue
            
            try:
                # 직접 WeSpeaker 모델 사용하여 임베딩 추출 (파일 없이 메모리에서 직접)
                from pyannote.audio import Model
                
                # 내부 모델에 직접 접근 또는 새로 로드
                if hasattr(embedding_model, 'model_'):
                    direct_model = embedding_model.model_
                else:
                    # WeSpeaker 모델 직접 로드
                    model_path = "models/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin"
                    direct_model = Model.from_pretrained(model_path)
                    if torch.cuda.is_available():
                        direct_model = direct_model.cuda()
                
                # 올바른 텐서 차원으로 변환: (batch, channels, samples)
                audio_tensor = torch.from_numpy(segment_audio).float()
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
                
                if torch.cuda.is_available():
                    audio_tensor = audio_tensor.cuda()
                
                # 임베딩 추출
                with torch.no_grad():
                    embedding = direct_model(audio_tensor)
                
                # GPU tensor를 CPU numpy로 변환
                embedding_vector = embedding.cpu().numpy().flatten()
                
                # 화자별 임베딩 저장
                speaker_embeddings[speaker].append({
                    'embedding': embedding_vector,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'text': result['text'],
                    'filename': result['filename'],
                    'timestamp': current_time
                })
                
                print(f"[{idx+1:2d}/{len(results)}] 임베딩 추출: {speaker} ({start_time:.1f}s-{end_time:.1f}s) - 차원: {embedding_vector.shape}")
                
            except Exception as e:
                print(f"임베딩 추출 실패 ({speaker}, {idx}): {e}")
        
        # 화자별 임베딩 데이터 저장
        print(f"\n임베딩 데이터 저장 중...")
        total_embeddings = 0
        
        for speaker, embeddings in speaker_embeddings.items():
            if not embeddings:
                continue
                
            # 개별 임베딩 파일 저장
            embedding_file = os.path.join(embeddings_dir, f"{current_time}_{speaker}_embeddings.pkl")
            with open(embedding_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # 평균 임베딩 계산 (화자 대표 벡터)
            all_embeddings = np.array([emb['embedding'] for emb in embeddings])
            mean_embedding = np.mean(all_embeddings, axis=0)
            std_embedding = np.std(all_embeddings, axis=0)
            
            # 화자 프로파일 저장
            speaker_profile = {
                'speaker_id': speaker,
                'mean_embedding': mean_embedding,
                'std_embedding': std_embedding,
                'num_segments': len(embeddings),
                'total_duration': sum([emb['duration'] for emb in embeddings]),
                'timestamp': current_time,
                'audio_file': audio_file,
                'embedding_dim': mean_embedding.shape[0],
                'sample_embeddings': embeddings[:3] if len(embeddings) > 3 else embeddings  # 샘플 저장
            }
            
            profile_file = os.path.join(embeddings_dir, f"{current_time}_{speaker}_profile.pkl")
            with open(profile_file, 'wb') as f:
                pickle.dump(speaker_profile, f)
            
            total_embeddings += len(embeddings)
            print(f"✅ {speaker} 프로파일 저장: {profile_file}")
            print(f"   - 세그먼트 수: {len(embeddings)}")
            print(f"   - 총 발화 시간: {speaker_profile['total_duration']:.2f}초")
            print(f"   - 임베딩 차원: {mean_embedding.shape[0]}")
        
        # 전체 세션 메타데이터 저장
        session_metadata = {
            'timestamp': current_time,
            'audio_file': audio_file,
            'speakers': list(speaker_embeddings.keys()),
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
        print(f"👥 화자 수: {len(speaker_embeddings)}명")
        
        return speaker_embeddings, session_metadata
        
    except Exception as e:
        print(f"❌ 임베딩 추출 중 오류 발생: {e}")
        return {}, {}

def load_speaker_embeddings(embeddings_dir="embeddings"):
    """저장된 화자 임베딩 데이터 로드"""
    import glob
    
    profiles = {}
    profile_files = glob.glob(f"{embeddings_dir}/*_profile.pkl")
    
    print(f"\n=== 저장된 화자 프로파일 로드 ===")
    print(f"발견된 프로파일 파일: {len(profile_files)}개")
    
    for profile_file in profile_files:
        try:
            with open(profile_file, 'rb') as f:
                profile = pickle.load(f)
                speaker_id = profile['speaker_id']
                profiles[speaker_id] = profile
                print(f"✅ 로드: {speaker_id} ({profile['num_segments']}개 세그먼트)")
        except Exception as e:
            print(f"❌ 프로파일 로드 실패 ({profile_file}): {e}")
    
    return profiles

def identify_speaker(new_embedding, known_profiles, threshold=0.8):
    """새로운 임베딩과 기존 화자 프로파일을 비교하여 화자 식별"""
    if not known_profiles:
        return "UNKNOWN", 0.0
    
    best_match = None
    best_similarity = 0
    
    # 입력 임베딩을 2D 배열로 변환 (cosine_similarity 요구사항)
    if new_embedding.ndim == 1:
        new_embedding = new_embedding.reshape(1, -1)
    
    for speaker_id, profile in known_profiles.items():
        mean_embedding = profile['mean_embedding'].reshape(1, -1)
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(new_embedding, mean_embedding)[0][0]
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = speaker_id
    
    if best_similarity > threshold:
        return best_match, best_similarity
    else:
        return "UNKNOWN", best_similarity

print("=== 화자 분리 프로세스 시작 ===")
print("시스템 리소스 모니터링을 시작합니다...")

# 리소스 모니터링 시작
stop_monitoring = threading.Event()
monitor_thread = threading.Thread(target=monitor_resources, args=(stop_monitoring,))
monitor_thread.daemon = True
monitor_thread.start()

# 초기 메모리 상태 출력
initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
print(f"초기 메모리 사용량: {initial_memory:.1f}MB")

try:
    # yaml 불러오기
    print("파이프라인 로딩 중...")
    pipeline = Pipeline.from_pretrained("models/pyannote_diarization_config.yaml")
    
    # GPU 사용 가능 여부 확인 및 파이프라인을 GPU로 이동
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pipeline.to(device)
        print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"파이프라인을 GPU로 이동했습니다.")
    else:
        print("GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
    
    # 파이프라인 로딩 후 메모리 사용량
    after_load_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"파이프라인 로딩 후 메모리 사용량: {after_load_memory:.1f}MB (증가량: {after_load_memory - initial_memory:.1f}MB)")

    # 오디오 파일 경로
    audio_file = "audio/test.wav"  # 테스트할 음성 파일

    # 화자 분리 실행
    print("화자 분리 처리 중...")
    start_time = time.time()
    diarization = pipeline(audio_file)
    end_time = time.time()
    
    # 처리 완료 후 메모리 사용량
    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"처리 완료 후 메모리 사용량: {final_memory:.1f}MB")
    print(f"총 처리 시간: {end_time - start_time:.2f}초")
    
    # 리소스 모니터링 중단
    stop_monitoring.set()
    
    print("\n=== 화자 분리 및 STT 처리 ===")
    
    # temp 폴더 생성
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Whisper 모델 로딩 (한국어에 최적화된 large-v3 모델 사용)
    print("Whisper STT 모델 로딩 중...")
    whisper_model = whisper.load_model("large-v3")
    if torch.cuda.is_available():
        # Whisper 모델도 GPU로 이동
        whisper_model = whisper_model.to(device)
        print("Whisper 모델을 GPU로 이동했습니다.")
    
    # 원본 오디오 파일 로딩 (librosa 사용)
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    
    # 현재 시간 (파일명용)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n화자별 음성 분리 및 텍스트 변환 시작...")
    results = []
    
    # 각 화자별 구간을 처리
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_sample = int(turn.start * sample_rate)
        end_sample = int(turn.end * sample_rate)
        
        # 해당 구간의 오디오 추출
        segment_audio = audio_data[start_sample:end_sample]
        
        # 파일명 생성: {현재시간}_{화자번호}_{구간번호}.wav
        filename = f"{current_time}_{speaker}_{i+1:03d}.wav"
        filepath = os.path.join(temp_dir, filename)
        
        # 오디오 세그먼트를 파일로 저장
        sf.write(filepath, segment_audio, sample_rate)
        
        # Whisper를 사용하여 STT 수행
        try:
            result = whisper_model.transcribe(filepath, language="ko")
            text = result["text"].strip()
            
            # 결과 저장
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "filename": filename,
                "text": text
            })
            
            print(f"[{i+1:2d}] {turn.start:.1f}s-{turn.end:.1f}s | {speaker} | {filename}")
            print(f"     텍스트: {text}")
            print()
            
        except Exception as e:
            print(f"STT 처리 오류 ({filename}): {e}")
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "filename": filename,
                "text": "[STT 처리 실패]"
            })
    
    print(f"\n=== 처리 완료 ===")
    print(f"총 {len(results)}개 구간 처리 완료")
    print(f"분리된 오디오 파일들이 '{temp_dir}' 폴더에 저장되었습니다.")
    
    # 화자별 요약 출력
    print(f"\n=== 화자별 발화 요약 ===")
    speaker_texts = {}
    for result in results:
        speaker = result["speaker"]
        if speaker not in speaker_texts:
            speaker_texts[speaker] = []
        speaker_texts[speaker].append(result["text"])
    
    for speaker, texts in speaker_texts.items():
        print(f"\n[{speaker}]:")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")
    
    print(f"\n전체 대화 내용:")
    for result in results:
        print(f"{result['start']:.1f}s [{result['speaker']}]: {result['text']}")
    
    # 임베딩 추출 및 저장
    speaker_embeddings, session_metadata = extract_and_save_embeddings(pipeline, diarization, audio_file, results)
    
    # 저장된 임베딩 데이터 요약 출력
    if speaker_embeddings:
        print(f"\n=== 임베딩 저장 요약 ===")
        for speaker, embeddings in speaker_embeddings.items():
            print(f"{speaker}: {len(embeddings)}개 임베딩 벡터 저장")
        
        # 임베딩 활용 예시 (기존 화자 프로파일 로드 테스트)
        print(f"\n=== 저장된 프로파일 로드 테스트 ===")
        loaded_profiles = load_speaker_embeddings()
        
        if loaded_profiles:
            print(f"로드된 화자 프로파일: {list(loaded_profiles.keys())}")
            
            # 화자 식별 테스트 (첫 번째 임베딩으로 테스트)
            if results:
                first_speaker = results[0]['speaker']
                if first_speaker in speaker_embeddings and speaker_embeddings[first_speaker]:
                    test_embedding = speaker_embeddings[first_speaker][0]['embedding']
                    identified_speaker, similarity = identify_speaker(test_embedding, loaded_profiles, threshold=0.7)
                    print(f"테스트 임베딩 식별 결과: {identified_speaker} (유사도: {similarity:.3f})")
        else:
            print("로드된 프로파일이 없습니다.")

except Exception as e:
    print(f"오류 발생: {e}")
    stop_monitoring.set()

print("=== 프로세스 완료 ===")
