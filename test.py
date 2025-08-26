from pyannote.audio import Pipeline
import psutil
import GPUtil
import time
import threading
import os
import whisper
import librosa
import soundfile as sf
from datetime import datetime

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

except Exception as e:
    print(f"오류 발생: {e}")
    stop_monitoring.set()

print("=== 프로세스 완료 ===")
