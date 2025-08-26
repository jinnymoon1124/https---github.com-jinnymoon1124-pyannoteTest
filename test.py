from pyannote.audio import Pipeline
import psutil
import GPUtil
import time
import threading
import os

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
    
    print("\n=== 화자 분리 결과 ===")
    # 결과 출력
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

except Exception as e:
    print(f"오류 발생: {e}")
    stop_monitoring.set()

print("=== 프로세스 완료 ===")
