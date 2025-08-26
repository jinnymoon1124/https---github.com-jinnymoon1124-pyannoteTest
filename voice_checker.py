# -*- coding: utf-8 -*-
"""
voice_checker.py

간단한 음성 화자 확인 도구
- 음성 파일 하나 업로드 → 기존 화자인지 확인
- 있으면 누구인지 알려주고, 없으면 "새로운 화자" 리턴
"""

import torch
import numpy as np
from pyannote.audio import Model
import librosa
import pickle
import glob
import os
from sklearn.metrics.pairwise import cosine_similarity

class VoiceChecker:
    """간단한 음성 화자 확인 클래스"""
    
    def __init__(self):
        self.model = None
        self.known_speakers = {}
        
        print("🎙️ 음성 화자 확인 시스템 초기화 중...")
        
        # 모델 로드
        self._load_model()
        
        # 기존 화자 데이터 로드
        self._load_speakers()
    
    def _load_model(self):
        """임베딩 모델 로드"""
        try:
            model_path = "models/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin"
            self.model = Model.from_pretrained(model_path)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            print("✅ 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def _load_speakers(self):
        """기존 화자 데이터 로드 (embeddings.pkl 파일 사용으로 정확도 향상)"""
        try:
            # embeddings 파일들 찾기
            embeddings_files = glob.glob("embeddings/*_embeddings.pkl")
            profile_files = glob.glob("embeddings/*_profile.pkl")
            
            if not embeddings_files:
                print("⚠️ 등록된 화자가 없습니다. 먼저 test.py를 실행하세요.")
                return
            
            print(f"📊 개별 임베딩 파일 사용으로 정확도 향상 모드")
            
            for embeddings_file in embeddings_files:
                try:
                    # embeddings 데이터 로드
                    with open(embeddings_file, 'rb') as f:
                        embeddings_data = pickle.load(f)
                    
                    # 화자 ID 추출 (파일명에서)
                    filename = os.path.basename(embeddings_file)
                    # 예: 20250826_042213_SPEAKER_00_embeddings.pkl
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        speaker_id = f"{parts[2]}_{parts[3]}"  # SPEAKER_00
                    else:
                        speaker_id = parts[2] if len(parts) > 2 else "UNKNOWN"
                    
                    # 해당하는 profile도 로드 (메타데이터용)
                    profile_file = embeddings_file.replace('_embeddings.pkl', '_profile.pkl')
                    profile_data = {}
                    if os.path.exists(profile_file):
                        with open(profile_file, 'rb') as f:
                            profile_data = pickle.load(f)
                    
                    # 모든 임베딩 벡터들 추출
                    all_embeddings = []
                    for emb_data in embeddings_data:
                        all_embeddings.append(emb_data['embedding'])
                    
                    self.known_speakers[speaker_id] = {
                        'speaker_id': speaker_id,
                        'all_embeddings': np.array(all_embeddings),  # 모든 개별 임베딩들
                        'num_segments': len(all_embeddings),
                        'total_duration': profile_data.get('total_duration', 0),
                        'mean_embedding': profile_data.get('mean_embedding'),  # 백업용
                        'embeddings_data': embeddings_data  # 상세 정보
                    }
                    
                    print(f"✅ {speaker_id}: {len(all_embeddings)}개 개별 임베딩 로드")
                    
                except Exception as e:
                    print(f"❌ 임베딩 파일 로드 실패 ({embeddings_file}): {e}")
            
            print(f"✅ 총 {len(self.known_speakers)}명의 화자 데이터 로드 완료 (고정확도 모드)")
            
        except Exception as e:
            print(f"❌ 화자 데이터 로드 실패: {e}")
    
    def check_voice(self, audio_file):
        """
        음성 파일 확인
        
        Args:
            audio_file: 확인할 음성 파일 경로
            
        Returns:
            dict: 확인 결과
        """
        
        print(f"\n🎵 음성 파일 확인: {audio_file}")
        
        if not os.path.exists(audio_file):
            return {"error": f"파일을 찾을 수 없습니다: {audio_file}"}
        
        if not self.known_speakers:
            return {"error": "등록된 화자가 없습니다"}
        
        try:
            # 1단계: 오디오 로드
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            duration = len(audio_data) / sample_rate
            
            print(f"📏 음성 길이: {duration:.1f}초")
            
            # 너무 짧은 파일 체크
            if duration < 0.5:
                return {"error": f"음성이 너무 짧습니다 ({duration:.1f}초). 최소 0.5초 필요"}
            
            # 2단계: 임베딩 추출
            print("🔄 임베딩 추출 중...")
            embedding = self._extract_embedding(audio_data)
            
            if embedding is None:
                return {"error": "임베딩 추출 실패"}
            
            # 3단계: 화자 확인 (기본적으로 best_match 방법 사용)
            result = self._find_speaker(embedding, threshold=0.6, method="best_match")
            
            # 4단계: 결과 출력
            self._print_result(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return {"error": str(e)}
    
    def _extract_embedding(self, audio_data):
        """임베딩 추출"""
        try:
            # 너무 긴 경우 중간 부분만 사용 (최대 10초)
            if len(audio_data) > 10 * 16000:
                start_idx = len(audio_data) // 2 - 5 * 16000
                end_idx = len(audio_data) // 2 + 5 * 16000
                audio_data = audio_data[start_idx:end_idx]
            
            # 텐서 변환
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # 임베딩 추출
            with torch.no_grad():
                embedding = self.model(audio_tensor)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"❌ 임베딩 추출 실패: {e}")
            return None
    
    def _find_speaker(self, test_embedding, threshold=0.6, method="best_match"):
        """
        화자 찾기 (개별 임베딩들과 비교로 정확도 향상)
        
        Args:
            test_embedding: 테스트 임베딩 벡터
            threshold: 유사도 임계값
            method: 비교 방법
                - "best_match": 가장 유사한 세그먼트와 비교
                - "average": 모든 세그먼트와의 평균 유사도
                - "top_k": 상위 K개 세그먼트 평균 (K=3)
        """
        
        if test_embedding.ndim == 1:
            test_embedding = test_embedding.reshape(1, -1)
        
        best_speaker = None
        best_similarity = 0
        similarity_details = {}
        
        print(f"🔍 화자 식별 중... (방법: {method})")
        
        # 모든 등록된 화자와 비교
        for speaker_id, speaker_data in self.known_speakers.items():
            all_embeddings = speaker_data['all_embeddings']  # (N, 256) 형태
            num_segments = speaker_data['num_segments']
            
            # 각 개별 임베딩과의 유사도 계산
            similarities = []
            for i, stored_embedding in enumerate(all_embeddings):
                stored_embedding = stored_embedding.reshape(1, -1)
                sim = cosine_similarity(test_embedding, stored_embedding)[0][0]
                similarities.append(sim)
            
            # 방법별로 최종 유사도 계산
            if method == "best_match":
                # 가장 유사한 세그먼트 사용
                final_similarity = max(similarities)
                best_segment_idx = similarities.index(final_similarity)
                
            elif method == "average":
                # 모든 세그먼트와의 평균 유사도
                final_similarity = np.mean(similarities)
                best_segment_idx = similarities.index(max(similarities))
                
            elif method == "top_k":
                # 상위 3개 세그먼트 평균
                k = min(3, len(similarities))
                top_k_similarities = sorted(similarities, reverse=True)[:k]
                final_similarity = np.mean(top_k_similarities)
                best_segment_idx = similarities.index(max(similarities))
            
            # 상세 정보 저장
            similarity_details[speaker_id] = {
                'final_similarity': final_similarity,
                'best_segment_similarity': max(similarities),
                'average_similarity': np.mean(similarities),
                'num_segments': num_segments,
                'best_segment_idx': best_segment_idx,
                'all_similarities': similarities
            }
            
            # 최고 유사도 업데이트
            if final_similarity > best_similarity:
                best_similarity = final_similarity
                best_speaker = speaker_id
        
        # 결과 출력
        print(f"\n📊 화자별 유사도 분석:")
        for speaker_id, details in similarity_details.items():
            status = "✅" if details['final_similarity'] >= threshold else "❌"
            print(f"{status} {speaker_id}: {details['final_similarity']:.4f} "
                  f"(최고: {details['best_segment_similarity']:.4f}, "
                  f"평균: {details['average_similarity']:.4f}, "
                  f"{details['num_segments']}개 세그먼트)")
        
        # 결과 판단
        if best_similarity >= threshold:
            confidence = "매우 높음" if best_similarity >= 0.9 else "높음" if best_similarity >= 0.8 else "보통"
            
            return {
                "found": True,
                "speaker": best_speaker,
                "similarity": best_similarity,
                "confidence": confidence,
                "method": method,
                "details": similarity_details[best_speaker],
                "all_details": similarity_details
            }
        else:
            return {
                "found": False,
                "speaker": None,
                "similarity": best_similarity,
                "confidence": "낮음",
                "method": method,
                "all_details": similarity_details
            }
    
    def _print_result(self, result):
        """결과 출력 (고정확도 모드 정보 포함)"""
        print(f"\n🎯 === 고정확도 화자 식별 결과 ===")
        
        if "error" in result:
            print(f"❌ 오류: {result['error']}")
            return
        
        if result["found"]:
            print(f"✅ 기존 화자 발견!")
            print(f"👤 화자: {result['speaker']}")
            print(f"📊 최종 유사도: {result['similarity']:.4f}")
            print(f"🎯 신뢰도: {result['confidence']}")
            print(f"🔍 비교 방법: {result['method']}")
            
            # 상세 정보 출력
            if 'details' in result:
                details = result['details']
                print(f"📈 상세 분석:")
                print(f"   - 최고 세그먼트 유사도: {details['best_segment_similarity']:.4f}")
                print(f"   - 평균 유사도: {details['average_similarity']:.4f}")
                print(f"   - 비교한 세그먼트 수: {details['num_segments']}개")
                
                # 가장 유사한 세그먼트 정보
                best_idx = details['best_segment_idx']
                if hasattr(self, 'known_speakers') and result['speaker'] in self.known_speakers:
                    embeddings_data = self.known_speakers[result['speaker']].get('embeddings_data', [])
                    if best_idx < len(embeddings_data):
                        best_segment = embeddings_data[best_idx]
                        print(f"   - 가장 유사한 세그먼트: {best_segment['start_time']:.1f}s-{best_segment['end_time']:.1f}s")
                        print(f"   - 해당 텍스트: '{best_segment['text'][:50]}{'...' if len(best_segment['text']) > 50 else ''}'")
        else:
            print(f"❓ 새로운 화자입니다")
            print(f"📊 최고 유사도: {result['similarity']:.4f} (임계값: 0.6 미만)")
            print(f"💡 등록된 모든 화자와 개별 비교했지만 일치하지 않습니다")
            
            # 가장 가까운 화자 정보
            if 'all_details' in result:
                all_details = result['all_details']
                closest_speaker = max(all_details.items(), key=lambda x: x[1]['final_similarity'])
                print(f"🔍 가장 가까운 화자: {closest_speaker[0]} (유사도: {closest_speaker[1]['final_similarity']:.4f})")
    
    def check_voice_advanced(self, audio_file, method="best_match", threshold=0.6):
        """
        고급 화자 확인 (방법 선택 가능)
        
        Args:
            audio_file: 음성 파일 경로
            method: 비교 방법 ("best_match", "average", "top_k")
            threshold: 유사도 임계값
        """
        print(f"\n🎵 고정확도 음성 파일 확인: {audio_file}")
        print(f"🔧 비교 방법: {method}, 임계값: {threshold}")
        
        if not os.path.exists(audio_file):
            return {"error": f"파일을 찾을 수 없습니다: {audio_file}"}
        
        if not self.known_speakers:
            return {"error": "등록된 화자가 없습니다"}
        
        try:
            # 오디오 로드
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            duration = len(audio_data) / sample_rate
            
            print(f"📏 음성 길이: {duration:.1f}초")
            
            if duration < 0.5:
                return {"error": f"음성이 너무 짧습니다 ({duration:.1f}초). 최소 0.5초 필요"}
            
            # 임베딩 추출
            print("🔄 임베딩 추출 중...")
            embedding = self._extract_embedding(audio_data)
            
            if embedding is None:
                return {"error": "임베딩 추출 실패"}
            
            # 화자 확인 (고급 방법 사용)
            result = self._find_speaker(embedding, threshold, method)
            
            # 결과 출력
            self._print_result(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return {"error": str(e)}

def main():
    """메인 함수"""
    import sys
    
    print("🎙️ 간단 음성 화자 확인 도구")
    print("="*40)
    
    checker = VoiceChecker()
    
    if len(sys.argv) > 1:
        # 명령행에서 파일 지정
        audio_file = sys.argv[1]
        result = checker.check_voice(audio_file)
        
    else:
        # 대화형 모드
        while True:
            print(f"\n📁 음성 파일 경로를 입력하세요 (종료: quit)")
            audio_file = input("파일 경로: ").strip()
            
            if audio_file.lower() in ['quit', 'exit', 'q']:
                print("👋 종료합니다")
                break
            
            if not audio_file:
                # 기본 테스트 파일들
                test_files = [
                    "temp/20250826_025237_SPEAKER_00_010.wav",
                    "temp/20250826_025237_SPEAKER_01_008.wav",
                    "audio/test.wav"
                ]
                
                print(f"\n🧪 기본 테스트 파일들:")
                for i, file in enumerate(test_files, 1):
                    if os.path.exists(file):
                        print(f"{i}. {file}")
                        result = checker.check_voice(file)
                        print()
                
            else:
                result = checker.check_voice(audio_file)

if __name__ == "__main__":
    main()
