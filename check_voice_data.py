# -*- coding: utf-8 -*-
"""
check_voice_data.py

학습된 음성 데이터 확인 도구
- 저장된 화자 데이터 상세 정보 확인
- embeddings vs profile 파일 차이점 설명
- 학습 데이터 품질 분석
"""

import pickle
import json
import glob
import numpy as np
import os
from datetime import datetime

class VoiceDataChecker:
    """음성 데이터 확인 클래스"""
    
    def __init__(self, embeddings_dir="temp/embeddings"):
        self.embeddings_dir = embeddings_dir
        
    def check_all_data(self):
        """저장된 모든 음성 데이터 확인"""
        print("🎙️ 학습된 음성 데이터 확인")
        print("=" * 60)
        
        # 1. 파일 목록 확인
        self._show_file_list()
        
        # 2. 세션 메타데이터 확인
        self._show_session_metadata()
        
        # 3. 화자별 상세 정보
        self._show_speaker_details()
        
        # 4. embeddings vs profile 차이점 설명
        self._explain_file_differences()
    
    def _show_file_list(self):
        """저장된 파일 목록 표시"""
        print(f"\n📁 저장된 파일 목록 ({self.embeddings_dir}/)")
        print("-" * 40)
        
        all_files = glob.glob(f"{self.embeddings_dir}/*")
        
        if not all_files:
            print("❌ 저장된 파일이 없습니다.")
            return
        
        # 파일 타입별로 분류
        embeddings_files = []
        profile_files = []
        metadata_files = []
        
        for file in all_files:
            basename = os.path.basename(file)
            size_mb = os.path.getsize(file) / 1024 / 1024
            
            if "embeddings.pkl" in basename:
                embeddings_files.append((basename, size_mb))
            elif "profile.pkl" in basename:
                profile_files.append((basename, size_mb))
            elif "metadata.json" in basename:
                metadata_files.append((basename, size_mb))
        
        print(f"📊 임베딩 파일 ({len(embeddings_files)}개):")
        for name, size in embeddings_files:
            print(f"  🔹 {name} ({size:.2f}MB)")
        
        print(f"\n👤 프로파일 파일 ({len(profile_files)}개):")
        for name, size in profile_files:
            print(f"  🔸 {name} ({size:.2f}MB)")
        
        print(f"\n📋 메타데이터 파일 ({len(metadata_files)}개):")
        for name, size in metadata_files:
            print(f"  🔹 {name} ({size:.3f}MB)")
    
    def _show_session_metadata(self):
        """세션 메타데이터 표시"""
        print(f"\n📋 세션 메타데이터")
        print("-" * 40)
        
        metadata_files = glob.glob(f"{self.embeddings_dir}/*metadata.json")
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                print(f"📅 생성 시간: {metadata['timestamp']}")
                print(f"🎵 원본 오디오: {metadata['audio_file']}")
                print(f"👥 화자 수: {len(metadata['speakers'])}명")
                print(f"🔢 총 세그먼트: {metadata['total_segments']}개")
                print(f"📊 총 임베딩: {metadata['total_embeddings']}개")
                print(f"🔢 임베딩 차원: {metadata['embedding_dim']}")
                print(f"🎚️  샘플레이트: {metadata['sample_rate']}Hz")
                print(f"👤 화자 목록: {', '.join(metadata['speakers'])}")
                
            except Exception as e:
                print(f"❌ 메타데이터 로드 실패: {e}")
    
    def _show_speaker_details(self):
        """화자별 상세 정보 표시"""
        print(f"\n👥 화자별 상세 정보")
        print("=" * 60)
        
        profile_files = glob.glob(f"{self.embeddings_dir}/*_profile.pkl")
        
        for i, profile_file in enumerate(profile_files, 1):
            try:
                with open(profile_file, 'rb') as f:
                    profile = pickle.load(f)
                
                speaker_id = profile['speaker_id']
                
                print(f"\n🎯 화자 {i}: {speaker_id}")
                print("-" * 30)
                print(f"📊 세그먼트 수: {profile['num_segments']}개")
                print(f"⏱️  총 발화 시간: {profile['total_duration']:.2f}초")
                print(f"🔢 임베딩 차원: {profile['embedding_dim']}")
                print(f"🎵 원본 파일: {os.path.basename(profile['audio_file'])}")
                print(f"📅 생성 시간: {profile['timestamp']}")
                
                # 평균 임베딩 통계
                mean_emb = profile['mean_embedding']
                std_emb = profile['std_embedding']
                
                print(f"📈 평균 임베딩 범위: [{mean_emb.min():.4f}, {mean_emb.max():.4f}]")
                print(f"📊 표준편차 평균: {std_emb.mean():.4f}")
                
                # 샘플 임베딩 정보
                if 'sample_embeddings' in profile:
                    samples = profile['sample_embeddings']
                    print(f"🎯 샘플 데이터: {len(samples)}개")
                    
                    for j, sample in enumerate(samples[:3], 1):
                        print(f"  샘플 {j}: {sample['start_time']:.1f}s-{sample['end_time']:.1f}s")
                        print(f"          텍스트: '{sample['text'][:30]}{'...' if len(sample['text']) > 30 else ''}'")
                        print(f"          파일: {sample['filename']}")
                
                # 해당 화자의 embeddings 파일도 확인
                embeddings_file = profile_file.replace('_profile.pkl', '_embeddings.pkl')
                if os.path.exists(embeddings_file):
                    self._show_embeddings_details(embeddings_file, speaker_id)
                
            except Exception as e:
                print(f"❌ 프로파일 로드 실패 ({profile_file}): {e}")
    
    def _show_embeddings_details(self, embeddings_file, speaker_id):
        """개별 임베딩 파일 상세 정보"""
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            print(f"🔍 상세 임베딩 데이터:")
            print(f"   - 개별 임베딩 수: {len(embeddings_data)}개")
            
            # 각 임베딩의 길이와 텍스트 정보
            total_duration = sum([emb['duration'] for emb in embeddings_data])
            avg_duration = total_duration / len(embeddings_data)
            
            print(f"   - 평균 세그먼트 길이: {avg_duration:.2f}초")
            print(f"   - 가장 긴 세그먼트: {max([emb['duration'] for emb in embeddings_data]):.2f}초")
            print(f"   - 가장 짧은 세그먼트: {min([emb['duration'] for emb in embeddings_data]):.2f}초")
            
            # 텍스트가 있는 세그먼트 비율
            text_segments = [emb for emb in embeddings_data if emb['text'].strip()]
            text_ratio = len(text_segments) / len(embeddings_data) * 100
            print(f"   - 텍스트 포함 비율: {text_ratio:.1f}%")
            
        except Exception as e:
            print(f"   ❌ 임베딩 데이터 로드 실패: {e}")
    
    def _explain_file_differences(self):
        """embeddings vs profile 파일 차이점 설명"""
        print(f"\n🔍 파일 타입별 차이점 설명")
        print("=" * 60)
        
        print(f"📊 **embeddings.pkl 파일**:")
        print(f"   🎯 목적: 모든 개별 임베딩 벡터 저장")
        print(f"   📦 내용: 각 음성 세그먼트별 256차원 벡터 + 메타데이터")
        print(f"   📝 구조: List[Dict] 형태")
        print(f"        - embedding: numpy.ndarray (256차원)")
        print(f"        - start_time, end_time: float")
        print(f"        - duration: float")
        print(f"        - text: str (STT 결과)")
        print(f"        - filename: str")
        print(f"        - timestamp: str")
        print(f"   💾 크기: 크다 (모든 개별 데이터 포함)")
        print(f"   🎯 용도: 상세 분석, 데이터 탐색, 품질 확인")
        
        print(f"\n👤 **profile.pkl 파일**:")
        print(f"   🎯 목적: 화자의 대표 특징 저장 (요약본)")
        print(f"   📦 내용: 평균 임베딩 + 통계 + 메타데이터")
        print(f"   📝 구조: Dict 형태")
        print(f"        - mean_embedding: numpy.ndarray (256차원 평균)")
        print(f"        - std_embedding: numpy.ndarray (256차원 표준편차)")
        print(f"        - num_segments: int")
        print(f"        - total_duration: float")
        print(f"        - sample_embeddings: List (샘플 3개)")
        print(f"   💾 크기: 작다 (요약 데이터만)")
        print(f"   🎯 용도: **실제 화자 식별** (voice_checker.py에서 사용)")
        
        print(f"\n📋 **metadata.json 파일**:")
        print(f"   🎯 목적: 전체 세션 정보 저장")
        print(f"   📦 내용: 생성 시간, 파일 정보, 통계")
        print(f"   💾 크기: 매우 작다")
        print(f"   🎯 용도: 세션 관리, 디버깅")
        
        print(f"\n🔄 **실제 사용 흐름:**")
        print(f"   1️⃣ test.py 실행 → 3가지 파일 모두 생성")
        print(f"   2️⃣ voice_checker.py → profile.pkl만 사용 (빠른 식별)")
        print(f"   3️⃣ 데이터 분석 시 → embeddings.pkl 사용 (상세 분석)")
    
    def show_sample_data(self, speaker_id=None, num_samples=3):
        """실제 데이터 샘플 보기"""
        print(f"\n🎯 실제 데이터 샘플 보기")
        print("-" * 40)
        
        if speaker_id:
            pattern = f"{self.embeddings_dir}/*{speaker_id}_embeddings.pkl"
        else:
            pattern = f"{self.embeddings_dir}/*_embeddings.pkl"
        
        embeddings_files = glob.glob(pattern)
        
        for embeddings_file in embeddings_files[:1]:  # 첫 번째 파일만
            try:
                with open(embeddings_file, 'rb') as f:
                    embeddings_data = pickle.load(f)
                
                speaker = os.path.basename(embeddings_file).split('_')[2]
                
                print(f"\n👤 {speaker} 샘플 데이터:")
                
                for i, sample in enumerate(embeddings_data[:num_samples], 1):
                    print(f"\n  📊 샘플 {i}:")
                    print(f"     ⏱️  시간: {sample['start_time']:.1f}s - {sample['end_time']:.1f}s ({sample['duration']:.1f}초)")
                    print(f"     🎵 파일: {sample['filename']}")
                    print(f"     💬 텍스트: '{sample['text']}'")
                    print(f"     🔢 임베딩: 256차원 벡터 [{sample['embedding'][:3]}...]")
                    print(f"     📊 범위: [{sample['embedding'].min():.4f}, {sample['embedding'].max():.4f}]")
                
            except Exception as e:
                print(f"❌ 샘플 데이터 로드 실패: {e}")

def main():
    """메인 함수"""
    checker = VoiceDataChecker()
    
    # 전체 데이터 확인
    checker.check_all_data()
    
    # 샘플 데이터 보기
    checker.show_sample_data(num_samples=2)
    
    print(f"\n✅ 데이터 확인 완료!")

if __name__ == "__main__":
    main()
