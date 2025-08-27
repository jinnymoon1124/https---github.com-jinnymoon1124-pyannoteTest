"""
화자 관리 관련 서비스 클래스
화자 프로파일 관리, 이름 매핑, 화자 조회 등의 비즈니스 로직을 담당
"""
import os
import json
import pickle
import glob
from typing import Dict, Any


class SpeakerService:
    """화자 관리 서비스 클래스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.embeddings_dir = "temp/embeddings"
        self.mapping_file = os.path.join(self.embeddings_dir, "speaker_name_mapping.json")
    
    def load_speaker_name_mapping(self) -> Dict[str, str]:
        """화자 이름 매핑 파일 로드"""
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"이름 매핑 파일 로드 실패: {e}")
                return {}
        return {}
    
    def save_speaker_name_mapping(self, mapping: Dict[str, str]) -> bool:
        """화자 이름 매핑 파일 저장"""
        try:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)
            
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"이름 매핑 파일 저장 실패: {e}")
            return False
    
    def get_display_name(self, speaker_id: str) -> str:
        """화자 ID에 대한 표시명 반환 (실제 이름이 있으면 실제 이름, 없으면 원래 ID)"""
        name_mapping = self.load_speaker_name_mapping()
        return name_mapping.get(speaker_id, speaker_id)
    
    def get_all_speakers(self) -> Dict[str, Any]:
        """등록된 모든 화자 정보 조회"""
        try:
            existing_profiles = {}
            profile_files = glob.glob(f"{self.embeddings_dir}/*_profile.pkl")
            name_mapping = self.load_speaker_name_mapping()
            
            for profile_file in profile_files:
                try:
                    with open(profile_file, 'rb') as f:
                        profile = pickle.load(f)
                        speaker_id = profile['speaker_id']
                        display_name = name_mapping.get(speaker_id, speaker_id)
                        
                        existing_profiles[speaker_id] = {
                            'speaker_id': speaker_id,
                            'display_name': display_name,
                            'is_named': speaker_id in name_mapping,
                            'num_segments': profile['num_segments'],
                            'total_duration': profile['total_duration'],
                            'timestamp': profile['timestamp'],
                            'embedding_dim': profile['embedding_dim']
                        }
                except Exception as e:
                    print(f"프로파일 파일 읽기 실패 {profile_file}: {e}")
                    continue
            
            return {
                'success': True,
                'speakers': existing_profiles,
                'total_speakers': len(existing_profiles)
            }
            
        except Exception as e:
            print(f"화자 목록 조회 중 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'speakers': {},
                'total_speakers': 0
            }
    
    def update_speaker_name(self, speaker_id: str, new_name: str) -> Dict[str, Any]:
        """화자 이름 변경"""
        try:
            # 입력 검증
            if not new_name or not new_name.strip():
                return {
                    'success': False,
                    'error': '빈 이름은 설정할 수 없습니다.'
                }
            
            new_name = new_name.strip()
            
            # 화자가 존재하는지 확인
            if not self._speaker_exists(speaker_id):
                return {
                    'success': False,
                    'error': f'화자 "{speaker_id}"를 찾을 수 없습니다.'
                }
            
            # 이름 매핑 업데이트
            name_mapping = self.load_speaker_name_mapping()
            old_name = name_mapping.get(speaker_id, speaker_id)
            name_mapping[speaker_id] = new_name
            
            if self.save_speaker_name_mapping(name_mapping):
                return {
                    'success': True,
                    'message': f'화자 이름이 성공적으로 변경되었습니다.',
                    'speaker_id': speaker_id,
                    'old_name': old_name,
                    'new_name': new_name
                }
            else:
                return {
                    'success': False,
                    'error': '이름 매핑 저장에 실패했습니다.'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def reset_speaker_name(self, speaker_id: str) -> Dict[str, Any]:
        """화자 이름 초기화 (원래 ID로 되돌리기)"""
        try:
            name_mapping = self.load_speaker_name_mapping()
            
            if speaker_id not in name_mapping:
                return {
                    'success': False,
                    'error': f'화자 "{speaker_id}"에 설정된 이름이 없습니다.'
                }
            
            old_name = name_mapping[speaker_id]
            del name_mapping[speaker_id]
            
            if self.save_speaker_name_mapping(name_mapping):
                return {
                    'success': True,
                    'message': f'화자 이름이 초기화되었습니다.',
                    'speaker_id': speaker_id,
                    'old_name': old_name,
                    'current_name': speaker_id
                }
            else:
                return {
                    'success': False,
                    'error': '이름 매핑 저장에 실패했습니다.'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_all_speaker_names(self) -> Dict[str, Any]:
        """모든 화자 이름 매핑 조회"""
        try:
            name_mapping = self.load_speaker_name_mapping()
            
            return {
                'success': True,
                'name_mapping': name_mapping,
                'total_named_speakers': len(name_mapping)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'name_mapping': {},
                'total_named_speakers': 0
            }
    
    def _speaker_exists(self, speaker_id: str) -> bool:
        """화자가 존재하는지 확인하는 내부 메서드"""
        profile_files = glob.glob(f"{self.embeddings_dir}/*_profile.pkl")
        
        for profile_file in profile_files:
            try:
                with open(profile_file, 'rb') as f:
                    profile = pickle.load(f)
                    if profile['speaker_id'] == speaker_id:
                        return True
            except Exception as e:
                continue
        
        return False
    
    def delete_speaker_profile(self, speaker_id: str) -> Dict[str, Any]:
        """화자 프로파일 삭제 (필요시 사용)"""
        try:
            profile_files = glob.glob(f"{self.embeddings_dir}/*_profile.pkl")
            deleted_files = []
            
            for profile_file in profile_files:
                try:
                    with open(profile_file, 'rb') as f:
                        profile = pickle.load(f)
                        if profile['speaker_id'] == speaker_id:
                            os.remove(profile_file)
                            deleted_files.append(profile_file)
                except Exception as e:
                    continue
            
            if deleted_files:
                # 이름 매핑에서도 제거
                name_mapping = self.load_speaker_name_mapping()
                if speaker_id in name_mapping:
                    del name_mapping[speaker_id]
                    self.save_speaker_name_mapping(name_mapping)
                
                return {
                    'success': True,
                    'message': f'화자 "{speaker_id}" 프로파일이 삭제되었습니다.',
                    'deleted_files': deleted_files
                }
            else:
                return {
                    'success': False,
                    'error': f'화자 "{speaker_id}"의 프로파일을 찾을 수 없습니다.'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
