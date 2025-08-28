"""
화자 관리 관련 서비스 클래스
화자 프로파일 관리, 이름 매핑, 화자 조회 등의 비즈니스 로직을 담당
DynamoDB를 사용하여 화자 이름 정보를 관리
"""
import os
import pickle
import glob
from typing import Dict, Any
from datetime import datetime

from app.models.speaker_model import SpeakerModel
from pynamodb.exceptions import DoesNotExist


class SpeakerService:
    """화자 관리 서비스 클래스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.embeddings_dir = "temp/embeddings"
    
    def get_display_name(self, speaker_id: str) -> str:
        """화자 ID에 대한 표시명 반환 (DynamoDB에서 조회하여 실제 이름이 있으면 실제 이름, 없으면 원래 ID)"""
        try:
            speaker = SpeakerModel.get(speaker_id)
            return speaker.speakerName
        except DoesNotExist:
            # DynamoDB에 없으면 speaker_id 반환
            return speaker_id
        except Exception as e:
            # datetime 파싱 오류나 기타 DynamoDB 오류 처리
            error_msg = str(e)
            if "Datetime string" in error_msg and "does not match format" in error_msg:
                print(f"❌ DynamoDB datetime 파싱 오류 ({speaker_id}): 잘못된 날짜 형식으로 저장된 데이터가 있습니다.")
                print(f"   오류 상세: {error_msg}")
                print(f"   해당 화자 데이터를 수동으로 확인하고 정리가 필요합니다.")
            else:
                print(f"❌ 화자 이름 조회 실패 ({speaker_id}): {e}")
            return speaker_id
    
    def get_all_speakers(self) -> Dict[str, Any]:
        """등록된 모든 화자 정보 조회 (프로파일 파일 + DynamoDB 이름 정보)"""
        try:
            existing_profiles = {}
            profile_files = glob.glob(f"{self.embeddings_dir}/*_profile.pkl")
            
            # DynamoDB에서 모든 화자 이름 정보 조회 (삭제되지 않은 화자만)
            name_mapping = {}
            try:
                for speaker in SpeakerModel.scan(SpeakerModel.deletedAt.does_not_exist()):
                    name_mapping[speaker.speakerId] = speaker.speakerName
            except Exception as e:
                error_msg = str(e)
                if "Datetime string" in error_msg and "does not match format" in error_msg:
                    print(f"❌ DynamoDB 스캔 중 datetime 파싱 오류: 잘못된 날짜 형식 데이터 존재")
                    print(f"   오류 상세: {error_msg}")
                else:
                    print(f"DynamoDB 스캔 실패: {e}")
            
            for profile_file in profile_files:
                try:
                    with open(profile_file, 'rb') as f:
                        profile = pickle.load(f)
                        speaker_id = profile['speaker_id']
                        display_name = name_mapping.get(speaker_id, speaker_id)
                        
                        existing_profiles[speaker_id] = {
                            'speaker_id': speaker_id,
                            'display_name': display_name,
                            'is_named': speaker_id in name_mapping and name_mapping[speaker_id] != speaker_id,
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
        """화자 이름 변경 (DynamoDB 업데이트)"""
        try:
            # 입력 검증
            if not new_name or not new_name.strip():
                return {
                    'success': False,
                    'error': '빈 이름은 설정할 수 없습니다.'
                }
            
            new_name = new_name.strip()

            print(f"🔍 화자 이름 변경 시도: {speaker_id} -> {new_name}")

            
            # 화자가 존재하는지 확인
            if not self._speaker_exists(speaker_id):
                return {
                    'success': False,
                    'error': f'화자 "{speaker_id}"를 찾을 수 없습니다.'
                }
            
            # 기존 이름 조회
            old_name = self.get_display_name(speaker_id)
            
            # DynamoDB에서 화자 정보 업데이트 (기존 레코드가 있을 때만)
            try:
                try:
                    # 기존 레코드가 있는지 확인하고 업데이트
                    speaker = SpeakerModel.get(speaker_id)
                    speaker.speakerName = new_name
                    speaker.updatedAt = datetime.utcnow()
                    speaker.save()
                    print(f"✅ 화자 이름 업데이트 완료: {speaker_id} -> {new_name}")
                    
                    return {
                        'success': True,
                        'message': f'화자 이름이 성공적으로 변경되었습니다.',
                        'speaker_id': speaker_id,
                        'old_name': old_name,
                        'new_name': new_name
                    }
                    
                except DoesNotExist:
                    # DynamoDB에 레코드가 없으면 오류 반환 (업데이트는 기존 데이터가 있을 때만 가능)
                    return {
                        'success': False,
                        'error': f'DynamoDB에 화자 "{speaker_id}" 정보가 없습니다. 먼저 화자를 등록해야 합니다.'
                    }
                    
            except Exception as e:
                error_msg = str(e)
                if "Datetime string" in error_msg and "does not match format" in error_msg:
                    error_description = f'DynamoDB datetime 파싱 오류: 잘못된 날짜 형식 데이터가 있습니다. 해당 화자 데이터를 수동으로 정리해야 합니다.'
                    print(f"❌ DynamoDB datetime 파싱 오류 ({speaker_id}): {error_msg}")
                else:
                    error_description = f'DynamoDB 연동 중 오류: {str(e)}'
                
                return {
                    'success': False,
                    'error': error_description
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_or_update_speaker_name(self, speaker_id: str, speaker_name: str = None) -> Dict[str, Any]:
        """화자 이름 생성 또는 업데이트 (음성 분석 과정에서 사용, 없으면 생성하고 있으면 기존 이름 유지)"""
        try:
            # DynamoDB에서 화자 정보 생성 또는 기존 정보 유지
            try:
                try:
                    # 기존 레코드가 있는지 확인
                    speaker = SpeakerModel.get(speaker_id)
                    # 기존 레코드가 있으면 이름을 변경하지 않고 그대로 유지
                    old_name = speaker.speakerName
                    current_name = speaker.speakerName
                    speaker.updatedAt = datetime.utcnow()
                    speaker.save()
                    print(f"✅ 기존 화자 정보 유지: {speaker_id} -> {current_name}")
                    action = "existing"
                    
                except DoesNotExist:
                    # 레코드가 없으면 새로 생성 (이때만 speaker_name 사용)
                    if not speaker_name:
                        speaker_name = speaker_id
                    else:
                        speaker_name = speaker_name.strip()
                        if not speaker_name:
                            speaker_name = speaker_id
                    
                    speaker = SpeakerModel(
                        id=speaker_id,
                        speakerId=speaker_id,
                        speakerName=speaker_name,
                        createdAt=datetime.utcnow(),
                        updatedAt=datetime.utcnow()
                    )
                    speaker.save()
                    print(f"✅ 새 화자 정보 생성 완료: {speaker_id} -> {speaker_name}")
                    old_name = speaker_id
                    current_name = speaker_name
                    action = "created"
                
                return {
                    'success': True,
                    'message': f'화자 정보가 성공적으로 {"유지" if action == "existing" else "생성"}되었습니다.',
                    'speaker_id': speaker_id,
                    'old_name': old_name,
                    'new_name': current_name,
                    'action': action
                }
                    
            except Exception as e:
                error_msg = str(e)
                if "Datetime string" in error_msg and "does not match format" in error_msg:
                    error_description = f'DynamoDB datetime 파싱 오류: 잘못된 날짜 형식 데이터가 있습니다. 해당 화자 데이터를 수동으로 정리해야 합니다.'
                    print(f"❌ DynamoDB datetime 파싱 오류 ({speaker_id}): {error_msg}")
                else:
                    error_description = f'DynamoDB 연동 중 오류: {str(e)}'
                
                return {
                    'success': False,
                    'error': error_description
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def reset_speaker_name(self, speaker_id: str) -> Dict[str, Any]:
        """화자 이름 초기화 (원래 ID로 되돌리기, DynamoDB에서 speaker_id와 동일하게 설정)"""
        try:
            # 기존 이름 조회
            old_name = self.get_display_name(speaker_id)
            
            # 이미 speaker_id와 동일하면 초기화할 필요 없음
            if old_name == speaker_id:
                return {
                    'success': False,
                    'error': f'화자 "{speaker_id}"는 이미 기본 이름으로 설정되어 있습니다.'
                }
            
            # DynamoDB에서 이름을 speaker_id로 업데이트 (초기화)
            try:
                try:
                    speaker = SpeakerModel.get(speaker_id)
                    speaker.speakerName = speaker_id
                    speaker.updatedAt = datetime.utcnow()
                    speaker.save()
                    print(f"✅ 화자 이름 초기화 완료: {speaker_id}")
                    
                    return {
                        'success': True,
                        'message': f'화자 이름이 초기화되었습니다.',
                        'speaker_id': speaker_id,
                        'old_name': old_name,
                        'current_name': speaker_id
                    }
                    
                except DoesNotExist:
                    # DynamoDB에 레코드가 없으면 오류 반환 (초기화도 기존 데이터가 있을 때만 가능)
                    return {
                        'success': False,
                        'error': f'DynamoDB에 화자 "{speaker_id}" 정보가 없습니다. 먼저 화자를 등록해야 합니다.'
                    }
                
            except Exception as e:
                error_msg = str(e)
                if "Datetime string" in error_msg and "does not match format" in error_msg:
                    error_description = f'DynamoDB datetime 파싱 오류: 잘못된 날짜 형식 데이터가 있습니다. 해당 화자 데이터를 수동으로 정리해야 합니다.'
                    print(f"❌ DynamoDB datetime 파싱 오류 ({speaker_id}): {error_msg}")
                else:
                    error_description = f'DynamoDB 연동 중 오류: {str(e)}'
                
                return {
                    'success': False,
                    'error': error_description
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_all_speaker_names(self) -> Dict[str, Any]:
        """모든 화자 이름 매핑 조회 (DynamoDB에서 조회)"""
        try:
            name_mapping = {}
            
            # DynamoDB에서 모든 화자 정보 조회 (삭제되지 않은 화자만)
            try:
                for speaker in SpeakerModel.scan(SpeakerModel.deletedAt.does_not_exist()):
                    name_mapping[speaker.speakerId] = speaker.speakerName
            except Exception as e:
                error_msg = str(e)
                if "Datetime string" in error_msg and "does not match format" in error_msg:
                    print(f"❌ DynamoDB 스캔 중 datetime 파싱 오류: 잘못된 날짜 형식 데이터 존재")
                    print(f"   오류 상세: {error_msg}")
                    error_description = f'DynamoDB datetime 파싱 오류: 잘못된 날짜 형식 데이터'
                else:
                    print(f"DynamoDB 스캔 실패: {e}")
                    error_description = f'DynamoDB 조회 실패: {str(e)}'
                
                return {
                    'success': False,
                    'error': error_description,
                    'name_mapping': {},
                    'total_named_speakers': 0
                }
            
            # 이름이 설정된 화자들만 필터링 (speaker_id와 다른 이름을 가진 화자들)
            named_speakers = {k: v for k, v in name_mapping.items() if v != k}
            
            return {
                'success': True,
                'name_mapping': name_mapping,
                'named_speakers': named_speakers,
                'total_speakers': len(name_mapping),
                'total_named_speakers': len(named_speakers)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'name_mapping': {},
                'total_named_speakers': 0
            }
    
    def _speaker_exists(self, speaker_id: str) -> bool:
        """화자가 존재하는지 확인하는 내부 메서드 (DynamoDB GSI를 사용하여 speakerId로 조회)"""
        try:
            # DynamoDB GSI를 사용하여 speakerId로 쿼리하여 화자 존재 여부 확인
            # 성능 최적화: limit=1로 첫 번째 레코드만 확인
            speakers = list(SpeakerModel.speakerId_index.query(
                speaker_id,  # 해시 키로 speakerId 사용
                filter_condition=SpeakerModel.deletedAt.does_not_exist(),  # 삭제되지 않은 화자만
                limit=1  # 성능 최적화: 존재 여부만 확인하므로 첫 번째 레코드만 가져옴
            ))
            
            exists = len(speakers) > 0
            print(f"🔍 DynamoDB GSI에서 화자 존재 확인 (speakerId: {speaker_id}): {'존재함' if exists else '존재하지 않음'}")
            return exists
            
        except Exception as e:
            print(f"❌ DynamoDB GSI 화자 존재 확인 실패 (speakerId: {speaker_id}): {e}")
            return False
        
    def delete_speaker_profile(self, speaker_id: str) -> Dict[str, Any]:
        """화자 프로파일 삭제 (프로파일 파일 + DynamoDB 레코드 삭제)"""
        try:
            profile_files = glob.glob(f"{self.embeddings_dir}/*_profile.pkl")
            deleted_files = []
            
            # 프로파일 파일 삭제
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
                # DynamoDB에서도 해당 화자 정보 삭제 (현재 DynamoDB 서비스에 삭제 메서드가 없으므로 로그만 출력)
                print(f"⚠️ DynamoDB에서 화자 {speaker_id} 삭제가 필요합니다. (현재 삭제 메서드 미구현)")
                
                return {
                    'success': True,
                    'message': f'화자 "{speaker_id}" 프로파일이 삭제되었습니다.',
                    'deleted_files': deleted_files,
                    'note': 'DynamoDB 레코드는 수동으로 삭제해야 합니다.'
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

    def get_next_available_speaker_id(self) -> str:
        """DynamoDB에서 다음 사용 가능한 화자 ID를 반환 (SPEAKER_XX 형식)"""
        try:
            # DynamoDB에서 모든 화자 ID 조회 (삭제되지 않은 화자만)
            used_ids = set()
            try:
                for speaker in SpeakerModel.scan(SpeakerModel.deletedAt.does_not_exist()):
                    speaker_id = speaker.speakerId
                    # SPEAKER_XX 패턴인지 확인
                    if speaker_id.startswith('SPEAKER_') and len(speaker_id) == 10:
                        used_ids.add(speaker_id)
                        
            except Exception as e:
                error_msg = str(e)
                if "Datetime string" in error_msg and "does not match format" in error_msg:
                    print(f"❌ DynamoDB 스캔 중 datetime 파싱 오류: 잘못된 날짜 형식 데이터 존재")
                    print(f"   오류 상세: {error_msg}")
                else:
                    print(f"DynamoDB 스캔 실패: {e}")
                # 오류 발생 시에도 계속 진행 (빈 set으로)
                used_ids = set()
            
            print(f"🔍 DynamoDB에서 기존 화자 ID 조회: {len(used_ids)}개 발견 - {sorted(used_ids)}")
            
            # 다음 사용 가능한 ID 찾기 (00부터 시작)
            counter = 0
            while True:
                new_id = f"SPEAKER_{counter:02d}"
                if new_id not in used_ids:
                    print(f"✅ 새로운 화자 ID 생성: {new_id}")
                    return new_id
                counter += 1
                
                # 안전장치: 99를 넘어가면 타임스탬프 기반으로 생성
                if counter > 99:
                    import time
                    fallback_id = f"SPEAKER_{int(time.time()) % 1000:03d}"
                    print(f"⚠️ SPEAKER_99 초과, 타임스탬프 기반 ID 생성: {fallback_id}")
                    return fallback_id
                
        except Exception as e:
            print(f"❌ 화자 ID 생성 중 오류: {e}")
            # 오류 발생 시 타임스탬프 기반 fallback
            import time
            fallback_id = f"SPEAKER_{int(time.time()) % 1000:03d}"
            print(f"🔄 Fallback ID 생성: {fallback_id}")
            return fallback_id
