"""
기본 컨트롤러 클래스
모든 컨트롤러가 상속받는 공통 기능을 제공
"""
from functools import wraps
from flask import request, jsonify
from typing import Dict, Any, Callable


class BaseController:
    """모든 컨트롤러가 상속받는 기본 컨트롤러 클래스"""
    
    def __init__(self):
        """기본 컨트롤러 초기화"""
        pass
    
    @staticmethod
    def with_request_data(func: Callable) -> Callable:
        """
        요청 데이터를 파싱하여 메서드에 전달하는 데코레이터
        JSON과 form-data 모두 처리 가능
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                # JSON 데이터 처리
                if request.is_json:
                    data = request.get_json() or {}
                else:
                    # Form 데이터 처리
                    data = request.form.to_dict()
                
                # URL 파라미터 추가
                data.update(request.args.to_dict())
                
                return func(self, data, *args, **kwargs)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'요청 데이터 처리 중 오류: {str(e)}'
                }), 400
        
        return wrapper
    
    @staticmethod
    def handle_exceptions(func: Callable) -> Callable:
        """
        예외를 처리하는 데코레이터
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'message': '서버 처리 중 오류가 발생했습니다.'
                }), 500
        
        return wrapper
    
    def success_response(self, data: Any = None, message: str = "성공") -> Dict[str, Any]:
        """성공 응답 생성"""
        response = {
            'success': True,
            'message': message
        }
        if data is not None:
            response['data'] = data
        return response
    
    def error_response(self, error: str, status_code: int = 400) -> tuple:
        """오류 응답 생성"""
        return jsonify({
            'success': False,
            'error': error
        }), status_code
    
    def invalid_param_response(self, param_name: str) -> tuple:
        """잘못된 파라미터 응답 생성"""
        return self.error_response(f'{param_name} 파라미터가 필요합니다.', 400)
    
    def not_found_response(self, resource: str) -> tuple:
        """리소스를 찾을 수 없음 응답 생성"""
        return self.error_response(f'{resource}를 찾을 수 없습니다.', 404)
