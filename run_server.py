"""
오디오 처리 서버 실행 스크립트
Flask 애플리케이션을 실행하는 메인 스크립트
"""
import os
import traceback
from app import create_app


if __name__ == '__main__':
    try:
        print("오디오 처리 서버 시작 중...")
        
        # 환경 설정
        os.environ['FLASK_ENV'] = os.getenv('FLASK_ENV', 'development')
        
        # Flask 앱 생성
        app = create_app()
        
        # 전역 예외 처리
        @app.errorhandler(Exception)
        def handle_flask_exception(e):
            tb_text = traceback.format_exc()
            error_msg = f"[전역 예외 처리]\n{type(e).__name__}: {str(e)}\n\n[트레이스백]\n{tb_text}"
            print(error_msg)
            return {"error": error_msg, "success": False}
        
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except Exception as e:
        print(f"서버 시작 실패: {str(e)}")
        traceback.print_exc()