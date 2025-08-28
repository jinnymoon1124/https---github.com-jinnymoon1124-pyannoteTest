"""
오디오 처리 애플리케이션
Flask 앱 팩토리 패턴을 사용한 애플리케이션 초기화
"""
import os
from flask import Flask
from flask_cors import CORS


def create_app():
    """Flask 애플리케이션 팩토리"""
    app = Flask(__name__)
    
    # CORS 설정
    CORS(app)
    
    # 기본 설정
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 제한
    app.config['UPLOAD_FOLDER'] = 'temp/uploads'
    
    # Blueprint 등록
    from app.controllers.audio_controller import audio_bp
    from app.controllers.speaker_controller import speaker_bp
    from app.controllers.s3_controller import s3_bp
    
    app.register_blueprint(audio_bp)
    app.register_blueprint(speaker_bp)
    app.register_blueprint(s3_bp)
    
    # 기본 라우트 추가
    @app.route('/')
    def index():
        return {
            "message": "오디오 처리 서버가 정상 실행 중입니다",
            "success": True
        }
    
    return app
