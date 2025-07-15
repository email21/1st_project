# 프로젝트명
내 상황에 딱 맞는 금융상품, AI가 답하다.

## 📋 프로젝트 소개
이 프로젝트는 카카오뱅크의 **KF-DeBERTa** 모델을 사용하여 한국어 금융 상품 데이터를 벡터화하고, 유사도 검색을 통해 관련 상품을 찾는 시스템입니다.

**도메인 특화 임베딩**
* KF-DeBERTa 사용: 금융 도메인에 특화된 임베딩 모델
* 768차원 벡터: 금융 상품의 미묘한 차이점 포착 가능
* 한국어 최적화: 한국 금융 용어와 표현에 최적화

### 주요 기능
- **금융 도메인 특화**: KF-DeBERTa 모델의 금융 도메인 특화 성능 활용
- **다양한 상품 지원**: 예금, 적금, 연금, 대출, 신용카드 등 다양한 금융 상품 처리

### 📊 단계별 흐름도
graph TD
    A[사용자 질문 입력<br/>app.py] --> B[질문 분석<br/>solar_client.py]
    B --> C[관련 상품 검색<br/>vector_store.py]
    C --> D[AI 답변 생성<br/>solar_client.py]
    D --> E[결과 화면 표시<br/>app.py]

### 📁 프로젝트 구조
financial_rag/
├── app/                    
│   └── app.py                      
├── config/
│   ├── .env
│   └── requirements.txt              # pip install -r config/requirements.txt
├── core/
│   ├── recommendation_system.py      # 전체 과정 관리: 질문 받기 → 검색 → AI 답변 → 결과 반환
│   ├── solar_client.py               # 2가지 작업: 질문 분석 + 추천 답변 생성
│   └── vector_store.py               # 벡터 데이터베이스 (검색 엔진)
├── preprocessing/              
│   ├── data_collector.py
│   ├── data_preprocessor.py
│   └── run_preprocessing.py
├── data/
│   ├── processed/
│   ├── raw/
│   └── vector_store/  
├── logs/
└── .gitignore               

### 🤖 질문분석과 추천답변 모두 solar-1-mini-chat을 사용한 이유
**일관성 유지**
- 동일한 한국어 처리 방식: 질문 분석과 답변 생성에서 동일한 언어 모델 사용
- 컨텍스트 연결성: 분석 결과와 답변 생성 간의 자연스러운 연결
- 응답 스타일 통일: 전체 시스템에서 일관된 톤앤매너 유지
