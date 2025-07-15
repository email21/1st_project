import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from .vector_store import VectorStore
from .solar_client import SolarAPIClient

class FinancialRecommendationSystem:
    
    def __init__(self):
        self.vector_store: VectorStore = VectorStore()
        self.solar_client: SolarAPIClient = SolarAPIClient()
        self._setup_logging()
        self._initialize_vector_store()
    
    def _setup_logging(self) -> None:      
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger: logging.Logger = logging.getLogger(__name__)
    
    def _initialize_vector_store(self) -> None:
        self.logger.info("🔧 벡터 스토어 초기화 시작 Start initializing vector store")
        
        if not self.vector_store.load_index():
            self.logger.info("📁 기존 인덱스 없음, 새로 구축합니다  No existing index, build a new one")
            self.vector_store.load_processed_data()
            self.vector_store.build_index()
            self.vector_store.save_index()
        
        self.logger.info("✅ 벡터 스토어 초기화 완료 Vector store initialization complete")
    
    def get_recommendation(self, user_query: str) -> Dict[str, Any]:
        self.logger.info(f"🔍 추천 요청 Request for recommendation : {user_query[:50]}...")
        
        try:
            self.logger.info("1️⃣  쿼리 분석 시작 Query analysis started")
            analysis_result = self.solar_client.analyze_query(user_query)
            self.logger.info("2️⃣  상품 검색 시작 Product search started")
            product_category = analysis_result.get('product_category')
            search_keywords = analysis_result.get('search_keywords', [user_query])
            search_query = " ".join(search_keywords)
            search_results = self.vector_store.search(
                query=search_query,
                top_k=5,
                product_type=product_category if product_category != 'general' else None
            )          
            self.logger.info("3️⃣  추천 생성 시작  Recommendation generation started")
            recommendation = self.solar_client.generate_recommendation(
                user_query=user_query,
                analysis_result=analysis_result,
                search_results=search_results
            )            
            result = {
                'user_query': user_query,
                'analysis': analysis_result,
                'search_results': search_results,
                'recommendation': recommendation,
                'status': 'success'
            }            
            self.logger.info("✅ 추천 완료 Recommendation Completed")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 추천 실패 Recommendation failed : {e}")
            return {
                'user_query': user_query,
                'analysis': {},
                'search_results': [],
                'recommendation': "죄송합니다. 추천을 생성하는 중 오류가 발생했습니다.",
                'status': 'error',
                'error': str(e)
            }
