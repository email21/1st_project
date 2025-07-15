import requests
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv('../config/.env')

class SolarAPIClient:
  
    def __init__(self, api_key: Optional[str] = None):
        self.api_key: str = api_key or os.getenv("SOLAR_API_KEY", "")
        self.base_url: str = "https://api.upstage.ai/v1/solar"
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',            
        )
        self.logger: logging.Logger = logging.getLogger(__name__)
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """1단계: 쿼리 분석 및 의도 파악"""
        
        system_prompt = """
        당신은 금융상품 전문가입니다. 사용자의 질문을 분석하여 다음 정보를 JSON 형태로 추출해주세요:

        1. product_category: 상품 카테고리 (deposit, saving, annuity, mortgage, rent, credit 중 하나)
        2. user_intent: 사용자 의도 (목돈마련, 노후준비, 주택구매, 생활자금 등)
        3. key_requirements: 주요 요구사항 리스트 (기간, 금액, 안정성, 수익성 등)
        4. search_keywords: 검색 키워드 리스트
        5. user_profile: 사용자 프로필 정보 (나이대, 직업, 소득 등이 언급된 경우)

        응답은 반드시 JSON 형태로만 해주세요.
        """
        
        payload = {
            "model": "solar-1-mini-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.3, # 낮은 창의성
            "max_tokens": 1000  # 적은 토큰 수
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # JSON 파싱 시도
            try:
                analysis_result = json.loads(content)
                self.logger.info("✅ 쿼리 분석 완료  query analysis completed")
                return analysis_result
            except json.JSONDecodeError:
                self.logger.warning("⚠️ JSON 파싱 실패, 기본값 반환  JSON parsing failed, returning default value")
                return {
                    "product_category": "deposit",
                    "user_intent": "일반상담",
                    "key_requirements": [],
                    "search_keywords": [user_query],
                    "user_profile": {}
                }
                
        except Exception as e:
            self.logger.error(f"❌ 쿼리 분석 실패  Query analysis failed : {e}")
            return {
                "product_category": "deposit",
                "user_intent": "일반상담", 
                "key_requirements": [],
                "search_keywords": [user_query],
                "user_profile": {}
            }
    
    def generate_recommendation(self, user_query: str, analysis_result: Dict[str, Any], 
                              search_results: List[Dict]) -> str:
        """2단계: 상품 추천 및 설명 생성"""
        
        # 검색 결과를 텍스트로 변환
        products_text = ""
        for i, product in enumerate(search_results[:3], 1):
            products_text += f"\n상품 {i}:\n"
            products_text += f"- 상품명: {product.get('product_name', 'N/A')}\n"
            products_text += f"- 금융회사: {product.get('company', 'N/A')}\n"
            products_text += f"- 상품 정보: {product.get('search_text', 'N/A')[:200]}...\n"
            products_text += f"- 유사도: {product.get('similarity_score', 0):.3f}\n"
        
        system_prompt = f"""
        당신은 친근하고 전문적인 금융상품 추천 전문가입니다.
        
        사용자 분석 결과:
        - 상품 카테고리: {analysis_result.get('product_category', 'N/A')}
        - 사용자 의도: {analysis_result.get('user_intent', 'N/A')}
        - 주요 요구사항: {analysis_result.get('key_requirements', [])}
        - 사용자 프로필: {analysis_result.get('user_profile', {})}
        
        검색된 상품들:
        {products_text}
        
        다음 형식으로 추천해주세요:
        1. 상황 분석 (사용자의 니즈 요약)
        2. 추천 상품 (상위 1-2개 상품 상세 설명)
        3. 추천 이유 (왜 이 상품이 적합한지)
        4. 주의사항 (고려해야 할 점들)
        5. 다음 단계 (실제 가입 시 확인사항)
        
        - 어려운 금융 용어는 쉽게 풀어서 설명해주세요.
        - 가상상품 추천하지 말고 실제 운용되고 있는 상품으로 추천해주세요.
        - 상품 추천해줄 때 시중에 운용되고 있는 상품의 최저금리, 최고금리, 평균 정보도 함께 제공해주세요.
        """
        
        payload = {
            "model": "solar-1-mini-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.7, # 높은 창의성
            "max_tokens": 2000  # 많은 토큰 수, 상세한 설명
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            recommendation = result['choices'][0]['message']['content']
            
            self.logger.info("✅ 추천 생성 완료   Recommendation generation completed")
            return recommendation
            
        except Exception as e:
            self.logger.error(f"❌ 추천 생성 실패  Recommendation generation failed : {e}")
            return "죄송합니다. 추천을 생성하는 중 오류가 발생했습니다. 다시 시도해주세요."
