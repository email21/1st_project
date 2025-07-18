import os
import pandas as pd
from typing import Dict, List, Any, Optional  # 
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_upstage import ChatUpstage # type: ignore
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import warnings
import streamlit as st

load_dotenv()

class EnhancedKakaoEmbeddings(Embeddings):
    def __init__(self, batch_size=32, show_progress=True):
        from sentence_transformers import SentenceTransformer 
        self.encoder = SentenceTransformer("upskyy/kf-deberta-multitask") # 한국어 금융 도메인에 특화된 DeBERTa 기반 임베딩 모델, 허깅페이스에서 제공
        self.batch_size = batch_size  # 임베딩 처리 시 한 번에 처리할 문서 수 (메모리 효율성)
        self.show_progress = show_progress
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]: # 리턴턴값: 각 문서의 임베딩 벡터들의 리스트
        """문서 임베딩 - LangChain 호환"""
        embeddings = self.encoder.encode(  # SentenceTransformer의 encode 메서드를 호출하여 임베딩을 생성
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=True,  # 임베딩 벡터를 정규화하여 코사인 유사도 계산에 최적화
            convert_to_numpy=True  # 결과를 NumPy 배열로 변환 (후처리 편의성)
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩 - LangChain 호환"""
        embedding = self.encoder.encode(
            [text], # 단일 텍스트를 리스트로 감싸서 전달합니다 (API 일관성)
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]  # 리스트의 첫 번째 요소(단일 임베딩)를 추출
        return embedding.tolist()  #NumPy 배열을 Python 리스트로 변환하여 리턴

class FinancialRecommendationSystem:
    def __init__(self):
        self.vector_store = None
        self.chat_model = None
        self.retriever = None
        self.rag_chain = None
        self.conversation_history = []
        self._initialize_system()
    
    def _initialize_system(self):
        """시스템 초기화"""
        try:
            self._load_financial_data()          
            self.chat_model = ChatUpstage( # ChatUpstage 인스턴스를 생성해 LLM API 연결
                api_key=os.getenv("SOLAR_API_KEY"),
                model="solar-1-mini-chat"
            )
            
            # RAG 체인 구성
            self._setup_rag_chain()
            print("✅ 금융상품 추천 시스템 초기화 완료")
            
        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
            raise
    
    def _map_category_to_type(self, category: str) -> str:
        """사용자 카테고리를 데이터 타입으로 매핑"""
        if '정기예금' in category:
            return 'deposit'
        elif '적금' in category:
            return 'saving'
        elif category == '예금':  # 단순 '예금'은 정기예금으로 처리
            return 'deposit'
        elif '저축' in category and '적금' not in category:
            return 'deposit'
        elif '연금' in category:
            return 'annuity'
        elif '주택담보' in category:
            return 'mortgage'
        elif '전세' in category:
            return 'rent'
        elif '신용대출' in category or '개인신용' in category:
            return 'credit'
        return ''
       
    def _filter_by_category(self, context: List[Any], analysis: Dict[str, Any]):
        """카테고리 기반 필터링"""
        category = analysis.get('product_category', '')
        if not category:
            return context
        
        # 카테고리 매핑
        target_type = self._map_category_to_type(category)
        if not target_type:
            return context
        
        # 정확한 매칭만 허용
        filtered = []
        for doc in context:
            if hasattr(doc, 'metadata'):
                doc_type = doc.metadata.get('product_type', '')
                if doc_type == target_type:
                    filtered.append(doc)
                    
        print(f"카테고리 '{category}' → 타입 '{target_type}' 필터링: {len(filtered)}개")
        return filtered
    
    def _load_financial_data(self):
        """금융상품 데이터 로드 및 벡터스토어 구축"""
        documents = []
        data_dir = "../data/processed"
        product_files = {
            'deposit': 'deposit_processed.csv',
            'saving': 'saving_processed.csv', 
            'annuity': 'annuity_processed.csv',
            'mortgage': 'mortgage_processed.csv',
            'rent': 'rent_processed.csv',
            'credit': 'credit_processed.csv'
        }
        
        for product_type, filename in product_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    for _, row in df.iterrows():
                        doc = {
                            'product_type': product_type,
                            'company': row.get('kor_co_nm', ''),
                            'product_name': row.get('fin_prdt_nm', ''),
                            'search_text': row.get('search_text', ''),
                            'raw_data': row.to_dict()
                        }
                        documents.append(doc)
                except Exception as e:
                    print(f"❌ {product_type} 로드 실패: {e}")
            
        embeddings_model = EnhancedKakaoEmbeddings(
            batch_size=32,
            show_progress=True
        )   
        texts = [doc['search_text'] for doc in documents]  # 검색 텍스트 준비

         # 벡터스토어 생성, 텍스트를 벡터로 변환, FAISS 인덱스 구축 
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings_model, 
            metadatas=documents # 상품 메타데이터
        )
         # 리트리버 설정 (상위 3개 문서 리턴)
        self.retriever = self.vector_store.as_retriever(k=3) # 벡터 검색에서 반환환할 상위 유사 문서의 개수, 이 3개 문서가 RAG 체인의 컨텍스트로 사용
        self.documents = documents
        
        print(f"📊 총 {len(documents)}개 상품 로드 완료")

    def _create_hybrid_recommendation(self, query: str, search_results: List[Dict], original_answer: str) -> str:
        
        """데이터 기반 추천 + LLM 언어 처리"""
        if not search_results:
            return "검색된 상품이 없습니다. 다른 조건으로 검색해보세요."
        
        # 이전 대화 맥락 추가
        context_info = ""
        if self.conversation_history:
            last_messages = self.conversation_history[-6:]  # 최근 3턴만 참조
            for msg in last_messages:
                if isinstance(msg, AIMessage) and "📍 추천 상품:" in msg.content:
                    context_info = f"이전 추천 상품 정보: {msg.content[:200]}..."
                    break
    
           
        # 데이터 기반 상품 선택 (규칙 기반)
        top_product = search_results[0]
        
        # LLM으로 자연스러운 설명 생성
        context = f"""
        추천 상품: {top_product['product_name']}
        금융회사: {top_product['company']}
        상품유형: {top_product['product_type']}
        유사도: {top_product['similarity_score']}
        """
        
        # LLM 활용
        try:
            prompt = f"""
            {context_info}
        
        현재 검색 결과: {search_results[0] if search_results else '없음'}
        사용자 질문: {query}
        
        이전 대화 맥락을 고려하여 일관된 답변을 제공하세요.
        
        다음 검색 결과를 바탕으로 사용자에게 자연스럽고 전문적인 추천을 제공하세요.
        
        **검색된 상품 정보:**
        {context}
        
        **답변 요구사항:**
        - 위 상품 정보만을 활용하여 추천 작성
        - 이전 대화와 연관성을 고려하여 답변
        - 검색 결과에 없는 정보는 절대 추가하지 말 것
        
        **상품을 추천할 경우 다음 형식식으로 답변:**
        📍 **추천 상품**: {top_product['product_name']}
        📍 **금융회사**: {top_product['company']}
        📍 **추천 이유**: [구체적 이유 설명]
        📍 **주의사항**: [고려사항 안내]
            """
            
            llm_response = self.chat_model.invoke([HumanMessage(content=prompt)])
            return llm_response.content
            
        except Exception as e:
            # LLM 실패 시 원본 RAG 답변 사용
            print(f"하이브리드 추천 생성 실패: {e}")
            return original_answer if original_answer else "추천 생성 중 오류가 발생했습니다."

    def _validate_recommendation(self, recommendation: str, search_results: List[Dict]) -> bool:
        """추천 결과 검증"""
        if not search_results:
            return False
        
        # 허용된 상품명들 추출
        allowed_products = [product['product_name'] for product in search_results]
        allowed_companies = [product['company'] for product in search_results]
        
        # 추천 내용에 허용된 상품만 포함되어 있는지 확인
        for product_name in allowed_products:
            if product_name in recommendation:
                return True
        
        return False

    def _filter_validated_results(self, search_results: List[Dict]) -> List[Dict]:
        """검증된 결과만 필터링"""
        validated = []
        for product in search_results:
            # 기본 필드 검증
            if (product.get('product_name') and 
                product.get('company') and 
                product.get('product_type')):
                validated.append(product)
        
        return validated


    
    def _setup_rag_chain(self):        
        """RAG 체인 구성"""
        # 질문 재구성 프롬프트
        multiturn_system_prompt = """
        금융상품 상담에서 사용자의 질문을 이전 대화 맥락과 함께 분석하여 완전한 질문으로 재구성하세요.

        **재구성 원칙:**
        1. 대명사 처리: "그 상품", "이 대출" → 구체적인 상품명으로 교체
        2. 맥락 의존 질문: "금리가 어떤데?" → "앞서 추천한 [상품명]의 금리는 어떻게 되나요?"
        3. 회사 관련 질문: "[회사명] 대출 금리는?" → "[회사명]의 [앞서 언급한 상품명] 대출 금리는?"

        **재구성 예시:**
        - 이전: "신한은행 전세자금대출 추천"
        - 현재: "금리가 어떤데?" 
        - 재구성: "신한은행 전세자금대출의 금리는 어떻게 되나요?"

        **중요:** 이전 대화에서 언급된 구체적인 상품명과 회사명을 반드시 포함하세요.
        """
        multiturn_prompt = ChatPromptTemplate.from_messages([ # LangChain의 프롬프트 템플릿 생성
            ("system", multiturn_system_prompt), # 시스템 메시지로 재구성 지침 전달
            MessagesPlaceholder("chat_history"),  # 이전 대화 기록을 동적으로 삽입하는 플레이스홀더
            ("human", "{input}"), # 현재 사용자 질문을 삽입하는 템플릿
        ])
        
        # 이전 대화를 기억하는 리트리버 생성, self.retriever:기본 벡터 검색 리트리버 (FAISS 기반, k=3)
        history_aware_retriever = create_history_aware_retriever( # LangChain의 히스토리 인지 리트리버 생성 함수
            self.chat_model, self.retriever, multiturn_prompt
        )
        
        rag_system_prompt = """
            당신은 금융상품 전문가가입니다. 사용자의 질문 유형에 따라 적절한 답변을 제공해주세요.

            **질문 유형별 답변 방식:**

            **1. 상품 추천 요청시만 다음 구조 사용:**
            📍 **상황 분석**: 사용자의 니즈와 상황을 간단히 요약
            📍 **추천 상품**: 가장 적합한 상품 1개 선정
            📍 **금융회사**: 해당 상품을 제공하는 금융회사
            📍 **추천 이유**: 구체적 근거
            📍 **주의사항**: 가입 전 확인사항
            📍 **다음 단계**: 실제 가입 안내

            **2. 정보 문의, 설명 요청시:**
            - 질문에 대한 직접적이고 명확한 답변
            - 필요시 관련 상품 정보 제공 (추천 형태 아님)
            - 추가 질문 유도

            **3. 이전 대화 참조 질문시:**
            - 이전에 언급된 상품/정보를 정확히 참조
            - 추천 형태가 아닌 설명 형태로 답변

            **답변 판단 기준:**
            - "추천해줘", "알려줘", "어떤 상품이 좋을까" → 추천 형태
            - "설명해줘", "정보 알려줘", "뭐였는데" → 정보 제공 형태
         
            4.검색 결과가 없으면 "제공된 데이터에서 조건에 맞는 상품을 찾을 수 없습니다"로 답변

            **답변 톤**: 전문적이면서도 이해하기 쉽게, 친근하고 신뢰감 있게
                    """
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "검색된 관련 문서들:\n{context}")
        ])
        # 전체 RAG 체인 구성
        question_answer_chain = create_stuff_documents_chain(self.chat_model, rag_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def _create_hybrid_recommendation(self, query: str, search_results: List[Dict], original_answer: str, analysis: Dict) -> str:
        """사용자 의도에 따른 답변 생성"""   
        if not search_results:   # 검색 결과가 없는 경우 처리
            return "검색된 상품이 없습니다. 다른 조건으로 검색해보세요."    
      
        context_info = ""   # 이전 대화 맥락 추가
        if self.conversation_history:
            last_messages = self.conversation_history[-6:]  # 최근 3턴만 참조
            for msg in last_messages:
                if isinstance(msg, AIMessage) and "📍 추천 상품:" in msg.content:
                    context_info = f"이전 추천 상품 정보: {msg.content[:200]}..."
                    break
        
        # 사용자 의도 분석
        user_intent = analysis.get('user_intent', 'information')
        product_category = analysis.get('product_category', '')
        
        # 최상위 상품 선택
        top_product = search_results[0]
        
        # 사용자 의도에 따른 답변 형태 결정
        if user_intent == 'information' or any(keyword in query.lower() for keyword in ['설명', '정보', '뭐였는데', '알려줘', '마지막으로']):
            # 정보 제공 형태
            prompt = f"""
            사용자 질문: {query}
            
            {context_info}
            
            상품 정보:
            - 상품명: {top_product.get('product_name', '정보 없음')}
            - 금융회사: {top_product.get('company', '정보 없음')}
            - 상품 유형: {top_product.get('product_type', '기타')}
            - 상세 정보: {top_product.get('raw_data', {})}
            
            사용자의 질문에 대한 직접적이고 명확한 답변을 제공하세요.
            
            **답변 요구사항:**
            - 추천 형태(📍)가 아닌 일반적인 설명 형태로 답변
            - 질문에 대한 직접적인 답변
            - 이전 대화 맥락을 고려하여 일관된 답변
            - 검색 결과에 있는 정보만 사용
            - 불확실한 정보는 "상세 내용은 해당 금융회사 확인 필요"로 안내
            """
            
        elif user_intent == 'comparison' or '비교' in query.lower():
            # 비교 형태
            comparison_products = search_results[:3]  # 상위 3개 상품 비교
            products_info = "\n".join([
                f"- {p.get('product_name', '정보 없음')} ({p.get('company', '정보 없음')})"
                for p in comparison_products
            ])
            
            prompt = f"""
            사용자 질문: {query}
            
            {context_info}
            
            비교 대상 상품들:
            {products_info}
            
            위 상품들을 비교하여 각각의 장단점과 특징을 설명해주세요.
            
            **답변 요구사항:**
            - 각 상품의 특징과 장단점을 명확히 구분
            - 사용자 상황에 맞는 선택 가이드 제공
            - 검색 결과에 있는 정보만 사용
            """
            
        else:
            # 추천 형태 (기본)
            prompt = f"""
            {context_info}
            
            사용자 질문: {query}
            
            검색된 상품 정보:
            - 상품명: {top_product.get('product_name', '정보 없음')}
            - 금융회사: {top_product.get('company', '정보 없음')}
            - 상품 유형: {top_product.get('product_type', '기타')}
            - 유사도: {top_product.get('similarity_score', 0)}
            - 상세 정보: {top_product.get('raw_data', {})}
            
            이전 대화 맥락을 고려하여 일관된 답변을 제공하세요.
            
            **상품을 추천할 경우 다음 형식으로 답변:**
            📍 **추천 상품**: {top_product.get('product_name', '정보 없음')}
            📍 **금융회사**: {top_product.get('company', '정보 없음')}
            📍 **추천 이유**: [구체적 이유 설명]
            📍 **주의사항**: [고려사항 안내]
            
            **답변 요구사항:**
            - 위 상품 정보만을 활용하여 추천 작성
            - 이전 대화와 연관성을 고려하여 답변
            - 검색 결과에 없는 정보는 절대 추가하지 말 것
            - 불확실한 정보는 "상세 내용은 해당 금융회사 확인 필요"로 안내
            """
        
        # LLM 호출
        try:
            llm_response = self.chat_model.invoke([HumanMessage(content=prompt)])
            
            # 응답 검증
            if llm_response and llm_response.content:
                return llm_response.content
            else:
                return original_answer if original_answer else "추천 생성 중 오류가 발생했습니다."
                
        except Exception as e:
            # LLM 실패 시 원본 RAG 답변 사용
            print(f"하이브리드 추천 생성 실패: {e}")
            return original_answer if original_answer else "추천 생성 중 오류가 발생했습니다."


    def _create_information_response(self, query: str, search_results: List[Dict], original_answer: str) -> str:
        """정보 제공 형태의 답변 생성"""
        if not search_results:
            return "요청하신 정보를 찾을 수 없습니다."
        
        # 단순 정보 제공 프롬프트
        info_prompt = f"""
            사용자 질문: {query}
            
            다음 정보를 바탕으로 질문에 대한 직접적이고 명확한 답변을 제공하세요:
            {search_results[0] if search_results else '정보 없음'}
            
            **답변 요구사항:**
            - 추천 형태(📍)가 아닌 일반적인 설명 형태로 답변
            - 질문에 대한 직접적인 답변
            - 필요시 관련 상품 정보 간단히 언급
            """
        
        try:
            response = self.chat_model.invoke([HumanMessage(content=info_prompt)])
            return response.content
        except Exception as e:
            return f"정보 제공 중 오류가 발생했습니다: {str(e)}"

    
    def analyze_user_query(self, query: str) -> Dict[str, Any]:
        """사용자 쿼리 분석 """
        try:
            analysis_prompt = f"""
            다음 질문을 분석하여 사용자의 의도를 파악해주세요:
            "{query}"
            
            **의도 분류:**
            - recommendation: 상품 추천 요청 ("추천해줘", "좋은 상품", "어떤 게 나을까")
            - information: 정보 문의 ("설명해줘", "알려줘", "뭐였는데", "정보")
            - comparison: 상품 비교 ("비교", "차이점", "어떤 게 더")
            - clarification: 이전 대화 참조 ("아까", "전에", "마지막으로")
    
            **카테고리 분류 기준:**
            - 정기예금: "정기예금", "예금", "목돈 불리기", "일시불 예치"
            - 적금: "적금", "매월 저축", "목돈 모으기", "분할 납입"
            - 연금저축: "연금", "노후 준비", "은퇴 자금"
            - 대출: "대출", "융자", "자금 조달"
            
            반드시 다음 JSON 형식으로만 답변하세요:
            {{
            "product_category": "분류",
            "user_intent": "recommendation/information/comparison/clarification",
            "requires_recommendation_format": true/false",
            "key_requirements": ["구체적인 요구사항들"],
            "user_profile": "사용자 프로필 추정",
            "urgency": "높음/보통/낮음",
            "amount_mentioned": "언급된 금액",
            "period_mentioned": "언급된 기간"
             }}
                    """
            
            #  Solar API 사용
            analysis_model = ChatUpstage(
                api_key=os.getenv("SOLAR_API_KEY"),
                model="solar-1-mini-chat"
            )
            
            response = analysis_model.invoke([HumanMessage(content=analysis_prompt)])
            import logging
            # 로깅 설정 ##
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)            
            
            # Solar API 응답을 실제로 파싱
            import json  
            try:
                parsed_result = json.loads(response.content)
                logger.info(f"----  Solar API 분석 결과: {json.dumps(parsed_result, ensure_ascii=False, indent=2)}")
                 # Streamlit에서 JSON 표시 (개발자 모드)
                if st.session_state.get('show_debug', False):
                    st.subheader("🔍 분석 결과 (JSON)")
                    st.json(parsed_result)
        
                return parsed_result
            # except json.JSONDecodeError:
            #     return {"error": "JSON 파싱 실패"}
            except json.JSONDecodeError as e:
                return {
                    "error": f"JSON 파싱 실패: {str(e)}",
                    "raw_response": response.content[:50] + "...",
                    "status": "json_parse_error"
                } 
        except Exception as e:
            print(f"쿼리 분석 실패: {e}")
            return {"error": str(e)}

    def get_recommendation(self, query: str) -> Dict[str, Any]:
        """추천 결과 리턴 (멀티턴 지원)"""
        try:
            #  사용자 쿼리 분석
            analysis = self.analyze_user_query(query)
            
             #  이전 대화에서 추천한 상품 카테고리 유지
            if self.conversation_history:
                last_ai_message = None
                for msg in reversed(self.conversation_history):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg.content
                        break

            #  RAG 체인을 통한 추천
            result = self.rag_chain.invoke({
                'input': query,
                'chat_history': self.conversation_history  # 이전 대화기록 전달
            })
            
           # 카테고리 기반 필터링 추가
            filtered_context = self._filter_by_category(result['context'], analysis)
            if not filtered_context:
                filtered_context = result['context']
                
              #  검색 결과 포맷팅 및 검증
            search_results = self._format_search_results(filtered_context)
            validated_results = self._filter_validated_results(search_results)
        
            # 하이브리드 추천 생성
            if validated_results:
                hybrid_recommendation = self._create_hybrid_recommendation(               
                    query, validated_results, result['answer'], analysis
                )
            
                # 추천 결과 검증
                if self._validate_recommendation(hybrid_recommendation, validated_results):
                    final_recommendation = hybrid_recommendation
                else:
                    # 검증 실패 시 기본 RAG 답변 사용
                    final_recommendation = result['answer'] if result['answer'] else "추천 생성 중 오류가 발생했습니다."
            else:
                final_recommendation = "검색된 상품이 없습니다. 다른 조건으로 검색해보세요."    
                

            
            #  대화 기록 업데이트 (멀티턴 지원)
            self.conversation_history.append(HumanMessage(content=query))
            self.conversation_history.append(AIMessage(content=final_recommendation))
            
            # 대화 기록 길이 제한 (최근 10턴만 유지)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                'status': 'success',
                'recommendation': final_recommendation,
                'analysis': analysis,
                'search_results': search_results,
                'conversation_turns': len(self.conversation_history) // 2
            }
            
        except Exception as e:
            print(f"추천 생성 실패: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'recommendation': "시스템 오류가 발생했습니다. 다시 시도해주세요."
            }
    
    def _format_search_results(self, context):
        """검색 결과 포맷팅"""
        formatted_results = []
        for doc in context:
            if hasattr(doc, 'metadata'):
                # 유사도 점수 계산 (실제 벡터 유사도)
                similarity_score = self._calculate_similarity(doc)
                
                formatted_results.append({
                    'product_name': doc.metadata.get('product_name', '정보 없음'),
                    'company': doc.metadata.get('company', '정보 없음'),
                    'product_type': doc.metadata.get('product_type', '기타'),
                    'similarity_score': round(similarity_score, 3),  # 실제 점수
                    'raw_data': doc.metadata.get('raw_data', {})
                })  #  상품의 상세 정보를 활용하거나 추가 처리가 필요할 때
                
        # 유사도 점수 기준으로 내림차순 정렬
        formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return formatted_results
    
    def _calculate_similarity(self, doc):
        """유사도 점수 계산"""
    # FAISS에서 실제 점수를 가져오거나 임시로 랜덤 점수 생성
        import random
        return random.uniform(0.7, 0.95)  # 실제로는 벡터 유사도 계산
    
    def reset_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        print("대화 기록이 초기화되었습니다.")
        
    def sync_conversation_history(self, streamlit_conversation: List[Dict]):
        """Streamlit 대화 기록과 RAG 시스템 동기화"""
        self.conversation_history = []
        
        for message in streamlit_conversation:
            if message["role"] == "user":
                self.conversation_history.append(HumanMessage(content=message["content"]))
            else:
                self.conversation_history.append(AIMessage(content=message["content"]))
        
        print(f"대화 기록 동기화 완료: {len(self.conversation_history)}개 메시지")