import os
import pandas as pd
from typing import Dict, List, Any 
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_upstage import ChatUpstage # type: ignore
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

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
        
    def _load_financial_data(self):
        """금융상품 데이터 로드 및 벡터스토어 구축"""
        documents = []
        # data_dir = "../data/processed"
        data_dir = "data/processed"
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
         # 리트리버 설정 (상위 3개 문서 리턴) 3->5로 변경(0730)
        self.retriever = self.vector_store.as_retriever(k=5) # 벡터 검색에서 반환환할 상위 유사 문서의 개수, 이 3개 문서가 RAG 체인의 컨텍스트로 사용
        self.documents = documents
        
        print(f"📊 총 {len(documents)}개 상품 로드 완료")   

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
        # 제약사항 추가 
        rag_system_prompt = """
            당신은 금융상품 전문가입니다. 
            
            **절대 제약사항:** 
            - 오직 아래 검색 결과에 명시된 상품만 추천하세요
            - 검색 결과에 없는 상품명은 절대 언급하지 마세요
            - "검색된 상품 중에서" 라는 표현을 반드시 사용하세요

            **답변 원칙:**
            1. 이전 대화 맥락을 항상 고려하여 일관된 답변 제공
            2. "표로 정리", "목록으로", "요약해줘" 요청 시 → 마크다운 표 형식으로 답변
            3. "지금까지 추천한", "앞서 말한" 등 → 이전 대화 내용 참조하여 답변
            4. 새로운 추천 요청 시 → 📍 형식으로 상품 추천

            **표 요청 시 예시:**
            | 순번 | 상품명 | 금융회사 | 특징 |
            |------|--------|----------|------|
            | 1 | [이전 추천 상품1] | [회사] | [간단 설명] |

            **검색 결과를 활용하되, 사용자의 요청 형태에 맞게 자연스럽게 답변하세요.**
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
       
    def is_reference_query(self, query: str) -> bool:
        """LLM을 사용해 이전 대화 참조 여부 판단"""
        prompt = f"""
        다음 질문이 이전 대화 내용을 참조하는지 판단하세요:
        "{query}"
        
        이전 대화 참조하는 경우: True
        새로운 질문인 경우: False
        
        답변은 True 또는 False만 하세요.
        """
        
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        return "True" in response.content
    
    def _format_conversation_history(self) -> str:
        """대화 기록을 문자열로 포맷팅"""
        formatted = ""
        for msg in self.conversation_history:
            if isinstance(msg, HumanMessage):
                formatted += f"사용자: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                formatted += f"시스템: {msg.content}\n"
        return formatted      

    def get_recommendation(self, query: str) -> Dict[str, Any]:
        """추천 결과 리턴 (멀티턴 지원)"""
        try:           
            # 이전 대화 참조 여부 판단
            is_reference = self.is_reference_query(query)
        
            if is_reference and self.conversation_history:
                # 이전 대화 참조 질문인 경우 - 벡터 검색 없이 대화 기록만 활용
                prompt = f"""
                이전 대화 기록:
                {self._format_conversation_history()}                
                현재 질문: {query}                
                이전 대화에서 추천한 상품들을 참조하여 답변해주세요.
                새로운 상품을 추천하지 말고, 기존 추천 상품들만 활용하세요.
                """            
                response = self.chat_model.invoke([HumanMessage(content=prompt)])
                result_answer = response.content
            else:
                # 새로운 질문인 경우 - RAG 방식 사용
                result = self.rag_chain.invoke({
                    'input': query,
                    'chat_history': self.conversation_history  # 이전 대화기록 전달
                })
                # 검색 결과 검증 추가
                retrieved_docs = result.get('context', [])  
                if not retrieved_docs or len(retrieved_docs) == 0:
                    result_answer = "죄송합니다. 요청하신 조건에 맞는 상품을 구축된 데이터에서 찾을 수 없습니다. 더 구체적인 조건을 말씀해주시거나 다른 상품 유형으로 문의해보시겠어요?"
                else:
                    result_answer = result['answer']         
           
            #  대화 기록 업데이트 (멀티턴 지원)
            self.conversation_history.append(HumanMessage(content=query))
            self.conversation_history.append(AIMessage(content=result_answer))
            
            # 대화 기록 길이 제한 (최근 10턴만 유지)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]            
            
            return {
                'status': 'success',
                'recommendation': result_answer,
                'conversation_turns': len(self.conversation_history) // 2
            }     
                    
            # 대화 기록 업데이트 (멀티턴 지원) 이후 추가
            print(f"=========== [DEBUG][백엔드] get_recommendation 호출 후 대화 길이: {len(self.conversation_history)}, 턴수: {len(self.conversation_history)//2}")
            for idx, msg in enumerate(self.conversation_history):
                print(f"  [{idx}] {msg.__class__.__name__}: {msg.content}")
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}              

    def reset_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        print("=======--- [DEBUG] [백엔드] 대화 기록이 초기화되었습니다. --- financial_rag_system.py")
        print("--- 대화 기록이 초기화되었습니다. --- financial_rag_system.py")
        
    def sync_conversation_history(self, streamlit_conversation: List[Dict]):
        """Streamlit 대화 기록과 RAG 시스템 동기화"""
        self.conversation_history = []
        
        for message in streamlit_conversation:
            if message["role"] == "user":
                self.conversation_history.append(HumanMessage(content=message["content"]))
            else:
                self.conversation_history.append(AIMessage(content=message["content"]))
        
        print(f"대화 기록 동기화 완료: {len(self.conversation_history)}개 메시지")
        
         # 실제 누적된 메시지 목록 확인
        print("====================== [DEBUG][백엔드] 동기화된 전체 대화 내역:")
        for idx, msg in enumerate(self.conversation_history):
            print(f"  {idx}번째({type(msg).__name__}): {msg.content}")