import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os

class VectorStore:  
    def __init__(self, model_name: str = "kakaobank/kf-deberta-base"):
        self.model_name: str = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []
        self.dimension: int = 768  # KF-DeBERTa의 기본 hidden_size
        self._setup_logging()
    
    def _setup_logging(self) -> None:       
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
     
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info(f"🤖 KF-DeBERTa 모델 ({self.model_name}) 초기화 완료")
    
    def load_processed_data(self, data_dir: str = "../data/processed") -> None:
        """전처리된 데이터 로드"""
        self.logger.info("📁 전처리된 데이터 로드 시작")
        
        product_files: Dict[str, str] = {
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
                        self.documents.append(doc)
                    
                    self.logger.info(f"✅ {product_type}: {len(df)}개 상품 로드")
                except Exception as e:
                    self.logger.error(f"❌ {product_type} 로드 실패: {e}")
            else:
                self.logger.warning(f"⚠️ 파일 없음: {filepath}")
        
        self.logger.info(f"📊 총 {len(self.documents)}개 상품 로드 완료")
    
    def build_index(self) -> None:
        """FAISS 인덱스 구축 """
        if not self.documents:
            self.logger.error("❌ 문서가 없습니다. 먼저 데이터를 로드하세요.")
            return
        
        self.logger.info("🔧 KF-DeBERTa 벡터 인덱스 구축 시작")
        
        # 텍스트 임베딩 생성 (KF-DeBERTa 사용)
        texts = [doc['search_text'] for doc in self.documents]
        
        # KF-DeBERTa로 임베딩 생성
        self.logger.info("🧠 KF-DeBERTa로 임베딩 생성 중...")
        embeddings = self.encoder.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32,  # 배치 크기 설정
            convert_to_numpy=True
        )
        
        # FAISS 인덱스 생성
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 정규화 후 인덱스 추가
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.logger.info(f"✅ KF-DeBERTa 벡터 인덱스 구축 완료: {len(self.documents)}개 문서")
    
    def search(self, query: str, top_k: int = 5, product_type: Optional[str] = None) -> List[Dict]:
        """유사도 검색 """
        if self.index is None:
            self.logger.error("❌ 인덱스가 구축되지 않았습니다.")
            return []
        
        # 쿼리 임베딩 
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # 검색 실행
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                
                # 상품 타입 필터링
                if product_type is None or doc['product_type'] == product_type:
                    results.append(doc)
                    
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """KF-DeBERTa로 단일 텍스트 임베딩 생성"""
        embedding = self.encoder.encode([text], convert_to_numpy=True)
        return embedding[0]
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """배치 검색 - 여러 쿼리 동시 처리"""
        if self.index is None:
            self.logger.error("❌ 인덱스가 구축되지 않았습니다.")
            return []
        
        # 배치 임베딩 생성
        query_embeddings = self.encoder.encode(queries, convert_to_numpy=True)
        faiss.normalize_L2(query_embeddings)
        
        # 배치 검색
        scores, indices = self.index.search(query_embeddings.astype('float32'), top_k)
        
        batch_results = []
        for query_scores, query_indices in zip(scores, indices):
            query_results = []
            for score, idx in zip(query_scores, query_indices):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    query_results.append(doc)
            batch_results.append(query_results)
        
        return batch_results
    
    def save_index(self, save_path: str = "../data/vector_store") -> None:

        try:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)
            self.logger.info(f"📁 현재 작업 디렉토리: {os.getcwd()}")
            self.logger.info(f"💾 저장 경로: {save_path}")
            
            if self.index is None:
                self.logger.error("❌ 인덱스가 생성되지 않았습니다.")
                return
            
            # FAISS 인덱스 저장
            index_file = os.path.join(save_path, "kf_deberta_index.bin")
            faiss.write_index(self.index, index_file)
            self.logger.info(f"✅ FAISS 인덱스 저장: {index_file}")
            
            # 문서 메타데이터 저장
            docs_file = os.path.join(save_path, "documents.pkl")
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            self.logger.info(f"✅ 문서 메타데이터 저장: {docs_file}")
            
            # 모델 정보 저장
            info_file = os.path.join(save_path, "model_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Dimension: {self.dimension}\n")
                f.write(f"Documents: {len(self.documents)}\n")
            self.logger.info(f"✅ 모델 정보 저장: {info_file}")
            
            saved_files = os.listdir(save_path)
            self.logger.info(f"📋 저장된 파일들: {saved_files}")
            
        except Exception as e:
            self.logger.error(f"❌ 인덱스 저장 실패: {e}")
            print(f"저장 실패: {e}")
    
    def load_index(self, load_path: str = "../data/vector_store") -> bool:
        try:
            # FAISS 인덱스 로드
            index_file = os.path.join(load_path, "kf_deberta_index.bin")
            if os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
            else:
                # 기존 파일명도 시도
                self.index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))
            
            # 문서 메타데이터 로드
            with open(os.path.join(load_path, "documents.pkl"), 'rb') as f:
                self.documents = pickle.load(f)
            
            self.logger.info(f"✅ KF-DeBERTa 벡터 인덱스 로드 완료: {len(self.documents)}개 문서")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 인덱스 로드 실패: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_type': 'KF-DeBERTa',
            'dimension': self.dimension,
            'document_count': len(self.documents),
            'index_built': self.index is not None
        }

if __name__ == "__main__":
    print("🚀 KF-DeBERTa 벡터 스토어 구축 시작")
    vector_store = VectorStore(model_name="kakaobank/kf-deberta-base")
    
    info = vector_store.get_model_info()
    print(f"📊 모델 정보: {info}")
    
    vector_store.load_processed_data()
    vector_store.build_index()
    vector_store.save_index()
    
    test_query = "높은 금리 예금 상품"
    results = vector_store.search(test_query, top_k=3)
    
    print(f"\n🔍 테스트 검색: '{test_query}'")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['product_name']} (유사도: {result['similarity_score']:.4f})")
    
    print("✅ KF-DeBERTa 벡터 스토어 구축 완료!")
