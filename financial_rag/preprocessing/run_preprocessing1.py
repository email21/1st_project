# financial_rag/preprocessing/run_preprocessing.py 실행 스크립트
import os
import sys
from pathlib import Path
from data_preprocessor import DataPreprocessor

def create_directories():
    """필요한 디렉토리 생성"""
    processed_dir = Path("../data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 디렉토리 생성 완료: {processed_dir}")

def preprocess_all_products():
    """모든 금융상품 전처리 실행"""
    
    # 상품 유형별 파일 매핑
    product_configs = {
        'deposit': '../data/raw/D_20250714_175227.csv',
        'saving': '../data/raw/S_20250714_175227.csv', 
        'annuity': '../data/raw/P_20250714_175227.csv',
        'mortgage': '../data/raw/M_20250714_175227.csv',
        'rent': '../data/raw/R_20250714_175227.csv',
        'credit': '../data/raw/C_20250714_175227.csv'
    }
    
    # 디렉토리 생성
    create_directories()
    
    # 각 상품별 전처리 실행
    for product_type, input_file in product_configs.items():
        print(f"\n{'='*50}")
        print(f"🔄 {product_type} 상품 전처리 시작")
        print(f"{'='*50}")
        
        # 입력 파일 존재 확인
        if not os.path.exists(input_file):
            print(f"⚠️  파일이 존재하지 않습니다: {input_file}")
            continue
            
        # 전처리 실행
        preprocessor = DataPreprocessor(input_file, product_type)
        processed_data = preprocessor.preprocess()
        
        if processed_data is not None:
            # 출력 파일 경로 설정
            output_file = f"../data/processed/{product_type}_processed.csv"
            
            # 전처리된 데이터 저장
            if preprocessor.save_processed_data(output_file):
                print(f"✅ {product_type} 전처리 완료: {output_file}")
            else:
                print(f"❌ {product_type} 저장 실패")
        else:
            print(f"❌ {product_type} 전처리 실패")

if __name__ == "__main__":
    preprocess_all_products()
