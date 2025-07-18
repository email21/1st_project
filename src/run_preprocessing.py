import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from data_preprocessor import DataPreprocessor

def preprocess_all_products() -> None:
    """모든 금융상품 전처리 실행"""
    
    # 상품 유형별 파일 매핑
    product_configs: Dict[str, str] = {
        'deposit': '../data/raw/D_20250714_175227.csv',
        'saving': '../data/raw/S_20250714_175227.csv',
        'annuity': '../data/raw/P_20250714_175227.csv',
        'mortgage': '../data/raw/M_20250714_175227.csv',
        'rent': '../data/raw/R_20250714_175227.csv',
        'credit': '../data/raw/C_20250714_175227.csv'
    }
         
    # 각 상품별 전처리 실행
    for product_type, input_file in product_configs.items():
               
        # 입력 파일 존재 확인
        if not os.path.exists(input_file):
            continue
            
        try:
            # 전처리 실행
            preprocessor: DataPreprocessor = DataPreprocessor(input_file, product_type)
            processed_data: Optional[pd.DataFrame] = preprocessor.preprocess()
            
            if processed_data is not None:
                # 출력 파일 경로 설정
                output_file: str = f"../data/processed/{product_type}_processed.csv"
                
                # 전처리된 데이터 저장
                if preprocessor.save_processed_data(output_file):
                    print(f"✅ {product_type} 전처리 완료: {output_file}")
                else:
                    print(f"❌ {product_type} 저장 실패")
            else:
                print(f"❌ {product_type} 전처리 실패")
                
        except Exception as e:
            print(f"❌ {product_type} 처리 중 오류 발생: {str(e)}")
    
    print("=== 전처리 작업 완료 ===")

if __name__ == "__main__":
    preprocess_all_products()