# financial_rag/preprocessing/run_preprocessing.py 실행 스크립트

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional
from data_preprocessor import DataPreprocessor

def setup_logging() -> logging.Logger:
    """로깅 설정"""
    log_dir = Path('../logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'preprocessing_runner.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories(logger: logging.Logger) -> None:
    """필요한 디렉토리 생성"""
    processed_dir = Path("../data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ 디렉토리 생성 완료: {processed_dir}")

def preprocess_all_products() -> None:
    """모든 금융상품 전처리 실행"""
    
    # 로깅 설정
    logger: logging.Logger = setup_logging()
    
    # 상품 유형별 파일 매핑
    product_configs: Dict[str, str] = {
        'deposit': '../data/raw/D_20250714_175227.csv',
        'saving': '../data/raw/S_20250714_175227.csv',
        'annuity': '../data/raw/P_20250714_175227.csv',
        'mortgage': '../data/raw/M_20250714_175227.csv',
        'rent': '../data/raw/R_20250714_175227.csv',
        'credit': '../data/raw/C_20250714_175227.csv'
    }
    
    # 디렉토리 생성
    create_directories(logger)
    
    logger.info("=== 전처리 작업 시작 ===")
    
    # 각 상품별 전처리 실행
    for product_type, input_file in product_configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 {product_type} 상품 전처리 시작")
        logger.info(f"{'='*50}")
        
        # 입력 파일 존재 확인
        if not os.path.exists(input_file):
            logger.warning(f"⚠️ 파일이 존재하지 않습니다: {input_file}")
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
                    logger.info(f"✅ {product_type} 전처리 완료: {output_file}")
                else:
                    logger.error(f"❌ {product_type} 저장 실패")
            else:
                logger.error(f"❌ {product_type} 전처리 실패")
                
        except Exception as e:
            logger.error(f"❌ {product_type} 처리 중 오류 발생: {str(e)}")
    
    logger.info("=== 전처리 작업 완료 ===")

if __name__ == "__main__":
    preprocess_all_products()
