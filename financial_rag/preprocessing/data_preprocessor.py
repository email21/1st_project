from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

class DataPreprocessor:
    """
    금융상품 데이터 전처리 클래스
    - 정기예금, 적금, 연금저축, 주택담보대출, 전세자금대출, 개인신용대출
    """

    def __init__(self, file_path: str, product_type: str) -> None:
        """
        Args:
            file_path (str): CSV 파일 경로
            product_type (str): 상품 유형 ('deposit', 'saving', 'annuity', 'mortgage', 'rent', 'credit')
        """
        self.file_path: str = file_path
        self.product_type: str = product_type
        self.df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        
        # 로깅 설정
        self._setup_logging()

    def _setup_logging(self) -> None:
        """로깅 설정"""
        log_dir = Path('../logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'data_preprocessor.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

    def load_data(self) -> Optional[pd.DataFrame]:
        """데이터 로드"""
        try:
            self.df = pd.read_csv(self.file_path)
            self.logger.info(f"✅ 데이터 로드 완료: {len(self.df)}행, {len(self.df.columns)}열")
            return self.df
        except Exception as e:
            self.logger.error(f"❌ 데이터 로드 실패: {e}")
            return None

    def get_core_columns(self) -> Dict[str, List[str]]:
        """상품 유형별 핵심 컬럼 정의"""
        column_mapping: Dict[str, Dict[str, List[str]]] = {
            # 정기예금
            'deposit': {
                'core': [
                    'kor_co_nm', 'fin_prdt_nm', 'join_way',
                    'mtrt_int', 'spcl_cnd', 'join_deny',
                    'join_member', 'etc_note', 'max_limit',
                    'save_trm', 'intr_rate', 'intr_rate2'
                ]
            },
            # 적금
            'saving': {
                'core': [
                    'kor_co_nm', 'fin_prdt_nm', 'join_way',
                    'mtrt_int', 'spcl_cnd', 'join_deny',
                    'join_member', 'etc_note', 'max_limit',
                    'rsrv_type_nm', 'save_trm',
                    'intr_rate', 'intr_rate2'
                ]
            },
            # 연금저축 P 16
            'annuity': {
                'core': [
                    'kor_co_nm', 'fin_prdt_nm', 'join_way',
                    'pnsn_kind_nm', 'prdt_type_nm', 'dcls_rate',
                    'guar_rate', 'btrm_prft_rate_1', 'btrm_prft_rate_2',
                    'btrm_prft_rate_3', 'etc', 'pnsn_recp_trm_nm',
                    'pnsn_entr_age_nm', 'mon_paym_atm_nm',
                    'paym_prd_nm', 'pnsn_strt_age_nm'
                ]
            },
            # 주택담보대출
            'mortgage': {
                'core': [
                    'kor_co_nm', 'fin_prdt_nm', 'join_way',
                    'loan_inci_expn', 'erly_rpay_fee', 'dly_rate',
                    'loan_lmt', 'mrtg_type_nm', 'rpay_type_nm',
                    'lend_rate_type_nm', 'lend_rate_min',
                    'lend_rate_max', 'lend_rate_avg'
                ]
            },
            # 전세자금대출
            'rent': {
                'core': [
                    'kor_co_nm', 'fin_prdt_nm', 'join_way',
                    'loan_inci_expn', 'erly_rpay_fee', 'dly_rate',
                    'loan_lmt', 'rpay_type_nm', 'lend_rate_type_nm',
                    'lend_rate_min', 'lend_rate_max', 'lend_rate_avg'
                ]
            },
            # 개인신용대출 12
            'credit': {
                'core': [
                    'kor_co_nm', 'fin_prdt_nm', 'join_way',
                    'crdt_prdt_type_nm', 'crdt_lend_rate_type_nm',
                    'crdt_grad_1', 'crdt_grad_4', 'crdt_grad_5',
                    'crdt_grad_6', 'crdt_grad_10', 'crdt_grad_11',
                    'crdt_grad_avg'
                ]
            }
        }
        
        return column_mapping.get(self.product_type, {})

    def clean_data(self) -> Optional[pd.DataFrame]:
        """데이터 정제"""
        if self.df is None:
            self.logger.error("❌ 데이터가 로드되지 않았습니다.")
            return None

        # 핵심 컬럼 추출
        column_config: Dict[str, List[str]] = self.get_core_columns()
        if not column_config:
            self.logger.error(f"❌ 지원하지 않는 상품 유형: {self.product_type}")
            return None

        # 핵심 컬럼만 선택
        available_cols: List[str] = [col for col in column_config['core'] if col in self.df.columns]
        self.processed_df = self.df[available_cols].copy()

        # 필수 컬럼 결측값 제거
        essential_cols: List[str] = ['kor_co_nm', 'fin_prdt_nm']
        initial_count: int = len(self.processed_df)
        self.processed_df.dropna(subset=essential_cols, inplace=True)
        self.logger.info(f"🔶 필수 컬럼 결측값 제거: {initial_count} → {len(self.processed_df)}행")

        # 텍스트 데이터 정제
        self.processed_df['kor_co_nm'] = self.processed_df['kor_co_nm'].str.strip()
        self.processed_df['fin_prdt_nm'] = self.processed_df['fin_prdt_nm'].str.strip()

        # 금리 데이터 정제
        if self.product_type in ['deposit', 'saving', 'annuity','mortgage', 'rent', 'credit']:
            rate_columns: List[str] = [col for col in self.processed_df.columns if 'rate' in col or 'grad' in col]
            for col in rate_columns:
                if col in self.processed_df.columns:
                    self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors='coerce')

        # 중복 제거
        before_dedup: int = len(self.processed_df)
        self.processed_df.drop_duplicates(subset=['kor_co_nm', 'fin_prdt_nm'], inplace=True)
        self.logger.info(f"🔶 중복 제거: {before_dedup} → {len(self.processed_df)}행")

        # 연금상품 etc 컬럼 특별 처리 추가
        if self.product_type == 'annuity' and 'etc' in self.processed_df.columns:
            self.logger.info("🔧 연금상품 etc 컬럼 특별 처리 중...")
            # etc 컬럼 사용 가능 여부 판단
            self.processed_df['etc_available'] = (
                self.processed_df['etc'].notna() &
                (self.processed_df['etc'].str.strip() != '')
            )
            # etc 정보 정제
            self.processed_df['etc_clean'] = self.processed_df['etc'].fillna('').str.strip()
            
            # 연금 etc 통계 출력
            available_count: int = self.processed_df['etc_available'].sum()
            total_count: int = len(self.processed_df)
            self.logger.info(f" 📋 etc 정보 보유: {available_count}/{total_count}개 상품 ({available_count/total_count*100:.1f}%)")

        return self.processed_df

    def create_search_text(self, row: pd.Series) -> str:
        """RAG 검색용 통합 텍스트 생성"""
        if self.product_type == 'deposit':
            search_text: str = f"""
상품명: {row.get('fin_prdt_nm', 'N/A')}
금융회사: {row.get('kor_co_nm', 'N/A')}
기본금리: {row.get('intr_rate', 'N/A')}%
최고금리: {row.get('intr_rate2', 'N/A')}%
저축기간: {row.get('save_trm', 'N/A')}개월
최고한도: {row.get('max_limit', 'N/A')}
우대조건: {row.get('spcl_cnd', 'N/A')}
가입제한: {row.get('join_deny', 'N/A')}
가입대상: {row.get('join_member', 'N/A')}
가입방법: {row.get('join_way', 'N/A')}
"""

        elif self.product_type == 'saving':
            search_text = f"""
상품명: {row.get('fin_prdt_nm', 'N/A')}
금융회사: {row.get('kor_co_nm', 'N/A')}
기본금리: {row.get('intr_rate', 'N/A')}%
최고금리: {row.get('intr_rate2', 'N/A')}%
저축기간: {row.get('save_trm', 'N/A')}개월
적립유형: {row.get('rsrv_type_nm', 'N/A')}
최고한도: {row.get('max_limit', 'N/A')}
우대조건: {row.get('spcl_cnd', 'N/A')}
가입제한: {row.get('join_deny', 'N/A')}
가입대상: {row.get('join_member', 'N/A')}
가입방법: {row.get('join_way', 'N/A')}
"""

        elif self.product_type == 'annuity':
            search_text = f"""
상품명: {row.get('fin_prdt_nm', 'N/A')}
금융회사: {row.get('kor_co_nm', 'N/A')}
연금종류: {row.get('pnsn_kind_nm', 'N/A')}
상품유형: {row.get('prdt_type_nm', 'N/A')}
공시이율: {row.get('dcls_rate', 'N/A')}%
최저보증이율: {row.get('guar_rate', 'N/A')}%
과거수익률1년: {row.get('btrm_prft_rate_1', 'N/A')}%
과거수익률2년: {row.get('btrm_prft_rate_2', 'N/A')}%
과거수익률3년: {row.get('btrm_prft_rate_3', 'N/A')}%
연금수령기간: {row.get('pnsn_recp_trm_nm', 'N/A')}
가입연령: {row.get('pnsn_entr_age_nm', 'N/A')}
월납입금액: {row.get('mon_paym_atm_nm', 'N/A')}
납입기간: {row.get('paym_prd_nm', 'N/A')}
연금개시연령: {row.get('pnsn_strt_age_nm', 'N/A')}
가입방법: {row.get('join_way', 'N/A')}
"""

        elif self.product_type == 'mortgage':
            search_text = f"""
상품명: {row['fin_prdt_nm']}
금융회사: {row['kor_co_nm']}
담보유형: {row.get('mrtg_type_nm', 'N/A')}
상환방식: {row.get('rpay_type_nm', 'N/A')}
금리유형: {row.get('lend_rate_type_nm', 'N/A')}
최저금리: {row.get('lend_rate_min', 'N/A')}%
최고금리: {row.get('lend_rate_max', 'N/A')}%
평균금리: {row.get('lend_rate_avg', 'N/A')}%
대출한도: {row.get('loan_lmt', 'N/A')}
가입방법: {row.get('join_way', 'N/A')}
부대비용: {row.get('loan_inci_expn', 'N/A')}
중도상환수수료: {row.get('erly_rpay_fee', 'N/A')}
연체금리: {row.get('dly_rate', 'N/A')}
"""

        elif self.product_type == 'rent':
            search_text = f"""
상품명: {row['fin_prdt_nm']}
금융회사: {row['kor_co_nm']}
상환방식: {row.get('rpay_type_nm', 'N/A')}
금리유형: {row.get('lend_rate_type_nm', 'N/A')}
최저금리: {row.get('lend_rate_min', 'N/A')}%
최고금리: {row.get('lend_rate_max', 'N/A')}%
평균금리: {row.get('lend_rate_avg', 'N/A')}%
대출한도: {row.get('loan_lmt', 'N/A')}
가입방법: {row.get('join_way', 'N/A')}
부대비용: {row.get('loan_inci_expn', 'N/A')}
중도상환수수료: {row.get('erly_rpay_fee', 'N/A')}
"""

        elif self.product_type == 'credit':
            search_text = f"""
상품명: {row['fin_prdt_nm']}
금융회사: {row['kor_co_nm']}
대출유형: {row.get('crdt_prdt_type_nm', 'N/A')}
금리유형: {row.get('crdt_lend_rate_type_nm', 'N/A')}
신용등급별금리:
- 900점초과: {row.get('crdt_grad_1', 'N/A')}%
- 801-900점: {row.get('crdt_grad_4', 'N/A')}%
- 701-800점: {row.get('crdt_grad_5', 'N/A')}%
- 601-700점: {row.get('crdt_grad_6', 'N/A')}%
- 501-600점: {row.get('crdt_grad_10', 'N/A')}%
- 401-500점: {row.get('crdt_grad_11', 'N/A')}%
평균금리: {row.get('crdt_grad_avg', 'N/A')}%
가입방법: {row.get('join_way', 'N/A')}
"""

        else:
            search_text = f"상품명: {row['fin_prdt_nm']}, 금융회사: {row['kor_co_nm']}"

        return search_text.strip()

    def add_search_text_column(self) -> Optional[pd.DataFrame]:
        """검색용 텍스트 컬럼 추가"""
        if self.processed_df is None:
            self.logger.error("❌ 전처리된 데이터가 없습니다.")
            return None

        self.processed_df['search_text'] = self.processed_df.apply(
            lambda row: self.create_search_text(row), axis=1
        )

        self.logger.info("✅ 검색용 텍스트 컬럼 추가 완료")
        return self.processed_df


    def get_statistics(self) -> Optional[Dict[str, Any]]:
        """데이터 통계 정보 출력"""
        if self.processed_df is None:
            return None

        stats: Dict[str, Any] = {
            '일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '총 상품 수': len(self.processed_df),
            '금융회사 수': self.processed_df['kor_co_nm'].nunique(),
            '상품 유형': self.product_type,
            '주요 금융회사': self.processed_df['kor_co_nm'].value_counts().head(5).to_dict()
        }

        # 상품별 금리 통계 추가
        if self.product_type in ['deposit', 'saving']:
        # 예적금 금리 통계
            if 'intr_rate' in self.processed_df.columns:
                valid_rates = self.processed_df['intr_rate'].dropna()
                if len(valid_rates) > 0:
                    stats['기본금리 범위'] = f"{valid_rates.min():.2f}% ~ {valid_rates.max():.2f}%"
                    stats['기본금리 평균'] = f"{valid_rates.mean():.2f}%"
            
            if 'intr_rate2' in self.processed_df.columns:
                valid_rates2 = self.processed_df['intr_rate2'].dropna()
                if len(valid_rates2) > 0:
                    stats['최고우대금리 범위'] = f"{valid_rates2.min():.2f}% ~ {valid_rates2.max():.2f}%"
                    stats['최고우대금리 평균'] = f"{valid_rates2.mean():.2f}%"
            
            # 적금 특화 통계
            if self.product_type == 'saving' and 'rsrv_type_nm' in self.processed_df.columns:
                stats['적립유형 분포'] = self.processed_df['rsrv_type_nm'].value_counts().to_dict()
    
        elif self.product_type == 'annuity':
            # 연금저축 금리 통계
            if 'dcls_rate' in self.processed_df.columns:
                valid_dcls = self.processed_df['dcls_rate'].dropna()
                if len(valid_dcls) > 0:
                    stats['공시이율 범위'] = f"{valid_dcls.min():.2f}% ~ {valid_dcls.max():.2f}%"
                    stats['공시이율 평균'] = f"{valid_dcls.mean():.2f}%"
            
            if 'guar_rate' in self.processed_df.columns:
                valid_guar = self.processed_df['guar_rate'].dropna()
                if len(valid_guar) > 0:
                    stats['최저보증이율 범위'] = f"{valid_guar.min():.2f}% ~ {valid_guar.max():.2f}%"
                    stats['최저보증이율 평균'] = f"{valid_guar.mean():.2f}%"
            
            # 과거 수익률 통계
            profit_rates = ['btrm_prft_rate_1', 'btrm_prft_rate_2', 'btrm_prft_rate_3']
            for i, rate_col in enumerate(profit_rates, 1):
                if rate_col in self.processed_df.columns:
                    valid_profit = self.processed_df[rate_col].dropna()
                    if len(valid_profit) > 0:
                        stats[f'과거수익률{i}년 범위'] = f"{valid_profit.min():.2f}% ~ {valid_profit.max():.2f}%"
            
            # 연금 종류 분포
            if 'pnsn_kind_nm' in self.processed_df.columns:
                stats['연금종류 분포'] = self.processed_df['pnsn_kind_nm'].value_counts().to_dict()
    
          # 금리 통계 (대출 상품)
        elif self.product_type in ['mortgage', 'rent']:
            if 'lend_rate_min' in self.processed_df.columns:
                stats['최저금리 범위'] = f"{self.processed_df['lend_rate_min'].min():.2f}% ~ {self.processed_df['lend_rate_min'].max():.2f}%"
            if 'lend_rate_max' in self.processed_df.columns:
                stats['최고금리 범위'] = f"{self.processed_df['lend_rate_max'].min():.2f}% ~ {self.processed_df['lend_rate_max'].max():.2f}%"
        elif self.product_type == 'credit':
            if 'crdt_grad_avg' in self.processed_df.columns:
                stats['평균금리 범위'] = f"{self.processed_df['crdt_grad_avg'].min():.2f}% ~ {self.processed_df['crdt_grad_avg'].max():.2f}%"

        return stats
    
    def preprocess(self) -> Optional[pd.DataFrame]:
        """전체 전처리 파이프라인 실행"""
        self.logger.info(f"▶️ {self.product_type} 상품 데이터 전처리 시작")

        # 1. 데이터 로드
        if self.load_data() is None:
            return None

        # 2. 데이터 정제
        if self.clean_data() is None:
            return None

        # 3. 검색용 텍스트 추가
        self.add_search_text_column()

        # 4. 통계 정보 출력
        stats: Optional[Dict[str, Any]] = self.get_statistics()
        if stats:
            self.logger.info("\n📊 전처리 완료 통계:")
            for key, value in stats.items():
                self.logger.info(f" {key}: {value}")

        self.logger.info("✅ 전처리 완료!")
        return self.processed_df

    def save_processed_data(self, output_path: str) -> bool:
        """전처리된 데이터 저장"""
        if self.processed_df is None:
            self.logger.error("❌ 전처리된 데이터가 없습니다.")
            return False

        try:
            self.processed_df.to_csv(output_path, index=False, encoding='utf-8')
            self.logger.info(f"💾 전처리된 데이터 저장 완료: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ 데이터 저장 실패: {e}")
            return False
