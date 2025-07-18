import requests
import pandas as pd
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("FINLIFE_API_KEY")

@dataclass
class FinlifeConfig:
    """금융감독원 API 설정 클래스"""
    API_URLS = {
        'company': 'http://finlife.fss.or.kr/finlifeapi/companySearch.json',
        'deposit': 'http://finlife.fss.or.kr/finlifeapi/depositProductsSearch.json',
        'saving': 'http://finlife.fss.or.kr/finlifeapi/savingProductsSearch.json',
        'annuity': 'http://finlife.fss.or.kr/finlifeapi/annuitySavingProductsSearch.json',
        'mortgage': 'http://finlife.fss.or.kr/finlifeapi/mortgageLoanProductsSearch.json',
        'rent': 'http://finlife.fss.or.kr/finlifeapi/rentHouseLoanProductsSearch.json',
        'credit': 'http://finlife.fss.or.kr/finlifeapi/creditLoanProductsSearch.json'
    }
    
    REGION_CODES = {
        "020000": "은행",
        "030200": "여신전문금융업",
        "030300": "저축은행",
        "050000": "보험회사",
        "060000": "금융투자업"
    }
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    REQUEST_TIMEOUT = 30  # seconds

class FinlifeAPIClient:
    """API 통신 클래스"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FinlifeDataCollector/1.0'
        })
              
    def fetch_data(self, url: str, params: Optional[Dict] = None, 
                   max_retries: int = FinlifeConfig.MAX_RETRIES) -> Optional[Dict]:
        """API에서 데이터를 가져오는 메서드 (재시도 로직 포함)"""
        if params is None:
            params = {}
        
        params['auth'] = self.api_key
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=FinlifeConfig.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                
                # API 응답 에러 체크
                if 'result' not in data:
                    raise ValueError(f"Invalid API response format: {data}")
                
                return data
                
            except requests.exceptions.RequestException as e:               
                if attempt < max_retries - 1:
                    time.sleep(FinlifeConfig.RETRY_DELAY * (attempt + 1))
                else:                    
                    raise
            except ValueError as e:                
                raise
        
        return None

class FinlifeDataProcessor:
    """데이터 처리 및 변환 클래스"""
    
    @staticmethod
    def merge_base_and_option_lists(base_list: List[Dict], option_list: List[Dict], 
                                   prdt_div: str) -> List[Dict]:
        """baseList와 optionList를 prdt_div에 따라 조건부 JOIN"""
        if not base_list and not option_list:
            return []
        
        try:
            df_base = pd.DataFrame(base_list) if base_list else pd.DataFrame()
            df_option = pd.DataFrame(option_list) if option_list else pd.DataFrame()
            
            join_key = 'fin_co_no' if prdt_div == 'F' else 'fin_prdt_cd'           
          
            # 데이터 병합
            if not df_base.empty and not df_option.empty:
                merged_df = pd.merge(df_base, df_option, on=join_key, how='left')
            elif not df_base.empty:
                merged_df = df_base
            else:
                merged_df = df_option
            
            return merged_df.to_dict('records') if not merged_df.empty else []
            
        except Exception as e:
            print(f"Error merging data: {e}")
            return []
    
    @staticmethod
    def add_region_info(data: List[Dict], region_code: str, region_name: str) -> List[Dict]:
        """데이터에 권역 정보 추가"""
        return [
            {**item, 'region_code': region_code, 'region_name': region_name}
            for item in data
        ]

class FinlifeDataSaver:
    """파일 저장 클래스"""
    
    def __init__(self, output_dir: str = '../data/raw'):
        self.output_dir = output_dir
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """출력 디렉토리 생성"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def save_to_csv(self, data: List[Dict], filename: str) -> bool:
        """데이터를 CSV 파일로 저장"""
        try:
            if not data:
                print(f"No data to save for {filename}")
                return False
                
            df = pd.DataFrame(data)
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"Saved {len(data)} records to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving CSV file {filename}: {e}")
            return False
    
    def save_count_info(self, count: int, filename: str) -> bool:
        """총 개수 정보를 텍스트 파일로 저장"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(count))
            return True
        except Exception as e:
            print(f"Error saving count file {filename}: {e}")
            return False
    
    @staticmethod
    def get_current_datetime() -> str:
        """현재 날짜와 시간을 문자열로 반환"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

class FinlifeDataCollector:
    """메인 데이터 수집 클래스"""
    
    def __init__(self, api_key: str, output_dir: str = '../data/raw'):
        self.api_client = FinlifeAPIClient(api_key)
        self.data_processor = FinlifeDataProcessor()
        self.data_saver = FinlifeDataSaver(output_dir)
        
        # 데이터 저장용
        self.data_by_prdt_div: Dict[str, List[Dict]] = {}
        self.total_counts_by_prdt_div: Dict[str, int] = {}
          
    def fetch_all_pages_by_region(self, api_key: str, region_code: str) -> Tuple[Optional[str], int, List[Dict]]:
        """특정 API와 권역코드의 모든 페이지 데이터를 가져오는 메서드"""
        url = FinlifeConfig.API_URLS[api_key]
        region_name = FinlifeConfig.REGION_CODES[region_code]
        
        try:
            # 첫 번째 요청으로 메타데이터 확인
            first_data = self.api_client.fetch_data(url, params={'pageNo': 1, 'topFinGrpNo': region_code})
            if not first_data:
                return None, 0, []
            
            result = first_data.get('result', {})
            total_count = result.get('total_count', 0)
            max_page_no = result.get('max_page_no', 1)
            prdt_div = result.get('prdt_div', api_key)
            
            if total_count == 0:
                print(f"No data found for {api_key} - {region_name}")
                return prdt_div, 0, []
            
            # 모든 페이지 데이터 수집
            all_base_list = []
            all_option_list = []
            
            for page in range(1, max_page_no + 1):
                data = self.api_client.fetch_data(url, params={'pageNo': page, 'topFinGrpNo': region_code})
                if data:
                    res = data.get('result', {})
                    all_base_list.extend(res.get('baseList', []))
                    all_option_list.extend(res.get('optionList', []))
            
            # 데이터 병합 및 권역 정보 추가
            merged_data = self.data_processor.merge_base_and_option_lists(
                all_base_list, all_option_list, prdt_div
            )
            merged_data = self.data_processor.add_region_info(
                merged_data, region_code, region_name
            )
            
            print(f"Collected {len(merged_data)} records for {api_key} - {region_name}")
            return prdt_div, total_count, merged_data
            
        except Exception as e:
            print(f"Error fetching data for {api_key} - {region_name}: {e}")
            return None, 0, []
    
    def collect_all_data(self):
        """모든 API의 모든 권역코드 데이터를 수집하여 prdt_div별로 통합"""
        print("---------- Starting data collection...")
        
        for api_key in FinlifeConfig.API_URLS.keys():
            print(f"Processing {api_key.upper()} API")
            
            for region_code in FinlifeConfig.REGION_CODES.keys():
                prdt_div, total_count, merged_data = self.fetch_all_pages_by_region(api_key, region_code)
                
                if prdt_div and merged_data:
                    # prdt_div별로 데이터 누적
                    if prdt_div not in self.data_by_prdt_div:
                        self.data_by_prdt_div[prdt_div] = []
                        self.total_counts_by_prdt_div[prdt_div] = 0
                    
                    self.data_by_prdt_div[prdt_div].extend(merged_data)
                    self.total_counts_by_prdt_div[prdt_div] += total_count
    
    def save_all_data(self):
        """prdt_div별로 통합된 데이터를 CSV 파일로 저장"""
        current_datetime = self.data_saver.get_current_datetime()
        print("---------- Saving collected data...")
        
        for prdt_div, data in self.data_by_prdt_div.items():
            if data:
                # CSV 파일 저장
                csv_filename = f'{prdt_div}_{current_datetime}.csv'
                self.data_saver.save_to_csv(data, csv_filename)
                
                # 총 개수 정보 저장
                count_filename = f'{prdt_div}_total_count_{current_datetime}.txt'
                self.data_saver.save_count_info(
                    self.total_counts_by_prdt_div[prdt_div], 
                    count_filename
                )
    
    def run_all(self):
        """모든 데이터를 수집하고 저장하는 메인 메서드"""
        try:
            self.collect_all_data()
            self.save_all_data()
            
            # 결과 요약
            print("=== Collection Summary ===")
            print(f"Total prdt_div categories: {len(self.data_by_prdt_div)}")
            for prdt_div, data in self.data_by_prdt_div.items():
                print(f"prdt_div '{prdt_div}': {len(data)} records")
                
        except Exception as e:
            print(f"Error during data collection: {e}")
            raise

if __name__ == "__main__":
    collector = FinlifeDataCollector(api_key)
    collector.run_all()