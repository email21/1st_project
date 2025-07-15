import streamlit as st
import logging
from typing import Dict, Any
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.recommendation_system import FinancialRecommendationSystem

st.set_page_config(
    page_title="금융상품 추천 시스템",
    page_icon="💰",
    layout="wide"
)
MAX_CONVERSATION_TURNS = 10
def setup_logging() -> logging.Logger:
    log_dir = Path('../logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

@st.cache_resource
def load_recommendation_system() -> FinancialRecommendationSystem:
    return FinancialRecommendationSystem()

def main():
    logger = setup_logging()
        
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
        
    if 'conversation' not in st.session_state: 
        st.session_state.conversation = []    
    if 'input_counter' not in st.session_state:
        st.session_state.input_counter = 0
    
    def handle_input_change():
        try:
            current_key = f"input_area_{st.session_state.input_counter}"
            if current_key in st.session_state:
                st.session_state.user_input = st.session_state[current_key]
        except:
            st.session_state.user_input = ""
    
    if 'system_initialized' not in st.session_state:
        logger.info("=" * 50)
        logger.info("💰 금융상품 추천 시스템 시작")
        logger.info("=" * 50)
        logger.info("🌐 웹 애플리케이션을 시작합니다...")
        st.session_state.system_initialized = True
    
    st.title("금융상품 추천 시스템")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📋 시스템 정보")
        st.info("""        
        **지원 상품:**
        - 📈 정기예금
        - 💰 적금
        - 🏦 연금저축
        - 🏠 주택담보대출
        - 🏘️ 전세자금대출
        - 💳 개인신용대출
        """)
      
        st.markdown("---")
        st.header("💬 대화 관리")
        
        if st.session_state.conversation:
            turn_count = len(st.session_state.conversation) // 2
            st.write(f"**현재 대화 턴:** {turn_count}/{MAX_CONVERSATION_TURNS}턴")     
            if st.button("🗑️ 대화 기록 초기화", type="secondary"):
                st.session_state.conversation = []
                st.rerun()
        else:
            st.write("대화 기록이 없습니다.")
        
    if st.session_state.conversation:
        st.subheader("💬 대화 기록")
        turn_count = len(st.session_state.conversation) // 2
        st.info(f"📊 총 {turn_count}턴의 대화 / 최대 {MAX_CONVERSATION_TURNS}턴")
        
        with st.container(height=400):  # 높이 제한으로 스크롤 가능
            for i, chat in enumerate(st.session_state.conversation):
                if chat['role'] == 'user':
                    st.markdown(f"**👤 사용자 ({i//2 + 1}턴):** {chat['content']}")
                else:
                    st.markdown(f"**🤖 시스템 ({i//2 + 1}턴):**")
                    st.markdown(chat['content'])
                    st.markdown("---")
        st.markdown("---")


    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 질문하기")
                                
        user_input = st.text_area(
            "질문을 입력하세요:",
            value="",
            height=100,
            placeholder="예: 안정적으로 돈을 모을 수 있는 상품이 있을까요?",
            key=f"input_area_{st.session_state.input_counter}", 
            on_change=handle_input_change
        )
        
         # 입력값이 변경되면 세션 상태 업데이트
        if user_input != st.session_state.user_input:
            st.session_state.user_input = user_input    
        
        if st.button("🔍 상품 추천받기", type="primary"):
            current_input = user_input.strip()
            if current_input:
                st.session_state.conversation.append({
                'role': 'user', 
                'content': current_input
            })
                manage_conversation_overflow()
                
                with st.spinner("추천 상품을 찾고 있습니다..."):
                    try:
                        rec_system = load_recommendation_system()
                        contextual_query = create_conversation_context(
                            st.session_state.conversation[:-1],  # 현재 질문 제외
                            current_input
                        )                
                        result = rec_system.get_recommendation(contextual_query)

                        if result['status'] == 'success':
                            st.session_state.conversation.append({
                            'role': 'system', 
                            'content': result['recommendation']
                            })
                        
                            st.session_state.input_counter += 1 
                            st.rerun()                                 

                            with st.expander("🔍 질문 분석 결과", expanded=False):
                                analysis = result['analysis']
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.write("**상품 카테고리:**", analysis.get('product_category', 'N/A'))
                                    st.write("**사용자 의도:**", analysis.get('user_intent', 'N/A'))
                                
                                with col_b:
                                    st.write("**주요 요구사항:**")
                                    for req in analysis.get('key_requirements', []):
                                        st.write(f"- {req}")
                            
                            st.subheader("💡 추천 결과")
                            st.markdown(result['recommendation'])
                            
                            with st.expander("📊 검색된 상품 목록", expanded=False):
                                search_results = result['search_results']
                                if search_results:
                                    for i, product in enumerate(search_results, 1):
                                        st.write(f"**{i}. {product.get('product_name', 'N/A')}**")
                                        st.write(f"- 금융회사: {product.get('company', 'N/A')}")
                                        st.write(f"- 유사도: {product.get('similarity_score', 0):.3f}")
                                        st.write("---")
                                else:
                                    st.write("검색된 상품이 없습니다.")                        
                        else:
                            st.error(f"❌ 오류 발생: {result.get('error', '알 수 없는 오류')}")                      

                    except Exception as e:
                        st.error(f"❌ 시스템 오류: {str(e)}")
                        logger.error(f"Streamlit error: {e}")
            else:
                st.warning("⚠️ 질문을 입력해주세요.")
    
    with col2:
        st.header("📈 시스템 현황")
        try:
            rec_system = load_recommendation_system()
            st.success("🟢 시스템 정상")
            total_products = len(rec_system.vector_store.documents)
            st.metric(" < 총 상품 수 > ", f"{total_products:,}개")
            
            PRODUCT_TYPE_MAPPING = {
                'deposit': '정기예금',
                'saving': '적금', 
                'annuity': '연금저축',
                'mortgage': '주택담보대출',
                'rent': '전세자금대출',
                'credit': '개인신용대출'
            }
        
            product_counts = {}
            for doc in rec_system.vector_store.documents:
                ptype = doc['product_type']
                product_counts[ptype] = product_counts.get(ptype, 0) + 1
            
            st.subheader("📊 상품 분포")
            for ptype, count in product_counts.items():
                korean_name = PRODUCT_TYPE_MAPPING.get(ptype, ptype)
                st.write(f"- {korean_name}: {count}개")     
                
        except Exception as e:
            st.error("🔴 시스템 오류")
            st.error(str(e))
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        💰 금융상품 추천 시스템 | 금융감독원 금융상품통합비교공시 데이터 수집 
    """, unsafe_allow_html=True)

def manage_conversation_overflow():
    MAX_MESSAGES = MAX_CONVERSATION_TURNS * 2
    
    if len(st.session_state.conversation) >= MAX_MESSAGES:
        st.warning(f"⚠️ 대화가 {MAX_CONVERSATION_TURNS}턴을 초과했습니다.")

        management_option = st.selectbox(
            "대화 기록 관리 방식:",
            ["자동 삭제 (오래된 대화)", "수동 관리", "전체 초기화"],
            key="conversation_management"
        )
        
        if management_option == "자동 삭제 (오래된 대화)":
            # 최근 대화 70% 유지
            keep_count = int(MAX_MESSAGES * 0.7)
            st.session_state.conversation = st.session_state.conversation[-keep_count:]
            st.success(f"✅ 오래된 대화 {len(st.session_state.conversation) - keep_count}개를 삭제했습니다.")
        
        elif management_option == "전체 초기화":
            st.session_state.conversation = []
            st.success("✅ 모든 대화 기록을 초기화했습니다.")

def create_conversation_context(conversation: list, current_query: str) -> str:
    """이전 대화를 기반으로 컨텍스트 생성"""
    if not conversation:
        return current_query
    
    # 최근 3턴(6개 메시지)만 사용
    recent_conversation = conversation[-6:] if len(conversation) > 6 else conversation
    
    context_parts = []
    for i, msg in enumerate(recent_conversation):
        role = "사용자" if msg['role'] == 'user' else "시스템"
        context_parts.append(f"{role}: {msg['content'][:100]}...")  # 100자 제한
    
    context = "\n".join(context_parts)
    return f"[이전 대화]\n{context}\n\n[현재 질문]\n{current_query}"

if __name__ == "__main__":
    main()