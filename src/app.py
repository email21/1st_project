import streamlit as st
from typing import Dict, Any
from financial_rag_system import FinancialRecommendationSystem

# 페이지 설정
st.set_page_config(
    page_title="금융상품 추천 시스템",
    page_icon="💰",
    layout="wide"
)

MAX_CONVERSATION_TURNS = 10 # 사용자와 시스템 간 대화 최대 턴 수

@st.cache_resource
def load_recommendation_system():
    """추천 시스템 로드 (캐싱)"""
    return FinancialRecommendationSystem()

def main():
    
    # 세션 상태 초기화
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    # if 'conversation_count' not in st.session_state:
    #     st.session_state.conversation_count = 0
    
    st.title("💰 금융상품 추천 시스템")
    st.markdown("---")
    
    # 사이드바 (기존 유지)
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
            
            데이터 출처: 금융감독원
            금융상품통합비교공시
            """)            
        st.markdown("---")
        st.header("💬 대화 관리")
        if st.session_state.conversation:
            turn_count = len(st.session_state.conversation) // 2 # 현재 턴 수 계산
            st.write(f"대화 횟수: {turn_count}/{MAX_CONVERSATION_TURNS}")     
            if st.button("🗑️ 대화 기록 초기화", type="secondary"): # 초기화 버튼튼
                print(" ######## [DEBUG] 초기화 버튼 클릭됨 - Streamlit 대화 초기화 전:", st.session_state.conversation)
                st.session_state.conversation = []
                try:
                    rec_system = load_recommendation_system()
                    rec_system.reset_conversation()  # 백엔드 대화 기록도 초기화
                except Exception as e:
                    st.error(f"대화 기록 초기화 오류: {e}")
                st.rerun() # 초기화 후 페이지 재실행행
        else:
            st.write("대화 기록이 없습니다.")
        st.markdown("---")
        # st.header("🔧 개발자 도구")
        # st.session_state.show_debug = st.checkbox("분석 결과 JSON 표시", value=False)
    
        # if st.session_state.show_debug:
        #     st.info("디버그 모드: API 분석 결과를 JSON 형태로 표시합니다.")
             
    col_center, col_right = st.columns([2, 1]) # 메인 레이아웃: 중앙(채팅) : 오른쪽(시스템 현황) = 2:1
    
    # 중앙: 채팅 영역
    with col_center:
        st.header("💬 대화 기록")           
        chat_container = st.container()    # 대화 기록 컨테이너     
        with chat_container:
            # 기존 대화 기록 표시
            if st.session_state.conversation:
                with st.container(height=400):  # 스크롤 가능한 대화 기록
                    for i, message in enumerate(st.session_state.conversation):
                        if message["role"] == "user":
                            st.markdown(f"**▶️ 사용자:** {message['content']}")
                        else:
                            st.markdown(f"**▶️ 시스템:**")
                            st.markdown(message['content'])
                        st.markdown("---")
            else:
                st.info(" 금융상품에 대해 궁금한 점을 질문해보세요!")
        
        # 채팅 입력 영역
        if prompt := st.chat_input("금융상품에 대해 질문해보세요!"):
            print("############# [DEBUG] 입력 전 현재 전체 대화:", st.session_state.conversation)  # [1] Streamlit 대화 상태 확인
            manage_conversation_overflow()
            # 사용자 메시지 추가
            st.session_state.conversation.append({"role": "user", "content": prompt})
            
            # AI 응답 생성
            with st.spinner("추천 상품을 찾고 있습니다..."):
                try:
                    rec_system = load_recommendation_system()                   
                    rec_system.sync_conversation_history(st.session_state.conversation)  # 대화 기록 동기화 추가
                    print("########## [DEBUG] 동기화 직후 백엔드 대화 길이:", len(rec_system.conversation_history))  # [2] 백엔드에 잘 전달되는지 확인
                                  
                    result = rec_system.get_recommendation(prompt)
                                     
                    if result['status'] == 'success':
                        # AI 응답 추가
                        st.session_state.conversation.append({
                            "role": "assistant", 
                            "content": result['recommendation']
                        })
                        #st.session_state.conversation_count += 1                        
                    else:
                        error_msg = f"❌ 오류: {result.get('error', '알 수 없는 오류')}"
                        st.session_state.conversation.append({
                            "role": "assistant", 
                            "content": error_msg
                        })                        
                except Exception as e:
                    error_msg = f"❌ 시스템 오류: {str(e)}"
                    st.session_state.conversation.append({
                        "role": "assistant", 
                        "content": error_msg
                    })            
            st.rerun()

        st.header("💡 사용 팁")
        st.info("""
        **질문 예시:**
        - "3000만원을 2년간 안전하게 운용할 수 있는 정기예금 추천해주세요"
        - "20대 사회초년생 매월 50만원씩 2년간 넣을 수 있는 적금 상품 추천해주세요"
        - "40대 직장인이 노후 준비를 위해 매월 150만원씩 넣을 수 있는 연금저축 추천해주세요"
        - "50억원 아파트 구매를 위해 7억원 주택담보대출 받을 수 있는 상품 추천해주세요"
        - "10억원 전세 계약을 위해 5억원 전세자금대출 받을 수 있는 상품 추천해주세요"
        - "생활자금 마련을 위해 1000만원 개인신용대출 받을 수 있는 상품 추천해주세요"
            """)
        st.markdown("---")
            
        # 오른쪽: 시스템 현황
        with col_right:
            st.header("📈 시스템 현황")
            try:
                rec_system = load_recommendation_system()
                st.success("🟢 시스템 정상")
                
                # 총 상품 수 표시
                if hasattr(rec_system, 'documents'):
                    total_products = len(rec_system.documents)
                    st.metric("총 상품 수", f"{total_products:,}개")
                    
                    # 상품 분포 표시
                    PRODUCT_TYPE_MAPPING = {
                        'deposit': '정기예금',
                        'saving': '적금',
                        'annuity': '연금저축',
                        'mortgage': '주택담보대출',
                        'rent': '전세자금대출',
                        'credit': '개인신용대출'
                    }                    
                    product_counts = {}
                    for doc in rec_system.documents:
                        ptype = doc.get('product_type', '기타')
                        product_counts[ptype] = product_counts.get(ptype, 0) + 1                    
                    st.subheader("📊 상품 분포")
                    for ptype, count in product_counts.items():
                        korean_name = PRODUCT_TYPE_MAPPING.get(ptype, ptype)
                        st.write(f"- {korean_name}: {count}개")                
            except Exception as e:
                st.error("🔴 시스템 오류")
                st.error(str(e))
            st.markdown("---")

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

if __name__ == "__main__":
    main()