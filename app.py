import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import re
import jieba

# ---------------- 页面配置 ----------------
st.set_page_config(
    page_title="课程推荐系统", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- 缓存配置 ----------------
@st.cache_resource
def load_resources():
    """加载模型和数据资源（缓存优化）"""
    try:
        # 加载TF-IDF模型
        tfidf = joblib.load(r'd:/选课系统/model/tfidf_model.pkl')
        # 加载课程-老师特征表
        course_teacher_tfidf = pd.read_pickle(r'd:/选课系统/data/course_teacher_tfidf.pkl')
        # 生成tfidf矩阵
        tfidf_matrix = tfidf.transform(course_teacher_tfidf['cleaned_review'])
        
        st.success("✅ 模型与数据加载成功")
        return tfidf, course_teacher_tfidf, tfidf_matrix
    except Exception as e:
        st.error(f"资源加载失败: {e}")
        return None, None, None

# ---------------- 会话状态初始化 ----------------
def init_session_state():
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.user_keywords = ""          # 用户输入的关键词
        st.session_state.new_courses = pd.DataFrame() # 上传的冷启动数据
        st.session_state.recommendations = pd.DataFrame() # 推荐结果
        st.session_state.current_step = 1            # 当前交互步骤（1:输入 2:推荐）

# ---------------- 核心功能函数 ----------------
def clean_text(text):
    """文本清洗函数（与预处理逻辑一致）"""
    with open(r'd:/选课系统/data/stopwords.txt', 'r', encoding='gbk') as f:
        stopwords = [line.strip() for line in f.readlines()]
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', '', str(text))
    text = text.lower()
    words = jieba.lcut(text)
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def handle_cold_start(new_course_teacher_df, tfidf, clean_text, keywords_tfidf, similarity_weight=0.7, rating_weight=0.3):
    """冷启动课程推荐函数"""
    new_course_teacher_df['替代文本'] = new_course_teacher_df['course_name'] + ' ' + new_course_teacher_df['teacher_name']
    new_course_teacher_df['cleaned_替代文本'] = new_course_teacher_df['替代文本'].apply(clean_text)
    new_tfidf_matrix = tfidf.transform(new_course_teacher_df['cleaned_替代文本'])
    similarity_scores = cosine_similarity(keywords_tfidf, new_tfidf_matrix).flatten()
    cold_start_results = new_course_teacher_df[['course_name', 'teacher_name']].copy()
    cold_start_results['similarity'] = similarity_scores
    cold_start_results['avg_rating'] = 0
    cold_start_results['composite_score'] = (
        similarity_weight * cold_start_results['similarity'] + 
        rating_weight * cold_start_results['avg_rating']
    )
    return cold_start_results

def recommend_courses(keywords, course_teacher_tfidf, tfidf, tfidf_matrix, top_n=5, similarity_weight=0.7, rating_weight=0.3):
    """常规课程推荐函数"""
    cleaned_keywords = clean_text(keywords)
    keywords_tfidf = tfidf.transform([cleaned_keywords])
    similarity_scores = cosine_similarity(keywords_tfidf, tfidf_matrix).flatten()
    
    temp_df = course_teacher_tfidf.copy()
    temp_df['similarity'] = similarity_scores
    temp_df['avg_rating'] = temp_df['avg_rating'].fillna(0)
    temp_df['composite_score'] = (
        similarity_weight * temp_df['similarity'] + 
        rating_weight * temp_df['avg_rating']
    )
    return temp_df[['course_name', 'teacher_name', 'composite_score']].sort_values(by='composite_score', ascending=False).head(top_n)

# ---------------- 界面展示函数 ----------------
def display_input_section(jokes_df):
    """显示用户输入区域"""
    st.header("📝 步骤1: 输入课程需求")
    
    # 关键词输入（保留，仅移除冷启动上传部分）
    st.session_state.user_keywords = st.text_input(
        "输入你的需求（如'数学 幽默 讲解清晰'）：",
        key="user_keywords_input"
    )
    
    # 移除原冷启动数据上传代码块（原L112-119）
    
    # 提交按钮（未修改）
    if st.button("获取推荐", type="primary", key="submit_recommend"):
        if not st.session_state.user_keywords:
            st.warning("请输入关键词！")
        else:
            st.session_state.current_step = 2
            st.rerun()

def final_recommend(keywords, course_teacher_tfidf, tfidf, tfidf_matrix, top_n=5):
    """简化后的推荐函数（仅常规推荐）"""
    # 直接返回常规推荐结果，不再合并冷启动
    return recommend_courses(keywords, course_teacher_tfidf, tfidf, tfidf_matrix, top_n)

def display_recommendation_section(tfidf, course_teacher_tfidf, tfidf_matrix):
    """显示推荐结果区域"""
    st.header("🎯 步骤2: 课程推荐结果")
    
    with st.spinner("正在生成推荐..."):
        # 调用时不再传递冷启动数据（st.session_state.new_courses）
        recommendations = final_recommend(
            st.session_state.user_keywords,
            course_teacher_tfidf,
            tfidf,
            tfidf_matrix
        )
        st.session_state.recommendations = recommendations
    
    if not st.session_state.recommendations.empty:
        st.subheader("为你推荐的课程：")
        st.dataframe(
            st.session_state.recommendations[['course_name', 'teacher_name', 'composite_score']],
            use_container_width=True
        )
    else:
        st.error("未找到符合条件的课程推荐")

# ---------------- 主函数 ----------------
def main():
    init_session_state()
    st.title("🎓 基于TF-IDF的课程推荐系统")
    st.markdown("### 智能匹配你的课程需求")
    
    # 侧边栏信息展示
    with st.sidebar:
        st.header("📌 系统信息")
        tfidf, course_teacher_tfidf, tfidf_matrix = load_resources()
        
        if tfidf and course_teacher_tfidf is not None and tfidf_matrix is not None:
            st.info(f"**模型类型**: TF-IDF文本相似度")
            st.info(f"**课程数量**: {len(course_teacher_tfidf):,}")
            st.info(f"**特征维度**: {tfidf_matrix.shape[1]}")
            st.info(f"**停用词数量**: {len(open(r'd:/选课系统/data/stopwords.txt', 'r', encoding='gbk').readlines()):,}")
            
            st.markdown("---")
            st.header("🔄 当前进度")
            progress = st.session_state.current_step / 2
            st.progress(progress)
            st.write(f"步骤 {st.session_state.current_step} / 2")
            
            st.markdown("---")
            st.success("✅ 已缓存模型与数据，响应更快速")
        else:
            st.error("❌ 系统初始化失败，请检查资源路径")
            return
    
    # 主界面根据步骤显示
    if st.session_state.current_step == 1:
        display_input_section(course_teacher_tfidf)
    elif st.session_state.current_step == 2:
        st.markdown("---")
        display_recommendation_section(tfidf, course_teacher_tfidf, tfidf_matrix)
    
    # 重新开始按钮
    if st.sidebar.button("🔄 重新开始"):
        st.session_state.current_step = 1
        st.session_state.user_keywords = ""
        st.session_state.new_courses = pd.DataFrame()
        st.session_state.recommendations = pd.DataFrame()
        st.rerun()

if __name__ == "__main__":
    main()