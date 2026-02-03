import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import các module cần thiết
try:
    from src.config import ADMISSION_FILE, ACADEMIC_RECORDS_FILE, TEST_FILE
    from src.data_loader import DataLoader
    from src.features import FeatureEngineer
    from src.evaluation import calculate_metrics
except ImportError as e:
    st.error(f"Lỗi import module: {e}")
    st.info("Vui lòng đảm bảo đã cài đặt đầy đủ thư viện và đúng cấu trúc thư mục")
    st.stop()


# Cấu hình trang
st.set_page_config(
    page_title="Learning Progress Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


# Cache data loading
@st.cache_data
def load_data():
    """Load và cache dữ liệu để tăng tốc"""
    try:
        loader = DataLoader(ADMISSION_FILE, ACADEMIC_RECORDS_FILE, TEST_FILE)
        loader.load_raw_data()
        loader.clean_data(is_test=False)
        df = loader.get_merged_data()
        
        # Tạo features
        engineer = FeatureEngineer()
        df_fe = engineer.create_features(df)
        
        return df_fe
    except FileNotFoundError as e:
        st.error(f"Không tìm thấy file dữ liệu: {e}")
        return None
    except Exception as e:
        st.error(f"Lỗi khi load dữ liệu: {e}")
        return None


# Header
st.markdown('<p class="main-header">📊 Learning Progress Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.image("https://via.placeholder.com/300x100.png?text=MULTOUR+TEAM", use_container_width=True)
st.sidebar.markdown("### ⚙️ Cài đặt Dashboard")

view_option = st.sidebar.selectbox(
    "Chọn chế độ xem",
    ["📈 Tổng quan", "👤 Phân tích sinh viên", "🎯 Hiệu suất model", "⚠️ Đánh giá rủi ro"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Hướng dẫn sử dụng")
with st.sidebar.expander("Xem hướng dẫn"):
    st.markdown("""
    - **Tổng quan**: Xem thống kê tổng thể
    - **Phân tích sinh viên**: Tra cứu thông tin sinh viên
    - **Hiệu suất model**: Upload predictions để đánh giá
    - **Đánh giá rủi ro**: Phát hiện sinh viên có nguy cơ
    """)

# Load dữ liệu
with st.spinner("Đang tải dữ liệu..."):
    df = load_data()

if df is None:
    st.error("Không thể load dữ liệu. Vui lòng kiểm tra lại file dữ liệu.")
    st.stop()


# ========== TỔNG QUAN ==========
if view_option == "📈 Tổng quan":
    st.header("📈 Tổng quan Dữ liệu")
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_students = df['MA_SO_SV'].nunique()
        st.metric("👥 Tổng số sinh viên", f"{total_students:,}")
    
    with col2:
        avg_credits = df['TC_DANGKY'].mean()
        st.metric("📚 TC đăng ký TB", f"{avg_credits:.1f}")
    
    with col3:
        avg_completed = df['TC_HOANTHANH'].mean()
        st.metric("✅ TC hoàn thành TB", f"{avg_completed:.1f}")
    
    with col4:
        completion_rate = (df['TC_HOANTHANH'].sum() / df['TC_DANGKY'].sum()) * 100
        delta = completion_rate - 80  # Giả sử mục tiêu 80%
        st.metric("📊 Tỷ lệ hoàn thành", f"{completion_rate:.1f}%", delta=f"{delta:+.1f}%")
    
    st.markdown("---")
    
    # Biểu đồ phân phối
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Phân phối Tín chỉ Đăng ký")
        fig1 = px.histogram(
            df, x='TC_DANGKY',
            nbins=30,
            title="",
            labels={'TC_DANGKY': 'Số tín chỉ', 'count': 'Số lượng'}
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("✅ Phân phối Tín chỉ Hoàn thành")
        fig2 = px.histogram(
            df, x='TC_HOANTHANH',
            nbins=30,
            title="",
            labels={'TC_HOANTHANH': 'Số tín chỉ', 'count': 'Số lượng'},
            color_discrete_sequence=['#2ecc71']
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # GPA & CPA Distribution
    st.subheader("📈 Phân phối Điểm số")
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.histogram(
            df, x='GPA',
            nbins=40,
            title="Phân phối GPA",
            labels={'GPA': 'Điểm GPA', 'count': 'Số lượng'},
            color_discrete_sequence=['#3498db']
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.histogram(
            df, x='CPA',
            nbins=40,
            title="Phân phối CPA",
            labels={'CPA': 'Điểm CPA', 'count': 'Số lượng'},
            color_discrete_sequence=['#9b59b6']
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Thống kê theo phương thức xét tuyển
    if 'PTXT' in df.columns:
        st.subheader("🎓 Thống kê theo Phương thức Xét tuyển")
        ptxt_stats = df.groupby('PTXT').agg({
            'MA_SO_SV': 'count',
            'TC_DANGKY': 'mean',
            'TC_HOANTHANH': 'mean',
            'GPA': 'mean',
            'CPA': 'mean'
        }).round(2)
        ptxt_stats.columns = ['Số lượng', 'TC ĐK TB', 'TC HT TB', 'GPA TB', 'CPA TB']
        ptxt_stats['Tỷ lệ HT (%)'] = ((ptxt_stats['TC HT TB'] / ptxt_stats['TC ĐK TB']) * 100).round(1)
        st.dataframe(ptxt_stats, use_container_width=True)


# ========== PHÂN TÍCH SINH VIÊN ==========
elif view_option == "👤 Phân tích sinh viên":
    st.header("👤 Phân tích Chi tiết Sinh viên")
    
    # Tìm kiếm sinh viên
    col1, col2 = st.columns([2, 1])
    with col1:
        student_id = st.text_input("🔍 Nhập mã số sinh viên:", placeholder="VD: 21120001")
    with col2:
        search_button = st.button("Tìm kiếm", type="primary")
    
    if student_id and search_button:
        student_data = df[df['MA_SO_SV'] == student_id].sort_values('semester_order')
        
        if len(student_data) > 0:
            st.success(f"✅ Tìm thấy sinh viên {student_id}")
            
            # Thông tin mới nhất
            latest = student_data.iloc[-1]
            
            st.subheader("📋 Thông tin hiện tại")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📅 Năm TS", latest.get('NAM_TUYENSINH', 'N/A'))
                st.metric("📊 Điểm TS", f"{latest.get('DIEM_TRUNGTUYEN', 0):.2f}")
            
            with col2:
                st.metric("📈 GPA", f"{latest.get('GPA', 0):.2f}")
                st.metric("📊 CPA", f"{latest.get('CPA', 0):.2f}")
            
            with col3:
                st.metric("📚 TC Đăng ký", int(latest.get('TC_DANGKY', 0)))
                st.metric("✅ TC Hoàn thành", int(latest.get('TC_HOANTHANH', 0)))
            
            with col4:
                completion_rate = (latest.get('TC_HOANTHANH', 0) / max(latest.get('TC_DANGKY', 1), 1)) * 100
                st.metric("📊 Tỷ lệ HT", f"{completion_rate:.1f}%")
                
                # Risk level
                if completion_rate < 50:
                    st.error("⚠️ Nguy cơ cao")
                elif completion_rate < 75:
                    st.warning("⚡ Nguy cơ trung bình")
                else:
                    st.success("✅ Ổn định")
            
            # Lịch sử học tập
            if len(student_data) > 1:
                st.subheader("📈 Xu hướng Học tập")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(student_data))),
                    y=student_data['GPA'],
                    mode='lines+markers',
                    name='GPA',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(student_data))),
                    y=student_data['CPA'],
                    mode='lines+markers',
                    name='CPA',
                    line=dict(color='#2ecc71', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    xaxis_title="Học kỳ",
                    yaxis_title="Điểm số",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Bảng chi tiết
                st.subheader("📊 Lịch sử Chi tiết")
                display_cols = ['HOC_KY', 'GPA', 'CPA', 'TC_DANGKY', 'TC_HOANTHANH']
                display_data = student_data[display_cols].copy()
                display_data['Tỷ lệ HT (%)'] = ((display_data['TC_HOANTHANH'] / display_data['TC_DANGKY']) * 100).round(1)
                st.dataframe(display_data, use_container_width=True)
            
        else:
            st.warning(f"❌ Không tìm thấy sinh viên {student_id}")
    
    # Phân tích theo nhóm
    st.markdown("---")
    st.subheader("📊 Phân tích theo Nhóm")
    
    segment_option = st.selectbox(
        "Chọn tiêu chí phân nhóm:",
        ["Phương thức xét tuyển (PTXT)", "Năm tuyển sinh", "Mức GPA"]
    )
    
    if segment_option == "Phương thức xét tuyển (PTXT)" and 'PTXT' in df.columns:
        segment_col = 'PTXT'
    elif segment_option == "Năm tuyển sinh" and 'NAM_TUYENSINH' in df.columns:
        segment_col = 'NAM_TUYENSINH'
    else:
        df['GPA_Level'] = pd.cut(
            df['GPA'],
            bins=[0, 2.0, 2.5, 3.0, 3.5, 4.0],
            labels=['Yếu (<2.0)', 'Trung bình (2.0-2.5)', 'Khá (2.5-3.0)', 'Giỏi (3.0-3.5)', 'Xuất sắc (>3.5)']
        )
        segment_col = 'GPA_Level'
    
    segment_stats = df.groupby(segment_col).agg({
        'TC_DANGKY': 'mean',
        'TC_HOANTHANH': 'mean',
        'MA_SO_SV': 'count'
    }).reset_index()
    segment_stats.columns = [segment_col, 'TC ĐK TB', 'TC HT TB', 'Số lượng']
    segment_stats['Tỷ lệ HT (%)'] = ((segment_stats['TC HT TB'] / segment_stats['TC ĐK TB']) * 100).round(2)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(segment_stats, use_container_width=True)
    
    with col2:
        fig = px.bar(
            segment_stats,
            x=segment_col,
            y='Tỷ lệ HT (%)',
            title=f"Tỷ lệ hoàn thành theo {segment_option}",
            color='Tỷ lệ HT (%)',
            color_continuous_scale='RdYlGn',
            text='Tỷ lệ HT (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)


# ========== HIỆU SUẤT MODEL ==========
elif view_option == "🎯 Hiệu suất model":
    st.header("🎯 Đánh giá Hiệu suất Model")
    
    st.info("📤 Upload file predictions để xem kết quả đánh giá model")
    
    uploaded_file = st.file_uploader(
        "Chọn file CSV chứa predictions",
        type=['csv'],
        help="File phải có 2 cột: MA_SO_SV và PRED_TC_HOANTHANH"
    )
    
    if uploaded_file is not None:
        try:
            predictions_df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['MA_SO_SV', 'PRED_TC_HOANTHANH']
            if not all(col in predictions_df.columns for col in required_cols):
                st.error(f"❌ File phải chứa các cột: {', '.join(required_cols)}")
            else:
                # Merge với actual values
                eval_df = df[['MA_SO_SV', 'TC_HOANTHANH', 'TC_DANGKY']].merge(
                    predictions_df,
                    on='MA_SO_SV',
                    how='inner'
                )
                
                if len(eval_df) == 0:
                    st.warning("⚠️ Không có MA_SO_SV nào khớp giữa predictions và dữ liệu thực tế")
                else:
                    y_true = eval_df['TC_HOANTHANH'].values
                    y_pred = eval_df['PRED_TC_HOANTHANH'].values
                    
                    # Tính metrics
                    metrics = calculate_metrics(y_true, y_pred)
                    
                    # Hiển thị metrics
                    st.subheader("📊 Kết quả Đánh giá")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    
                    with col2:
                        mae = np.mean(np.abs(y_true - y_pred))
                        st.metric("MAE", f"{mae:.4f}")
                    
                    with col3:
                        r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    # Scatter plot
                    st.subheader("📈 Predictions vs Actual")
                    
                    fig = px.scatter(
                        x=y_true,
                        y=y_pred,
                        labels={'x': 'TC thực tế', 'y': 'TC dự đoán'},
                        opacity=0.6
                    )
                    
                    # Perfect prediction line
                    min_val = min(y_true.min(), y_pred.min())
                    max_val = max(y_true.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Dự đoán hoàn hảo',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Error analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Phân phối Sai số")
                        errors = y_true - y_pred
                        fig_error = px.histogram(
                            x=errors,
                            nbins=50,
                            labels={'x': 'Sai số (Actual - Predicted)', 'y': 'Tần suất'}
                        )
                        fig_error.add_vline(x=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_error, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Thống kê Sai số")
                        error_stats = pd.DataFrame({
                            'Metric': ['Mean Error', 'Std Error', 'Min Error', 'Max Error', 'Median Error'],
                            'Value': [
                                errors.mean(),
                                errors.std(),
                                errors.min(),
                                errors.max(),
                                np.median(errors)
                            ]
                        })
                        error_stats['Value'] = error_stats['Value'].round(4)
                        st.dataframe(error_stats, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    eval_df['Error'] = eval_df['TC_HOANTHANH'] - eval_df['PRED_TC_HOANTHANH']
                    eval_df['Abs_Error'] = np.abs(eval_df['Error'])
                    
                    csv = eval_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download kết quả đánh giá",
                        data=csv,
                        file_name="model_evaluation_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý file: {e}")


# ========== ĐÁNH GIÁ RỦI RO ==========
elif view_option == "⚠️ Đánh giá rủi ro":
    st.header("⚠️ Đánh giá Rủi ro Sinh viên")
    
    # Tính completion rate và risk level
    df_risk = df.copy()
    df_risk['completion_rate'] = (df_risk['TC_HOANTHANH'] / df_risk['TC_DANGKY'] * 100).clip(0, 100)
    df_risk['risk_level'] = pd.cut(
        df_risk['completion_rate'],
        bins=[0, 50, 75, 90, 100],
        labels=['🔴 Nguy cơ cao', '🟠 Nguy cơ TB', '🟡 Nguy cơ thấp', '🟢 Ổn định']
    )
    
    # Tổng quan rủi ro
    st.subheader("📊 Tổng quan Phân bố Rủi ro")
    
    risk_counts = df_risk['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['Mức độ rủi ro', 'Số lượng']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(risk_counts, use_container_width=True)
        
        total = risk_counts['Số lượng'].sum()
        high_risk_count = risk_counts[risk_counts['Mức độ rủi ro'] == '🔴 Nguy cơ cao']['Số lượng'].values
        high_risk_pct = (high_risk_count[0] / total * 100) if len(high_risk_count) > 0 else 0
        
        st.metric("⚠️ Sinh viên nguy cơ cao", f"{high_risk_pct:.1f}%")
    
    with col2:
        fig = px.pie(
            risk_counts,
            values='Số lượng',
            names='Mức độ rủi ro',
            title="Phân bố mức độ rủi ro",
            color='Mức độ rủi ro',
            color_discrete_map={
                '🔴 Nguy cơ cao': '#e74c3c',
                '🟠 Nguy cơ TB': '#f39c12',
                '🟡 Nguy cơ thấp': '#f1c40f',
                '🟢 Ổn định': '#2ecc71'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Danh sách sinh viên nguy cơ cao
    st.markdown("---")
    st.subheader("🔴 Danh sách Sinh viên Nguy cơ Cao")
    
    high_risk = df_risk[df_risk['risk_level'] == '🔴 Nguy cơ cao'].copy()
    
    if len(high_risk) > 0:
        display_cols = ['MA_SO_SV', 'GPA', 'CPA', 'TC_DANGKY', 'TC_HOANTHANH', 'completion_rate']
        high_risk_display = high_risk[display_cols].sort_values('completion_rate')
        high_risk_display.columns = ['Mã SV', 'GPA', 'CPA', 'TC ĐK', 'TC HT', 'Tỷ lệ HT (%)']
        high_risk_display['Tỷ lệ HT (%)'] = high_risk_display['Tỷ lệ HT (%)'].round(1)
        
        st.dataframe(
            high_risk_display,
            use_container_width=True,
            height=400
        )
        
        # Download
        csv = high_risk_display.to_csv(index=False)
        st.download_button(
            label="📥 Download danh sách nguy cơ cao",
            data=csv,
            file_name="high_risk_students.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ Không có sinh viên nào ở mức nguy cơ cao!")
    
    # Phân tích yếu tố rủi ro
    st.markdown("---")
    st.subheader("📈 Phân tích Yếu tố Rủi ro")
    
    col1, col2 = st.columns(2)
    df_risk['GPA_group'] = pd.cut(
        df_risk['GPA'],
        bins=[0, 2.0, 2.5, 3.0, 4.0],
        labels=['<2.0', '2.0–2.5', '2.5–3.0', '>3.0']
    )

    with col1:
        st.markdown("#### Rủi ro theo GPA")
        gpa_risk = (
            df_risk
            .groupby('GPA_group')['risk_level']   # hoặc 'GPA' nếu bạn dùng trực tiếp
            .value_counts(normalize=True)
            .unstack()
            .fillna(0) * 100
        )

        gpa_risk.index = gpa_risk.index.astype(str)
        st.bar_chart(gpa_risk)

        gpa_risk.index = gpa_risk.index.astype(str)
        st.bar_chart(gpa_risk)
    
    with col2:
        if 'PTXT' in df_risk.columns:
            st.markdown("#### Rủi ro theo Phương thức XT")
            ptxt_risk = df_risk.groupby('PTXT')['risk_level'].value_counts(normalize=True).unstack().fillna(0) * 100
            st.bar_chart(ptxt_risk)
    
    # Khuyến nghị can thiệp
    st.markdown("---")
    st.subheader("💡 Khuyến nghị Can thiệp")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔴 Sinh viên Nguy cơ Cao
        - ✅ Hỗ trợ học tập cá nhân
        - ✅ Giảm tải tín chỉ học kỳ sau
        - ✅ Tư vấn học tập hàng tuần
        - ✅ Theo dõi sát sao tiến độ
        """)
    
    with col2:
        st.markdown("""
        ### 🟠 Sinh viên Nguy cơ Trung bình
        - ⚡ Cảnh báo sớm
        - ⚡ Chia sẻ tài liệu học tập
        - ⚡ Kết nối với bạn cố vấn
        - ⚡ Theo dõi định kỳ
        """)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d;'>
        <p>📊 Learning Progress Prediction Dashboard | 👥 Team Multour | 🏆 DATAFLOW 2026</p>
        <p>Powered by Streamlit & Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)