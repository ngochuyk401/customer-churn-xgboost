import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import google.generativeai as genai

# ==========================================
# CẤU HÌNH AI GEMINI
# ==========================================
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.warning("Chưa có API key, AI Assistant sẽ không hoạt động")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

system_instruction = """
Bạn là trợ lý AI thông minh hỗ trợ hệ thống dự báo khách hàng rời bỏ dịch vụ viễn thông.
Nhiệm vụ: Giải thích kết quả dự đoán, phân tích nguy cơ churn, gợi ý cách giữ chân khách hàng và trả lời thắc mắc về dữ liệu, mô hình.

Quy tắc:
- Trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu, thân thiện và chuyên nghiệp.
- Sử dụng gạch đầu dòng khi liệt kê.
- Luôn có mục "**Bước tiếp theo nên làm**" (3–5 ý cụ thể, thực tế).
- Giải thích rằng xác suất churn chỉ là ước lượng dựa trên dữ liệu, không phải kết quả tuyệt đối.
- Khi thiếu thông tin, hỏi tối đa 3 câu rõ ràng.
- Tập trung vào giá trị thực tế cho doanh nghiệp viễn thông (cách giảm churn, giữ chân khách hàng...).
"""

model_ai = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=system_instruction
)

# ==========================================
# CẤU HÌNH STREAMLIT
# ==========================================
st.set_page_config(page_title="Dự đoán khách hàng rời bỏ dịch vụ viễn thông", page_icon="📡", layout="wide")

BASE_DIR = Path(__file__).parent
feature_names = joblib.load(BASE_DIR / "feature_names.pkl")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3059/3059446.png", width=100)
st.sidebar.header("⚙️ Cấu hình hệ thống")
model_name = st.sidebar.selectbox("Chọn mô hình dự báo", ["XGBoost", "Random Forest", "SVM"])
threshold = st.sidebar.slider("Ngưỡng quyết định (Threshold)", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("📘 Hướng dẫn sử dụng")
st.sidebar.info("""
1. Chọn mô hình và điều chỉnh ngưỡng ở sidebar.
2. Nhập thông tin khách hàng → Nhấn **Dự đoán**.
3. Xem phân tích dữ liệu ở tab "Phân tích mô tả".
4. So sánh các mô hình ở tab "So sánh mô hình".
5. Hỏi trợ lý AI bất kỳ thắc mắc nào về dự báo churn và giữ chân khách hàng.
""")

# ==========================================
# TITLE
# ==========================================
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📡 Dự đoán khách hàng rời bỏ dịch vụ viễn thông</h1>", unsafe_allow_html=True)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dự đoán", "📊 Phân tích mô tả", "⚖️ So sánh mô hình", "🤖 Trợ lý AI"])

# ===================== TAB 1: DỰ ĐOÁN =====================
with tab1:
    st.subheader("🧾 Nhập thông tin khách hàng")

    with st.expander("👤 Thông tin cá nhân & Dịch vụ cơ bản", expanded=True):
        c1 = st.columns(5)
        gender = c1[0].selectbox("Giới tính", ["Nam", "Nữ"])
        senior = c1[1].selectbox("Khách hàng cao tuổi", ["Không", "Có"])
        partner = c1[2].selectbox("Có người thân", ["Không", "Có"])
        dependents = c1[3].selectbox("Có người phụ thuộc", ["Không", "Có"])
        tenure = c1[4].number_input("Thời gian sử dụng (tháng)", 0, 72, 12)

        c2 = st.columns(5)
        phone = c2[0].selectbox("Dịch vụ điện thoại", ["Có", "Không"])
        multiple_lines = c2[1].selectbox("Nhiều đường dây", ["Không", "Có", "Không có DV điện thoại"])
        internet = c2[2].selectbox("Internet", ["Cáp quang", "DSL", "Không sử dụng"])
        contract = c2[3].selectbox("Loại hợp đồng", ["Theo tháng", "1 năm", "2 năm"])
        paperless = c2[4].selectbox("Hóa đơn điện tử", ["Có", "Không"])

    with st.expander("📦 Dịch vụ gia tăng & Thanh toán", expanded=True):
        c3 = st.columns(6)
        online_security = c3[0].selectbox("Bảo mật trực tuyến", ["Không", "Có", "Không có Internet"])
        online_backup = c3[1].selectbox("Sao lưu trực tuyến", ["Không", "Có", "Không có Internet"])
        device_protection = c3[2].selectbox("Bảo vệ thiết bị", ["Không", "Có", "Không có Internet"])
        tech_support = c3[3].selectbox("Hỗ trợ kỹ thuật", ["Không", "Có", "Không có Internet"])
        streaming_tv = c3[4].selectbox("Streaming TV", ["Không", "Có", "Không có Internet"])
        streaming_movies = c3[5].selectbox("Streaming Movies", ["Không", "Có", "Không có Internet"])

        c4 = st.columns(3)
        payment_method = c4[0].selectbox("Phương thức thanh toán",
            ["Hóa đơn điện tử", "Hóa đơn bưu điện", "Chuyển khoản ngân hàng", "Thẻ tín dụng"])
        monthly_charges = c4[1].number_input("Chi phí hàng tháng (USD)", 0.0, 200.0, 70.0, step=0.5)
        total_charges = c4[2].number_input("Tổng chi phí (USD)", 0.0, 10000.0, 1000.0, step=10.0)

    if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
        input_data = {
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple_lines,
            'InternetService': internet, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }

        from preprocessing.preprocess import preprocess_input
        processed_df = preprocess_input(input_data)

        model_files = {"XGBoost": "xgb_model.pkl", "Random Forest": "rf_model.pkl", "SVM": "svm_model.pkl"}
        model = joblib.load(model_files[model_name])

        proba = model.predict_proba(processed_df)[0][1]
        prediction = 1 if proba >= threshold else 0

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ** (Xác suất: {proba:.1%})")
            st.info("💡 Khuyến nghị: Gửi ưu đãi giữ chân, liên hệ chăm sóc khách hàng khẩn cấp.")
        else:
            st.success(f"✅ **KHÁCH HÀNG Ở LẠI** (Xác suất churn: {proba:.1%})")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            with st.expander("📋 Thông tin khách hàng đã nhập", expanded=True):
                st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
            with st.expander("🔧 Dữ liệu sau tiền xử lý (30 cột)", expanded=False):
                st.dataframe(processed_df, use_container_width=True)

        with col_right:
            fig_gauge = px.pie(values=[proba, 1-proba], names=['Churn', 'Không churn'],
                               color_discrete_sequence=['#ef4444', '#22c55e'], hole=0.7)
            fig_gauge.update_layout(title="Xác suất churn", height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with st.expander("🤖 AI Giải thích dự đoán", expanded=True):
            explain_prompt = f"""
            Khách hàng có: thời gian sử dụng {tenure} tháng, hợp đồng {contract}, 
            chi phí tháng {monthly_charges} USD, tổng chi phí {total_charges} USD, 
            phương thức thanh toán {payment_method}.
            Mô hình dự đoán xác suất churn = {proba:.1%}.
            Giải thích ngắn gọn lý do và các yếu tố chính ảnh hưởng đến kết quả này.
            """
            try:
                response = model_ai.generate_content(explain_prompt)
                st.markdown(response.text)
            except:
                st.write("Không thể lấy giải thích từ AI lúc này.")

# ===================== TAB 2: PHÂN TÍCH MÔ TẢ =====================
with tab2:
    st.subheader("📊 Phân tích mô tả")
    uploaded_file = st.file_uploader("Tải file CSV để phân tích", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Đã tải: {uploaded_file.name} — {len(df)} dòng")

        with st.expander("👀 Xem trước dữ liệu thô"):
            st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📈 Thống kê mô tả")
        st.dataframe(df.describe(include='all'), use_container_width=True)

        is_ptmt = 'Churn' in df.columns and df['Churn'].dtype == 'object'

        st.subheader("🔍 Phân tích đơn biến")
        col_type = st.radio("Loại biến", ["Biến định lượng", "Biến định tính"], horizontal=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if col_type == "Biến định lượng":
            col = st.selectbox("Chọn biến định lượng", numeric_cols, key="num_single")
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histogram - {col}"), use_container_width=True)
            with c2: st.plotly_chart(px.box(df, y=col, title=f"Boxplot - {col}"), use_container_width=True)
        else:
            col = st.selectbox("Chọn biến định tính", cat_cols, key="cat_single")
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(px.pie(df, names=col, title=f"Pie chart - {col}"), use_container_width=True)
            with c2: st.plotly_chart(px.histogram(df, x=col, title=f"Bar chart - {col}"), use_container_width=True)

        if is_ptmt:
            st.subheader("🔗 Phân tích đa biến theo Churn")
            option = st.selectbox("Chọn loại biểu đồ",
                ["Boxplot theo Churn", "Bar chart theo Churn", "Scatter plot", "Heatmap tương quan"], key="multivar")

            if option == "Boxplot theo Churn":
                col = st.selectbox("Chọn biến định lượng", numeric_cols, key="box_churn")
                fig = px.box(df, x="Churn", y=col, color="Churn", title=f"{col} theo Churn")
                st.plotly_chart(fig, use_container_width=True)
            elif option == "Bar chart theo Churn":
                col = st.selectbox("Chọn biến định tính", [c for c in cat_cols if c != "Churn"], key="bar_churn")
                fig = px.histogram(df, x=col, color="Churn", barmode="group", title=f"{col} theo Churn")
                st.plotly_chart(fig, use_container_width=True)
            elif option == "Scatter plot":
                x = st.selectbox("Trục X", numeric_cols, key="scatter_x")
                y = st.selectbox("Trục Y", [c for c in numeric_cols if c != x], key="scatter_y")
                fig = px.scatter(df, x=x, y=y, color="Churn", title=f"{x} vs {y} theo Churn")
                st.plotly_chart(fig, use_container_width=True)
            else:
                corr = df[numeric_cols].corr()
                fig = ff.create_annotated_heatmap(
                    z=corr.values.round(2),
                    x=list(corr.columns),
                    y=list(corr.columns),
                    colorscale='RdBu',
                    showscale=True
                )
                fig.update_layout(title="Heatmap tương quan giữa các biến định lượng", height=650)
                st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 3: SO SÁNH MÔ HÌNH =====================
with tab3:
    st.subheader("⚖️ So sánh mô hình")
    uploaded_compare = st.file_uploader("Tải dataset (.csv) để so sánh", type="csv", key="compare_file")

    if uploaded_compare:
        df_comp = pd.read_csv(uploaded_compare)
        target_col = st.selectbox("Chọn cột Target (0/1)", df_comp.columns)

        if st.button("So sánh", type="primary"):
            with st.spinner("Đang huấn luyện và đánh giá mô hình bằng kiểm định chéo 10-fold... Vui lòng chờ một chút"):
                X = df_comp.drop(columns=[target_col]).select_dtypes(include=np.number)
                y = df_comp[target_col]

                if X.shape[1] != 30:
                    st.warning("⚠️ Dataset có ít cột số hơn mô hình được huấn luyện. "
                               "Kết quả có thể không chính xác. Nên dùng file tiền xử lý đã dùng để huấn luyện mô hình.")

                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                model_files = {"XGBoost": "xgb_model.pkl", "Random Forest": "rf_model.pkl", "SVM": "svm_model.pkl"}

                results = {}
                roc_data = {}

                for name, fname in model_files.items():
                    try:
                        model = joblib.load(fname)
                        precision_list, recall_list, f1_list, auc_list = [], [], [], []
                        fpr_list, tpr_list = [], []

                        for train_idx, test_idx in skf.split(X, y):
                            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)

                            smote = SMOTE(random_state=42)
                            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

                            model.fit(X_train_res, y_train_res)
                            y_pred = model.predict(X_test)
                            y_prob = model.predict_proba(X_test)[:, 1]

                            precision_list.append(precision_score(y_test, y_pred))
                            recall_list.append(recall_score(y_test, y_pred))
                            f1_list.append(f1_score(y_test, y_pred))
                            auc_list.append(roc_auc_score(y_test, y_prob))

                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            fpr_list.append(fpr)
                            tpr_list.append(tpr)

                        results[name] = {
                            "Precision": np.mean(precision_list),
                            "Recall": np.mean(recall_list),
                            "F1-score": np.mean(f1_list),
                            "ROC-AUC": np.mean(auc_list)
                        }
                        roc_data[name] = (fpr_list, tpr_list)

                    except Exception as e:
                        st.error(f"Lỗi với mô hình {name}: {e}")

                if results:
                    result_df = pd.DataFrame(results).T
                    st.dataframe(result_df.style.highlight_max(axis=0), use_container_width=True)

                    # Biểu đồ cột Mean
                    melted = result_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
                    fig_bar = px.bar(melted, x="Metric", y="Score", color="index", barmode="group",
                                     title="So sánh hiệu suất trung bình của 3 mô hình")
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # ROC Curve
                    fig_roc = px.line(title="Đường cong ROC so sánh 3 mô hình")
                    for name, (fpr_list, tpr_list) in roc_data.items():
                        mean_fpr = np.linspace(0, 1, 100)
                        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
                        fig_roc.add_scatter(x=mean_fpr, y=mean_tpr, name=f"{name} (AUC={results[name]['ROC-AUC']:.3f})")
                    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='gray'))
                    st.plotly_chart(fig_roc, use_container_width=True)

# ===================== TAB 4: TRỢ LÝ AI =====================
with tab4:
    st.subheader("🤖 Trợ lý AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Nút xóa hội thoại
    if st.button("🗑️ Xoá hội thoại"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("💡 Gợi ý câu hỏi nhanh"):
        st.write("- Tại sao khách hàng này có nguy cơ churn cao?")
        st.write("- Làm thế nào để giảm tỷ lệ khách hàng rời bỏ dịch vụ?")
        st.write("- Các yếu tố quan trọng nhất ảnh hưởng đến churn là gì?")
        st.write("- Giải thích xác suất churn từ mô hình nghĩa là gì?")

    # Thanh nhập luôn ở dưới cùng
    if prompt := st.chat_input("Hỏi trợ lý AI về dự báo churn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                try:
                    response = model_ai.generate_content(prompt)
                    ai_text = response.text
                    st.markdown(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                except Exception as e:
                    if "429" in str(e):
                        st.warning("Bạn đã dùng hết quota tạm thời. Vui lòng đợi một chút rồi thử lại.")
                    else:
                        st.error(f"Lỗi: {e}")

st.caption("Hệ thống dự đoán khách hàng rời bỏ dịch vụ viễn thông")