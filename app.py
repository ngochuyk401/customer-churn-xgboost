import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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

    # =====================================================
    # KHỞI TẠO SESSION STATE
    # =====================================================
    default_values = {
        "gender": "Nam",
        "senior": "Không",
        "partner": "Không",
        "dependents": "Không",
        "tenure": 12,
        "phone": "Có",
        "multiple_lines": "Không",
        "internet": "Cáp quang",
        "contract": "Theo tháng",
        "paperless": "Có",
        "online_security": "Không",
        "online_backup": "Không",
        "device_protection": "Không",
        "tech_support": "Không",
        "streaming_tv": "Không",
        "streaming_movies": "Không",
        "payment_method": "Hóa đơn điện tử",
        "monthly_charges": 70.0,
        "total_charges": 1000.0
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # =====================================================
    # TẢI FILE KHÁCH HÀNG
    # =====================================================
    with st.expander("📂 Tải thông tin khách hàng từ file", expanded=False):

        uploaded_customer = st.file_uploader(
            "Tải file JSON hoặc TXT",
            type=["json", "txt"],
            key="customer_upload"
        )

        if uploaded_customer:

            try:

                import json

                file_content = uploaded_customer.read().decode("utf-8")

                # ================= JSON =================
                if uploaded_customer.name.endswith(".json"):

                    customer_data = json.loads(file_content)

                # ================= TXT =================
                else:

                    customer_data = {}

                    for line in file_content.splitlines():

                        if ":" in line:
                            key, value = line.split(":", 1)
                            customer_data[key.strip()] = value.strip()

                # ================= CẬP NHẬT SESSION =================
                mapping_keys = {
                    "gender": "gender",
                    "SeniorCitizen": "senior",
                    "Partner": "partner",
                    "Dependents": "dependents",
                    "tenure": "tenure",
                    "PhoneService": "phone",
                    "MultipleLines": "multiple_lines",
                    "InternetService": "internet",
                    "Contract": "contract",
                    "PaperlessBilling": "paperless",
                    "OnlineSecurity": "online_security",
                    "OnlineBackup": "online_backup",
                    "DeviceProtection": "device_protection",
                    "TechSupport": "tech_support",
                    "StreamingTV": "streaming_tv",
                    "StreamingMovies": "streaming_movies",
                    "PaymentMethod": "payment_method",
                    "MonthlyCharges": "monthly_charges",
                    "TotalCharges": "total_charges"
                }

                for file_key, session_key in mapping_keys.items():

                    if file_key in customer_data:

                        value = customer_data[file_key]

                        # Ép kiểu số
                        if session_key in ["tenure"]:
                            value = int(value)

                        if session_key in ["monthly_charges", "total_charges"]:
                            value = float(value)

                        st.session_state[session_key] = value

                st.success("✅ Đã tải dữ liệu khách hàng thành công!")

                st.write("### 👀 Dữ liệu đã tải")
                st.json(customer_data)

            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")

    # =====================================================
    # FORM NHẬP LIỆU
    # =====================================================
    with st.expander("👤 Thông tin cá nhân & Dịch vụ cơ bản", expanded=True):

        c1 = st.columns(5)

        gender = c1[0].selectbox(
            "Giới tính",
            ["Nam", "Nữ"],
            index=["Nam", "Nữ"].index(st.session_state.gender)
        )

        senior = c1[1].selectbox(
            "Khách hàng cao tuổi",
            ["Không", "Có"],
            index=["Không", "Có"].index(st.session_state.senior)
        )

        partner = c1[2].selectbox(
            "Có người thân",
            ["Không", "Có"],
            index=["Không", "Có"].index(st.session_state.partner)
        )

        dependents = c1[3].selectbox(
            "Có người phụ thuộc",
            ["Không", "Có"],
            index=["Không", "Có"].index(st.session_state.dependents)
        )

        tenure = c1[4].number_input(
            "Thời gian sử dụng (tháng)",
            0,
            72,
            int(st.session_state.tenure)
        )

        c2 = st.columns(5)

        phone = c2[0].selectbox(
            "Dịch vụ điện thoại",
            ["Có", "Không"],
            index=["Có", "Không"].index(st.session_state.phone)
        )

        multiple_lines = c2[1].selectbox(
            "Nhiều đường dây",
            ["Không", "Có", "Không có DV điện thoại"],
            index=["Không", "Có", "Không có DV điện thoại"].index(st.session_state.multiple_lines)
        )

        internet = c2[2].selectbox(
            "Internet",
            ["Cáp quang", "DSL", "Không sử dụng"],
            index=["Cáp quang", "DSL", "Không sử dụng"].index(st.session_state.internet)
        )

        contract = c2[3].selectbox(
            "Loại hợp đồng",
            ["Theo tháng", "1 năm", "2 năm"],
            index=["Theo tháng", "1 năm", "2 năm"].index(st.session_state.contract)
        )

        paperless = c2[4].selectbox(
            "Hóa đơn điện tử",
            ["Có", "Không"],
            index=["Có", "Không"].index(st.session_state.paperless)
        )

    with st.expander("📦 Dịch vụ gia tăng & Thanh toán", expanded=True):

        c3 = st.columns(6)

        online_security = c3[0].selectbox(
            "Bảo mật trực tuyến",
            ["Không", "Có", "Không có Internet"],
            index=["Không", "Có", "Không có Internet"].index(st.session_state.online_security)
        )

        online_backup = c3[1].selectbox(
            "Sao lưu trực tuyến",
            ["Không", "Có", "Không có Internet"],
            index=["Không", "Có", "Không có Internet"].index(st.session_state.online_backup)
        )

        device_protection = c3[2].selectbox(
            "Bảo vệ thiết bị",
            ["Không", "Có", "Không có Internet"],
            index=["Không", "Có", "Không có Internet"].index(st.session_state.device_protection)
        )

        tech_support = c3[3].selectbox(
            "Hỗ trợ kỹ thuật",
            ["Không", "Có", "Không có Internet"],
            index=["Không", "Có", "Không có Internet"].index(st.session_state.tech_support)
        )

        streaming_tv = c3[4].selectbox(
            "Streaming TV",
            ["Không", "Có", "Không có Internet"],
            index=["Không", "Có", "Không có Internet"].index(st.session_state.streaming_tv)
        )

        streaming_movies = c3[5].selectbox(
            "Streaming Movies",
            ["Không", "Có", "Không có Internet"],
            index=["Không", "Có", "Không có Internet"].index(st.session_state.streaming_movies)
        )

        c4 = st.columns(3)

        payment_method = c4[0].selectbox(
            "Phương thức thanh toán",
            [
                "Hóa đơn điện tử",
                "Hóa đơn bưu điện",
                "Chuyển khoản ngân hàng",
                "Thẻ tín dụng"
            ],
            index=[
                "Hóa đơn điện tử",
                "Hóa đơn bưu điện",
                "Chuyển khoản ngân hàng",
                "Thẻ tín dụng"
            ].index(st.session_state.payment_method)
        )

        monthly_charges = c4[1].number_input(
            "Chi phí hàng tháng (USD)",
            0.0,
            200.0,
            float(st.session_state.monthly_charges),
            step=0.5
        )

        total_charges = c4[2].number_input(
            "Tổng chi phí (USD)",
            0.0,
            10000.0,
            float(st.session_state.total_charges),
            step=10.0
        )

    # =====================================================
    # NÚT CHỨC NĂNG
    # =====================================================
    left_btn, right_btn = st.columns(2)

    with left_btn:

        download_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        import json

        st.download_button(
            label="📥 Tải mẫu JSON",
            data=json.dumps(download_data, indent=4, ensure_ascii=False),
            file_name="customer_data.json",
            mime="application/json",
            use_container_width=True
        )

    with right_btn:

        predict_btn = st.button(
            "🔍 Dự đoán",
            type="primary",
            use_container_width=True
        )

    # =====================================================
    # DỰ ĐOÁN
    # =====================================================
    if predict_btn:

        input_data = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        from preprocessing.preprocess import preprocess_input

        processed_df = preprocess_input(input_data)

        model_files = {
            "XGBoost": "xgb_model.pkl",
            "Random Forest": "rf_model.pkl",
            "SVM": "svm_model.pkl"
        }

        model = joblib.load(model_files[model_name])

        proba = model.predict_proba(processed_df)[0][1]

        prediction = 1 if proba >= threshold else 0

        st.markdown("---")

        # ================= KẾT QUẢ =================
        if prediction == 1:

            st.error(
                f"⚠️ KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ "
                f"(Xác suất: {proba:.1%})"
            )

            st.info(
                "💡 Khuyến nghị: Gửi ưu đãi giữ chân "
                "và liên hệ chăm sóc khách hàng."
            )

        else:

            st.success(
                f"✅ KHÁCH HÀNG Ở LẠI "
                f"(Xác suất churn: {proba:.1%})"
            )

        # ================= HIỂN THỊ =================
        col_left, col_right = st.columns([2, 1])

        with col_left:

            with st.expander(
                "📋 Thông tin khách hàng đã nhập",
                expanded=True
            ):

                st.dataframe(
                    pd.DataFrame([input_data]),
                    use_container_width=True
                )

            with st.expander(
                "🔧 Dữ liệu sau tiền xử lý (30 cột)",
                expanded=False
            ):

                st.dataframe(
                    processed_df,
                    use_container_width=True
                )

        with col_right:

            fig_gauge = px.pie(
                values=[proba, 1 - proba],
                names=['Churn', 'Không churn'],
                color_discrete_sequence=['#ef4444', '#22c55e'],
                hole=0.7
            )

            fig_gauge.update_layout(
                title="Xác suất churn",
                height=320
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True
            )

        # ================= AI GIẢI THÍCH =================
        with st.expander(
            "🤖 AI Giải thích dự đoán",
            expanded=True
        ):

            explain_prompt = f"""
            Khách hàng có:
            - Thời gian sử dụng: {tenure} tháng
            - Hợp đồng: {contract}
            - Chi phí hàng tháng: {monthly_charges} USD
            - Tổng chi phí: {total_charges} USD
            - Phương thức thanh toán: {payment_method}

            Mô hình dự đoán xác suất churn = {proba:.1%}.

            Hãy giải thích ngắn gọn:
            - Vì sao mô hình dự đoán như vậy
            - Các yếu tố ảnh hưởng chính
            - Gợi ý giữ chân khách hàng
            """

            try:

                response = model_ai.generate_content(explain_prompt)

                st.markdown(response.text)

            except:

                st.write(
                    "Không thể lấy giải thích từ AI lúc này."
                )

# ===================== TAB 2: PHÂN TÍCH MÔ TẢ =====================
with tab2:
    st.subheader("📊 Phân tích mô tả")
    uploaded_file = st.file_uploader("Tải file CSV để phân tích", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.success(f"Đã tải: {uploaded_file.name} — {len(df)} dòng")

        # ===================== XEM DỮ LIỆU =====================
        with st.expander("👀 Xem trước dữ liệu thô"):
            st.dataframe(df.head(10), use_container_width=True)

        # ===================== THỐNG KÊ =====================
        st.subheader("📈 Thống kê mô tả")
        st.dataframe(df.describe(include='all'), use_container_width=True)

        # ===================== PHÂN LOẠI CỘT =====================
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # ===================== PHÂN TÍCH ĐƠN BIẾN =====================
        st.subheader("🔍 Phân tích đơn biến")

        col_type = st.radio(
            "Loại biến",
            ["Biến định lượng", "Biến định tính"],
            horizontal=True
        )

        if col_type == "Biến định lượng":

            if len(numeric_cols) == 0:
                st.warning("Không có biến định lượng trong dataset.")
            else:
                col = st.selectbox(
                    "Chọn biến định lượng",
                    numeric_cols,
                    key="num_single"
                )

                c1, c2 = st.columns(2)

                with c1:
                    fig_hist = px.histogram(
                        df,
                        x=col,
                        nbins=30,
                        title=f"Histogram - {col}"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with c2:
                    fig_box = px.box(
                        df,
                        y=col,
                        title=f"Boxplot - {col}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

        else:

            if len(cat_cols) == 0:
                st.warning("Không có biến định tính trong dataset.")
            else:
                col = st.selectbox(
                    "Chọn biến định tính",
                    cat_cols,
                    key="cat_single"
                )

                c1, c2 = st.columns(2)

                with c1:
                    fig_pie = px.pie(
                        df,
                        names=col,
                        title=f"Pie chart - {col}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    fig_bar = px.histogram(
                        df,
                        x=col,
                        title=f"Bar chart - {col}"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        # ===================== PHÂN TÍCH ĐA BIẾN =====================
        st.subheader("🔗 Phân tích đa biến")

        with st.expander("⚙️ Cấu hình phân tích đa biến", expanded=False):

            target_col = st.selectbox(
                "Chọn cột target (phân loại 2 nhóm)",
                df.columns
            )

        # ===================== KIỂM TRA TARGET =====================
        if df[target_col].nunique() != 2:

            st.warning(
                "⚠️ Cột target nên có đúng 2 nhóm giá trị "
                "(ví dụ: 0/1, Yes/No, True/False)."
            )

        else:

            option = st.selectbox(
                "Chọn loại biểu đồ",
                [
                    "Boxplot theo Target",
                    "Bar chart theo Target",
                    "Scatter plot",
                    "Heatmap tương quan"
                ],
                key="multivar"
            )

            # ===================== BOXPLOT =====================
            if option == "Boxplot theo Target":

                if len(numeric_cols) == 0:
                    st.warning("Không có biến định lượng.")
                else:

                    col = st.selectbox(
                        "Chọn biến định lượng",
                        numeric_cols,
                        key="box_target"
                    )

                    fig = px.box(
                        df,
                        x=target_col,
                        y=col,
                        color=target_col,
                        title=f"{col} theo {target_col}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # ===================== BAR CHART =====================
            elif option == "Bar chart theo Target":

                filtered_cat_cols = [
                    c for c in cat_cols if c != target_col
                ]

                if len(filtered_cat_cols) == 0:
                    st.warning("Không có biến định tính phù hợp.")
                else:

                    col = st.selectbox(
                        "Chọn biến định tính",
                        filtered_cat_cols,
                        key="bar_target"
                    )

                    fig = px.histogram(
                        df,
                        x=col,
                        color=target_col,
                        barmode="group",
                        title=f"{col} theo {target_col}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # ===================== SCATTER =====================
            elif option == "Scatter plot":

                if len(numeric_cols) < 2:
                    st.warning("Cần ít nhất 2 biến định lượng.")
                else:

                    x = st.selectbox(
                        "Trục X",
                        numeric_cols,
                        key="scatter_x"
                    )

                    y = st.selectbox(
                        "Trục Y",
                        [c for c in numeric_cols if c != x],
                        key="scatter_y"
                    )

                    fig = px.scatter(
                        df,
                        x=x,
                        y=y,
                        color=target_col,
                        title=f"{x} vs {y} theo {target_col}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # ===================== HEATMAP =====================
            else:

                if len(numeric_cols) < 2:
                    st.warning("Cần ít nhất 2 biến định lượng.")
                else:

                    corr = df[numeric_cols].corr()

                    fig = ff.create_annotated_heatmap(
                        z=corr.values.round(2),
                        x=list(corr.columns),
                        y=list(corr.columns),
                        colorscale='RdBu',
                        showscale=True
                    )

                    fig.update_layout(
                        title="Heatmap tương quan giữa các biến định lượng",
                        height=650
                    )

                    st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 3: SO SÁNH MÔ HÌNH =====================
with tab3:

    st.subheader("⚖️ So sánh mô hình")

    uploaded_compare = st.file_uploader(
        "Tải dataset (.csv) để so sánh",
        type="csv",
        key="compare_file"
    )

    if uploaded_compare:

        df_comp = pd.read_csv(uploaded_compare)

        target_col = st.selectbox(
            "Chọn cột Target",
            df_comp.columns
        )

        # =====================================================
        # KIỂM TRA TARGET
        # =====================================================
        unique_values = df_comp[target_col].dropna().unique()

        st.info(f"Số nhóm trong target: {len(unique_values)}")

        with st.expander("👀 Xem giá trị target"):
            st.write(unique_values)

        is_binary = len(unique_values) == 2

        if not is_binary:

            st.warning(
                "⚠️ Cột target phải là bài toán phân loại nhị phân "
                "(ví dụ: 0/1, Yes/No, True/False...)."
            )

        else:

            st.success("✅ Target hợp lệ cho bài toán phân loại nhị phân.")

            if st.button("So sánh", type="primary"):

                with st.spinner(
                    "Đang huấn luyện và đánh giá mô hình..."
                ):

                    # =====================================================
                    # TÁCH X VÀ y
                    # =====================================================
                    X = df_comp.drop(columns=[target_col]).select_dtypes(include=np.number)
                    y = df_comp[target_col]

                    # =====================================================
                    # CHUYỂN TARGET VỀ 0/1 NẾU CẦN
                    # =====================================================
                    if y.dtype == "object":

                        unique_sorted = sorted(y.unique())

                        mapping = {
                            unique_sorted[0]: 0,
                            unique_sorted[1]: 1
                        }

                        y = y.map(mapping)

                    # =====================================================
                    # CHIA TRAIN / TEST
                    # =====================================================
                    from sklearn.model_selection import train_test_split

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        stratify=y,
                        random_state=42
                    )

                    # =====================================================
                    # SCALE
                    # =====================================================
                    scaler = StandardScaler()

                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # =====================================================
                    # SMOTE
                    # =====================================================
                    smote = SMOTE(random_state=42)

                    X_train_res, y_train_res = smote.fit_resample(
                        X_train,
                        y_train
                    )

                    # =====================================================
                    # TÍNH scale_pos_weight CHO XGBOOST
                    # =====================================================
                    label_counts = y_train.value_counts()

                    scale_pos_weight = (
                        label_counts.iloc[0] / label_counts.iloc[1]
                    )

                    # =====================================================
                    # KHAI BÁO MÔ HÌNH
                    # =====================================================
                    models = {

                        "SVM": SVC(
                            kernel="linear",
                            probability=True,
                            cache_size=500
                        ),

                        "Random Forest": RandomForestClassifier(
                            n_estimators=100,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1
                        ),

                        "XGBoost": XGBClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            scale_pos_weight=scale_pos_weight,
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=-1
                        )
                    }

                    results = {}

                    fig_roc = px.line(
                        title="Đường cong ROC so sánh 3 mô hình"
                    )

                    # =====================================================
                    # TRAIN + ĐÁNH GIÁ
                    # =====================================================
                    for name, model in models.items():

                        try:

                            # ================= TRAIN =================
                            model.fit(X_train_res, y_train_res)

                            # ================= PREDICT =================
                            y_pred = model.predict(X_test)

                            y_prob = model.predict_proba(X_test)[:, 1]

                            # ================= METRICS =================
                            precision = precision_score(y_test, y_pred)

                            recall = recall_score(y_test, y_pred)

                            f1 = f1_score(y_test, y_pred)

                            roc_auc = roc_auc_score(y_test, y_prob)

                            results[name] = {
                                "Precision": precision,
                                "Recall": recall,
                                "F1-score": f1,
                                "ROC-AUC": roc_auc
                            }

                            # ================= ROC CURVE =================
                            fpr, tpr, _ = roc_curve(y_test, y_prob)

                            fig_roc.add_scatter(
                                x=fpr,
                                y=tpr,
                                mode="lines",
                                name=f"{name} (AUC={roc_auc:.3f})"
                            )

                        except Exception as e:

                            st.error(f"Lỗi với mô hình {name}: {e}")

                    # =====================================================
                    # HIỂN THỊ KẾT QUẢ
                    # =====================================================
                    if results:

                        result_df = pd.DataFrame(results).T

                        st.subheader("📋 Bảng kết quả")

                        st.dataframe(
                            result_df.style.highlight_max(axis=0),
                            use_container_width=True
                        )

                        # =====================================================
                        # BIỂU ĐỒ CỘT
                        # =====================================================
                        st.subheader("📊 Biểu đồ so sánh")

                        melted = result_df.reset_index().melt(
                            id_vars="index",
                            var_name="Metric",
                            value_name="Score"
                        )

                        fig_bar = px.bar(
                            melted,
                            x="Metric",
                            y="Score",
                            color="index",
                            barmode="group",
                            title="So sánh hiệu suất 3 mô hình"
                        )

                        st.plotly_chart(
                            fig_bar,
                            use_container_width=True
                        )

                        # =====================================================
                        # ROC CURVE
                        # =====================================================
                        fig_roc.add_shape(
                            type='line',
                            x0=0,
                            y0=0,
                            x1=1,
                            y1=1,
                            line=dict(
                                dash='dash',
                                color='gray'
                            )
                        )

                        fig_roc.update_layout(
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate"
                        )

                        st.plotly_chart(
                            fig_roc,
                            use_container_width=True
                        )

                        # =====================================================
                        # MÔ HÌNH TỐT NHẤT
                        # =====================================================
                        best_model = result_df["ROC-AUC"].idxmax()

                        st.success(
                            f"🏆 Mô hình có ROC-AUC cao nhất: {best_model}"
                        )

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