import pandas as pd

EXPECTED_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Chuyển input từ form (dict tiếng Việt) thành DataFrame dummy đúng với model.
    Mapping với selectbox trong app.py.
    """
    df = pd.DataFrame([data])

    # ---------------------
    # Gender
    # ---------------------
    df['gender_Male'] = 1 if df.loc[0, 'gender'] == 'Nam' else 0

    # ---------------------
    # Yes/No
    # ---------------------
    yes_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in yes_cols:
        df[f'{col}_Yes'] = 1 if df.loc[0, col] == 'Có' else 0

    # ---------------------
    # SeniorCitizen
    # ---------------------
    df['SeniorCitizen'] = 1 if df.loc[0, 'SeniorCitizen'] == 'Có' else 0

    # ---------------------
    # MultipleLines
    # ---------------------
    df['MultipleLines_Yes'] = 1 if df.loc[0, 'MultipleLines'] == 'Có' else 0
    df['MultipleLines_No phone service'] = 1 if df.loc[0, 'MultipleLines'] == 'Không có DV điện thoại' else 0

    # ---------------------
    # InternetService
    # ---------------------
    df['InternetService_Fiber optic'] = 1 if df.loc[0, 'InternetService'] == 'Cáp quang' else 0
    df['InternetService_No'] = 1 if df.loc[0, 'InternetService'] == 'Không' else 0

    # ---------------------
    # Các dịch vụ Internet
    # ---------------------
    internet_services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    for col in internet_services:
        value = df.loc[0, col]
        df[f'{col}_Yes'] = 1 if value == 'Có' else 0
        df[f'{col}_No internet service'] = 1 if value == 'Không có Internet' else 0
        # Nếu chọn "Không" → cả hai dummy = 0 → tương đương No (không có dịch vụ)

    # ---------------------
    # Contract
    # ---------------------
    df['Contract_One year'] = 1 if df.loc[0, 'Contract'] == '1 năm' else 0
    df['Contract_Two year'] = 1 if df.loc[0, 'Contract'] == '2 năm' else 0
    # "Theo tháng" → cả hai dummy = 0 → Month-to-month

    # ---------------------
    # PaymentMethod
    # ---------------------
    payment = df.loc[0, 'PaymentMethod']
    df['PaymentMethod_Electronic check'] = 1 if payment == 'Hóa đơn điện tử' else 0
    df['PaymentMethod_Mailed check'] = 1 if payment == 'Hóa đơn bưu điện' else 0
    df['PaymentMethod_Credit card (automatic)'] = 1 if payment == 'Thẻ tín dụng' else 0
    # Chuyển khoản ngân hàng → tất cả dummy = 0 → Bank transfer (automatic)

    # ---------------------
    # Các biến số
    # ---------------------
    df['tenure'] = df['tenure']
    df['MonthlyCharges'] = df['MonthlyCharges']
    df['TotalCharges'] = df['TotalCharges']

    # ---------------------
    # Giữ đúng các cột model
    # ---------------------
    df_final = df[EXPECTED_COLUMNS]

    return df_final

