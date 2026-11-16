import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# --- 기본 설정 ---
st.set_page_config(
    page_title="🏢 건물 에너지 데이터 분석 및 예측 시스템",
    page_icon="⚡",
    layout="wide",
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# 데이터 로드 함수 (캐시 적용)
@st.cache_data
def load_data():
  return pd.read_csv("Energy_consumption.csv")


try:
  df_raw = load_data()
except FileNotFoundError:
  st.error(
      "⚠️ 'Energy_consumption.csv' 파일을 찾을 수 없습니다. 스크립트와 동일한"
      " 폴더에 파일을 위치시켜 주세요."
  )
  st.stop()

# --- 사이드바 메뉴 구성 ---
st.sidebar.title("🧭 분석 및 모델링 단계")
menu = st.sidebar.radio(
    "이동할 단계를 선택하세요:",
    [
        "1단계: 결측치 확인",
        "2단계: 이상치 분석 및 시각화",
        "3단계: 이상치 정제(클리핑)",
        "4단계: EDA 및 독립변수 선정 근거",
        "5단계: 독립 변수 제거 타당성 검증 (특징 선택)",
        "6단계: 실시간 예측 시뮬레이터",
        "7단계: 다중 선형 회귀 모델 평가",
    ],
)

num_cols = [
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "RenewableEnergy",
    "EnergyConsumption",
]
cat_cols = ["HVACUsage", "LightingUsage", "DayOfWeek", "Holiday"]


# =====================================================================
# [1단계] 결측치 확인
# =====================================================================
if menu == "1단계: 결측치 확인":
  st.title("⚡ 건물 에너지 데이터 전처리 - 1단계: 결측치 확인")
  st.write(
      "업로드된 'Energy_consumption.csv' 파일에서 결측치(빈칸)를 확인합니다."
  )

  null_counts = df_raw.isnull().sum().reset_index()
  null_counts.columns = ["컬럼명", "결측치 개수"]
  total_nulls = df_raw.isnull().sum().sum()

  st.subheader("📊 컬럼별 결측치 현황 표")
  st.dataframe(null_counts, use_container_width=True)
  st.info(f"💡 데이터셋 내 총 결측치(빈칸) 개수는 **{total_nulls}개**입니다.")


# =====================================================================
# [2단계] 이상치 분석 및 시각화
# =====================================================================
elif menu == "2단계: 이상치 분석 및 시각화":
  st.title("📊 건물 에너지 데이터 전처리 - 2단계: 이상치 확인")
  st.write("IQR 공식을 사용하여 각 수치형 변수별 이상치를 확인합니다.")

  outlier_counts = {}
  total_outliers = 0

  for col in num_cols:
    Q1 = df_raw[col].quantile(0.25)
    Q3 = df_raw[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cnt = ((df_raw[col] < lower_bound) | (df_raw[col] > upper_bound)).sum()
    outlier_counts[col] = cnt
    total_outliers += cnt

  outlier_df = pd.DataFrame(
      list(outlier_counts.items()), columns=["컬럼명", "이상치 개수"]
  )

  col1, col2 = st.columns([2, 1])
  with col1:
    st.dataframe(outlier_df, use_container_width=True)
  with col2:
    st.info(f"📌 **총 이상치 개수**: **{total_outliers}개**")

  st.subheader("📦 주요 변수 상자 그림 (Boxplot)")
  fig, ax = plt.subplots(figsize=(10, 4))
  sns.boxplot(
      data=df_raw[["Temperature", "Humidity", "Occupancy", "SquareFootage"]],
      ax=ax,
      palette="Set2",
  )
  st.pyplot(fig)


# =====================================================================
# [3단계] 이상치 정제(클리핑)
# =====================================================================
elif menu == "3단계: 이상치 정제(클리핑)":
  st.title("✨ 건물 에너지 데이터 전처리 - 3단계: 이상치 정제")
  st.write("IQR 클리핑을 적용하여 정제된 데이터를 생성합니다.")

  df_processed = df_raw.copy()
  for col in num_cols:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

  st.session_state["df_cleaned"] = df_processed
  st.success("🎉 이상치 정제(클리핑) 완료 및 세션 저장 완료!")

  csv_data = df_processed.to_csv(index=False).encode("utf-8-sig")
  st.download_button(
      label="📥 정제된 데이터 파일 다운로드 (Energy_consumption_cleaned.csv)",
      data=csv_data,
      file_name="Energy_consumption_cleaned.csv",
      mime="text/csv",
  )
  st.dataframe(df_processed.head(), use_container_width=True)


# =====================================================================
# [4단계] EDA 및 독립변수 선정 근거
# =====================================================================
elif menu == "4단계: EDA 및 독립변수 선정 근거":
  st.title("📊 탐색적 데이터 분석(EDA) 및 독립변수 선정 근거")
  df_eda = st.session_state.get("df_cleaned", df_raw)

  st.subheader("📈 수치형 변수 vs 에너지 소비량 산점도")
  for col in ["Temperature", "SquareFootage", "Humidity", "RenewableEnergy"]:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df_eda,
        x=col,
        y="EnergyConsumption",
        ax=ax,
        alpha=0.6,
        color="dodgerblue",
    )
    sns.regplot(
        data=df_eda,
        x=col,
        y="EnergyConsumption",
        scatter=False,
        ax=ax,
        color="crimson",
    )
    plt.title(f"{col} vs EnergyConsumption")
    st.pyplot(fig)


# =====================================================================
# [5단계] 독립 변수 제거 타당성 검증 (특징 선택)
# =====================================================================
elif menu == "5단계: 독립 변수 제거 타당성 검증 (특징 선택)":
  st.title("🔍 5단계: 상관계수가 낮은 독립 변수 제거 타당성 검증")
  st.write(
      "전체 변수를 모두 사용한 **기준 모델**과, 시각화 상 상관관계가 낮거나"
      " 의심되는 변수들(`SquareFootage`, `Humidity`, `RenewableEnergy`,"
      " `DayOfWeek`, `Holiday`)을 **하나씩 제외한 모델들**의 성능(MAE, RMSE,"
      " $R^2$)을 비교하여 제거 타당성을 검증합니다."
  )

  df_val = st.session_state.get("df_cleaned", df_raw)

  X_raw = df_val.drop(columns=["EnergyConsumption"])
  y = df_val["EnergyConsumption"]

  X_train_raw, X_test_raw, y_train, y_test = train_test_split(
      X_raw, y, test_size=0.2, random_state=42
  )

  scenarios = {
      "1. 기준 모델 (전체 변수 사용)": list(X_raw.columns),
      "2. 사용면적(SquareFootage) 제외": [
          c for c in X_raw.columns if c != "SquareFootage"
      ],
      "3. 습도(Humidity) 제외": [c for c in X_raw.columns if c != "Humidity"],
      "4. 신재생에너지(RenewableEnergy) 제외": [
          c for c in X_raw.columns if c != "RenewableEnergy"
      ],
      "5. 요일(DayOfWeek) 제외": [c for c in X_raw.columns if c != "DayOfWeek"],
      "6. 휴일(Holiday) 제외": [c for c in X_raw.columns if c != "Holiday"],
  }

  results = []
  for name, cols in scenarios.items():
    X_tr_sub = X_train_raw[cols]
    X_te_sub = X_test_raw[cols]

    X_tr_encoded = pd.get_dummies(X_tr_sub, drop_first=True)
    X_te_encoded = pd.get_dummies(X_te_sub, drop_first=True)
    X_te_encoded = X_te_encoded.reindex(
        columns=X_tr_encoded.columns, fill_value=0
    )

    model = LinearRegression()
    model.fit(X_tr_encoded, y_train)
    y_pred = model.predict(X_te_encoded)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append(
        {
            "모델 시나리오": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R² Score": round(r2, 4),
        }
    )

  result_df = pd.DataFrame(results)
  st.subheader("📊 모델별 예측 성능 비교 표 (MAE, RMSE, R²)")
  st.dataframe(result_df, use_container_width=True)

  # 시뮬레이션 연동을 위해 여기서도 기본 모델 학습 정보를 세션에 미리 저장해둠
  X_encoded_full = pd.get_dummies(X_raw, drop_first=True)
  full_model = LinearRegression()
  full_model.fit(X_train_raw.pipe(lambda d: pd.get_dummies(d, drop_first=True).reindex(columns=X_encoded_full.columns, fill_value=0)), y_train)
  st.session_state["trained_model"] = full_model
  st.session_state["X_encoded_columns"] = X_encoded_full.columns
  st.session_state["original_X"] = X_raw

  st.subheader("📈 변수 제거 전후 예측 정확도($R^2$) 비교 시각화")
  fig, ax = plt.subplots(figsize=(10, 5))
  sns.barplot(
      data=result_df,
      x="R² Score",
      y="모델 시나리오",
      palette="Blues_r",
      ax=ax,
  )
  ax.set_xlim(
      result_df["R² Score"].min() - 0.002, result_df["R² Score"].max() + 0.002
  )
  for index, value in enumerate(result_df["R² Score"]):
    ax.text(value, index, f" {value:.4f}", color="black", va="center")
  st.pyplot(fig)


# =====================================================================
# [6단계] 실시간 예측 시뮬레이터
# =====================================================================
elif menu == "6단계: 실시간 예측 시뮬레이터":
  st.title("🎛️ 6단계: 실시간 건물 에너지 소비량 예측 시뮬레이터")
  st.write(
      "환경 조건(슬라이더)과 설비/일정 조건(셀렉트박스)을 직접 조작하여 예상"
      " 에너지 소비량(kWh)을 실시간으로 확인합니다."
  )

  df_sim = st.session_state.get("df_cleaned", df_raw)
  X_full = df_sim.drop(columns=["EnergyConsumption"])
  X_encoded_full = pd.get_dummies(X_full, drop_first=True)

  # 세션에 모델이 없으면 여기서 즉시 학습
  if "trained_model" not in st.session_state:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded_full, df_sim["EnergyConsumption"], test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.session_state["trained_model"] = model
    st.session_state["X_encoded_columns"] = X_encoded_full.columns
    st.session_state["original_X"] = X_full

  model = st.session_state["trained_model"]
  train_columns = st.session_state["X_encoded_columns"]
  orig_X = st.session_state["original_X"]

  user_input = {}
  st.subheader("🔧 수치형 환경 변수 설정")
  num_input_cols = orig_X.select_dtypes(include=[np.number]).columns
  for col in num_input_cols:
    user_input[col] = st.slider(
        f"{col}",
        min_value=float(df_sim[col].min()),
        max_value=float(df_sim[col].max()),
        value=float(df_sim[col].mean()),
    )

  st.subheader("🏷️ 범주형 설비 및 일정 변수 설정")
  cat_input_cols = orig_X.select_dtypes(
      include=["object", "category"]
  ).columns
  for col in cat_input_cols:
    unique_vals = df_sim[col].dropna().unique().tolist()
    user_input[col] = st.selectbox(f"{col}", options=unique_vals)

  input_df = pd.DataFrame([user_input])
  input_encoded = pd.get_dummies(input_df)
  input_encoded = input_encoded.reindex(
      columns=train_columns, fill_value=0
  )

  predicted_energy = model.predict(input_encoded)[0]

  st.markdown("---")
  st.subheader("💡 실시간 시뮬레이션 결과")
  st.success(f"### ⚡ 예상 에너지 소비량: **{predicted_energy:,.2f} kWh**")


# =====================================================================
# [7단계] 다중 선형 회귀 모델 평가
# =====================================================================
elif menu == "7단계: 다중 선형 회귀 모델 평가":
  st.title("📈 7단계: 다중 선형 회귀 모델 학습 및 성능 평가")
  st.write(
      "수치형과 범주형 변수 전체를 대상으로 `pd.get_dummies` 원-핫"
      " 인코딩을 거쳐 다중 선형 회귀 모델을 학습하고 성능을 평가합니다."
  )

  df_model = st.session_state.get("df_cleaned", df_raw)

  X = df_model.drop(columns=["EnergyConsumption"])
  y = df_model["EnergyConsumption"]

  X_encoded = pd.get_dummies(X, drop_first=True)
  X_train, X_test, y_train, y_test = train_test_split(
      X_encoded, y, test_size=0.2, random_state=42
  )

  model = LinearRegression()
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))

  col1, col2 = st.columns(2)
  with col1:
    st.metric(label="결정 계수 ($R^2$ Score)", value=f"{r2:.4f}")
  with col2:
    st.metric(label="평균 제곱근 오차 (RMSE)", value=f"{rmse:.4f}")

  st.subheader("📋 학습된 모델의 회귀 계수(가중치) 확인")
  coef_df = pd.DataFrame(
      {"Feature": X_encoded.columns, "Coefficient": model.coef_}
  )
  st.dataframe(coef_df, use_container_width=True)

  st.session_state["trained_model"] = model
  st.session_state["X_encoded_columns"] = X_encoded.columns
  st.session_state["original_X"] = X
