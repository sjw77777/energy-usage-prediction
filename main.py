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
    page_title="🏢 건물 에너지 데이터 분석 및 특징 선택 검증 시스템",
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
        "3단계: 이상치 정제(클리핑) 및 파일 저장",
        "4단계: 9개 변수 EDA 및 독립변수 선정 근거",
        "5단계: 독립 변수 제거 타당성 검증 (특징 선택)",
        "6단계: 다중 선형 회귀 모델 평가",
        "7단계: 실시간 예측 시뮬레이터",
    ],
)

num_cols = [
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "EnergyConsumption",
]
cat_cols = ["HVACUsage", "LightingUsage", "DayOfWeek", "Holiday"]


# =====================================================================
# [1단계] 결측치 확인
# =====================================================================
if menu == "1단계: 결측치 확인":
  st.title("⚡ 건물 에너지 데이터 전처리 - 1단계: 결측치 확인")
  st.write(
      "업로드된 원본 'Energy_consumption.csv' 파일에서 결측치(빈칸)를"
      " 확인합니다."
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
  st.write(
      "IQR(사분위수 범위) 공식을 사용하여 원본 데이터의 수치형 변수별 이상치"
      " 개수를 파악합니다."
  )

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

  st.subheader("📦 수치형 변수 상자 그림 (Boxplot)")
  fig, ax = plt.subplots(figsize=(10, 4))
  sns.boxplot(
      data=df_raw[["Temperature", "Humidity", "Occupancy", "SquareFootage"]],
      ax=ax,
      palette="Set2",
  )
  st.pyplot(fig)


# =====================================================================
# [3단계] 이상치 정제(클리핑) 및 파일 저장
# =====================================================================
elif menu == "3단계: 이상치 정제(클리핑) 및 파일 저장":
  st.title("✨ 건물 에너지 데이터 전처리 - 3단계: 이상치 정제 및 세션 공유")
  st.write(
      "IQR 클리핑을 통해 이상치를 정상 범위로 조정하고, 이후 모델링 및 평가에"
      " 공통으로 사용될 정제 데이터(`df_cleaned`)를 생성합니다."
  )

  df_processed = df_raw.copy()
  for col in num_cols:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

  # 🌟 세션 상태에 정제된 데이터 강제 저장 (이후 모든 단계에서 이 데이터를 공유함)
  st.session_state["df_cleaned"] = df_processed

  st.success(
      "🎉 이상치 정제(클리핑) 완료 및 세션 반영 완료! 이제 아래 버튼으로 정제된"
      " 파일을 다운로드할 수 있습니다."
  )

  csv_data = df_processed.to_csv(index=False).encode("utf-8-sig")
  st.download_button(
      label="📥 정제된 데이터 파일 다운로드 (Energy_consumption_cleaned.csv)",
      data=csv_data,
      file_name="Energy_consumption_cleaned.csv",
      mime="text/csv",
  )

  st.subheader("📋 정제된 데이터 미리보기")
  st.dataframe(df_processed.head(), use_container_width=True)


# =====================================================================
# [4단계] EDA 및 독립변수 선정 근거
# =====================================================================
elif menu == "4단계: 9개 변수 EDA 및 독립변수 선정 근거":
  st.title("📊 4단계: 탐색적 데이터 분석(EDA) 및 독립변수 시각화")
  st.write(
      "앞서 3단계에서 정제된 데이터(`df_cleaned`)를 기반으로 수치형 및 범주형"
      " 변수들의 관계를 시각화합니다."
  )

  # 세션에 정제된 데이터가 없으면 원본으로 대체 방어
  df_eda = st.session_state.get("df_cleaned", df_raw)

  st.subheader("📈 수치형 주요 변수 vs 에너지 소비량")
  col1, col2 = st.columns(2)
  with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df_eda,
        x="SquareFootage",
        y="EnergyConsumption",
        ax=ax,
        alpha=0.6,
        color="dodgerblue",
    )
    sns.regplot(
        data=df_eda,
        x="SquareFootage",
        y="EnergyConsumption",
        scatter=False,
        ax=ax,
        color="crimson",
    )
    st.pyplot(fig)
    st.info("💡 건물 면적에 따른 에너지 소비량 추세 확인")

  with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df_eda,
        x="Temperature",
        y="EnergyConsumption",
        ax=ax,
        alpha=0.6,
        color="seagreen",
    )
    sns.regplot(
        data=df_eda,
        x="Temperature",
        y="EnergyConsumption",
        scatter=False,
        ax=ax,
        color="crimson",
    )
    st.pyplot(fig)
    st.info("💡 온도에 따른 에너지 소비량 추세 확인")

  st.subheader("📊 범주형 변수별 평균 에너지 소비량")
  fig, ax = plt.subplots(figsize=(8, 4))
  hvac_mean = (
      df_eda.groupby("HVACUsage")["EnergyConsumption"].mean().reset_index()
  )
  sns.barplot(
      data=hvac_mean, x="HVACUsage", y="EnergyConsumption", palette="Set2", ax=ax
  )
  st.pyplot(fig)


# =====================================================================
# [5단계] 독립 변수 제거 타당성 검증 (특징 선택)
# =====================================================================
elif menu == "5단계: 독립 변수 제거 타당성 검증 (특징 선택)":
  st.title("🔍 5단계: 상관계수가 낮은 독립 변수 제거 타당성 검증")
  st.write(
      "정제된 데이터(`df_cleaned`)를 활용하여, 전체 변수를 사용한 **기준"
      " 모델**과 의심되는 변수들을 하나씩 제외한 모델들의 성능(MAE, RMSE,"
      " $R^2$)을 비교합니다."
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

    # 데이터 누출 방지: 훈련 데이터 기준 원-핫 인코딩 후 테스트 데이터에 동일 적용
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
  st.subheader("📊 모델별 예측 성능 비교 표")
  st.dataframe(result_df, use_container_width=True)

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
# [6단계] 다중 선형 회귀 모델 평가
# =====================================================================
elif menu == "6단계: 다중 선형 회귀 모델 평가":
  st.title("📈 6단계: 다중 선형 회귀 모델 학습 및 최종 성능 평가")
  st.write("정제된 데이터를 대상으로 최종 회귀 모델을 학습하고 평가합니다.")

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

  st.session_state["trained_model"] = model
  st.session_state["X_encoded_columns"] = X_encoded.columns
  st.session_state["original_X"] = X


# =====================================================================
# [7단계] 실시간 예측 시뮬레이터
# =====================================================================
elif menu == "7단계: 실시간 예측 시뮬레이터":
  st.title("🎛️ 7단계: 실시간 에너지 소비량 예측 시뮬레이터")
  st.write("정제된 데이터의 범위 내에서 변수 조건을 조작하여 예측합니다.")

  if "trained_model" not in st.session_state:
    st.warning("⚠️ 먼저 6단계(모델 평가)를 방문하여 모델을 학습시켜 주세요!")
  else:
    model = st.session_state["trained_model"]
    train_columns = st.session_state["X_encoded_columns"]
    orig_X = st.session_state["original_X"]
    df_sim = st.session_state.get("df_cleaned", df_raw)

    user_input = {}
    st.subheader("🔧 변수 조건 설정")
    for col in orig_X.columns:
      if pd.api.types.is_numeric_dtype(orig_X[col]):
        user_input[col] = st.slider(
            f"{col}",
            min_value=float(df_sim[col].min()),
            max_value=float(df_sim[col].max()),
            value=float(df_sim[col].mean()),
        )
      else:
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
