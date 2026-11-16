import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
        "5단계: 다중 선형 회귀 모델 평가",
        "6단계: 실시간 예측 시뮬레이터",
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
  st.write(
      "IQR(사분위수 범위) 공식을 사용하여 각 수치형 변수별 이상치가 몇 개인지"
      " 수치와 시각화로 확인합니다."
  )

  outlier_counts = {}
  total_outliers = 0

  for col in num_cols:
    Q1 = df_raw[col].quantile(0.25)
    Q3 = df_raw[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cond = (df_raw[col] < lower_bound) | (df_raw[col] > upper_bound)
    cnt = cond.sum()
    outlier_counts[col] = cnt
    total_outliers += cnt

  outlier_df = pd.DataFrame(
      list(outlier_counts.items()), columns=["컬럼명", "이상치 개수"]
  )

  col1, col2 = st.columns([2, 1])
  with col1:
    st.subheader("📋 컬럼별 이상치 개수 현황 표")
    st.dataframe(outlier_df, use_container_width=True)
  with col2:
    st.subheader("💡 분석 의미 해석")
    st.info(
        f"**전체 데이터 행 수**: {len(df_raw)}개\n\n**총 이상치 개수**:"
        f" **{total_outliers}개**"
    )

  st.subheader("📈 주요 수치형 변수 통계 요약 표")
  st.dataframe(df_raw[num_cols].describe(), use_container_width=True)

  st.subheader("📦 주요 변수 상자 그림 (Boxplot)")
  fig, ax = plt.subplots(figsize=(10, 4))
  sns.boxplot(
      data=df_raw[["Temperature", "Humidity", "Occupancy"]],
      ax=ax,
      palette="Set2",
  )
  st.pyplot(fig)


# =====================================================================
# [3단계] 이상치 정제(클리핑)
# =====================================================================
elif menu == "3단계: 이상치 정제(클리핑)":
  st.title("✨ 건물 에너지 데이터 전처리 - 3단계: 이상치 해결")
  st.write(
      "IQR 기준을 벗어난 이상치를 정상 범위 경계값으로 다듬는 클리핑(Clipping)"
      " 기법을 적용합니다."
  )

  # 클리핑 적용
  df_processed = df_raw.copy()
  for col in num_cols:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

  st.success("🎉 이상치 정제(IQR Clipping)가 성공적으로 완료되었습니다!")

  st.subheader("💾 전처리 완료된 데이터 파일 저장 및 확인")
  # 세션에 정제된 데이터 저장 (이후 단계에서 공유 사용)
  st.session_state["df_cleaned"] = df_processed

  csv_data = df_processed.to_csv(index=False).encode("utf-8-sig")
  st.download_button(
      label="📥 정제된 데이터(Energy_consumption_cleaned.csv) 다운로드",
      data=csv_data,
      file_name="Energy_consumption_cleaned.csv",
      mime="text/csv",
  )

  st.subheader("📋 정제된 최종 데이터프레임 (상위 5개 행)")
  st.dataframe(df_processed.head(), use_container_width=True)


# =====================================================================
# [4단계] EDA 및 독립변수 선정 근거
# =====================================================================
elif menu == "4단계: EDA 및 독립변수 선정 근거":
  st.title("📊 탐색적 데이터 분석(EDA) 및 독립변수 선정 근거 시각화")
  st.write(
      "시각화를 통해 수치형 변수와 범주형 변수 총 9개가 왜 에너지 소비량 예측의"
      " 독립변수로 적합한지 검증합니다."
  )

  # 정제된 데이터가 있으면 사용, 없으면 원본 사용
  df_eda = st.session_state.get("df_cleaned", df_raw)

  st.header("1. 📈 수치형 변수와 에너지 소비량 관계 (산점도 & 추세선)")
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("건물 면적 vs 에너지 소비량")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df_eda,
        x="SquareFootage",
        y="EnergyConsumption",
        hue="Occupancy",
        palette="viridis",
        ax=ax,
        alpha=0.7,
    )
    sns.regplot(
        data=df_eda,
        x="SquareFootage",
        y="EnergyConsumption",
        scatter=False,
        ax=ax,
        color="red",
    )
    st.pyplot(fig)
    st.info(
        "💡 면적이 넓을수록 에너지가 비례해서 증가하는 뚜렷한 양의 상관관계를"
        " 보임."
    )

  with col2:
    st.subheader("온도 vs 에너지 소비량")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df_eda,
        x="Temperature",
        y="EnergyConsumption",
        hue="Humidity",
        palette="coolwarm",
        ax=ax,
        alpha=0.7,
    )
    sns.regplot(
        data=df_eda,
        x="Temperature",
        y="EnergyConsumption",
        scatter=False,
        ax=ax,
        color="blue",
    )
    st.pyplot(fig)
    st.info("💡 기온 변화에 따른 냉난방 부하 변동성을 확인할 수 있음.")

  st.header("2. 📊 범주형 조건별 평균 에너지 소비량 (막대그래프)")
  col3, col4 = st.columns(2)
  with col3:
    st.subheader("냉난방기 가동 여부별 평균 에너지")
    fig, ax = plt.subplots(figsize=(6, 4))
    hvac_mean = (
        df_eda.groupby("HVACUsage")["EnergyConsumption"].mean().reset_index()
    )
    sns.barplot(
        data=hvac_mean, x="HVACUsage", y="EnergyConsumption", palette="Set2", ax=ax
    )
    st.pyplot(fig)
    st.info("💡 냉난방기 가동 유무에 따라 에너지 소비량에 확연한 차이가 발생함.")

  with col4:
    st.subheader("휴일 여부별 평균 에너지")
    fig, ax = plt.subplots(figsize=(6, 4))
    holiday_mean = (
        df_eda.groupby("Holiday")["EnergyConsumption"].mean().reset_index()
    )
    sns.barplot(
        data=holiday_mean,
        x="Holiday",
        y="EnergyConsumption",
        palette="Pastel1",
        ax=ax,
    )
    st.pyplot(fig)
    st.info("💡 휴일 상주 인원 변화 등에 따른 패턴 차이가 존재함.")

  st.header("3. 🔥 수치형 변수 종합 상관관계 히트맵")
  fig, ax = plt.subplots(figsize=(8, 4))
  sns.heatmap(
      df_eda[num_cols].corr(),
      annot=True,
      cmap="coolwarm",
      fmt=".2f",
      ax=ax,
      linewidths=0.5,
  )
  st.pyplot(fig)


# =====================================================================
# [5단계] 다중 선형 회귀 모델 학습 및 평가 (MSE 적용)
# =====================================================================
elif menu == "5단계: 다중 선형 회귀 모델 평가":
  st.title("📈 다중 선형 회귀 모델 학습 및 성능 평가")
  st.write(
      "수치형 5개와 범주형 4개 변수 전체를 대상으로 `pd.get_dummies` 원-핫"
      " 인코딩을 거쳐 다중 선형 회귀 모델을 학습하고 평가합니다."
  )

  df_model = st.session_state.get("df_cleaned", df_raw)

  # 독립변수(X)와 종속변수(y) 분리
  X = df_model.drop(columns=["EnergyConsumption"])
  y = df_model["EnergyConsumption"]

  # 범주형 변수 원-핫 인코딩 적용
  X_encoded = pd.get_dummies(X, drop_first=True)

  # 학습/테스트 데이터 분할
  X_train, X_test, y_train, y_test = train_test_split(
      X_encoded, y, test_size=0.2, random_state=42
  )

  # 모델 학습
  model = LinearRegression()
  model.fit(X_train, y_train)

  # 평가 예측
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)  # RMSE에서 MSE로 변경

  col1, col2 = st.columns(2)
  with col1:
    st.metric(label="결정 계수 ($R^2$ Score)", value=f"{r2:.4f}")
  with col2:
    st.metric(label="평균 제곱 오차 (MSE)", value=f"{mse:.4f}")  # 지표 수정

  st.subheader("📋 학습된 모델의 회귀 계수(가중치) 확인")
  coef_df = pd.DataFrame(
      {"Feature": X_encoded.columns, "Coefficient": model.coef_}
  )
  st.dataframe(coef_df, use_container_width=True)

  # 세션에 학습된 모델과 인코딩 구조 저장 (6단계 시뮬레이터 연동용)
  st.session_state["trained_model"] = model
  st.session_state["X_encoded_columns"] = X_encoded.columns
  st.session_state["original_X"] = X


# =====================================================================
# [6단계] 실시간 예측 시뮬레이터
# =====================================================================
elif menu == "6단계: 실시간 예측 시뮬레이터":
  st.title("🎛️ 실시간 건물 에너지 소비량 예측 시뮬레이터")
  st.write(
      "환경 조건(슬라이더)과 설비/일정 조건(셀렉트박스)을 직접 조작하여 예상"
      " 에너지 소비량(kWh)을 실시간으로 확인합니다."
  )

  if "trained_model" not in st.session_state:
    st.warning(
        "⚠️ 먼저 **5단계: 다중 선형 회귀 모델 평가** 메뉴를 방문하여 모델을"
        " 학습시켜 주세요!"
    )
  else:
    model = st.session_state["trained_model"]
    train_columns = st.session_state["X_encoded_columns"]
    orig_X = st.session_state["original_X"]
    df_sim = st.session_state.get("df_cleaned", df_raw)

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

    # 입력값 DataFrame 변환 및 원-핫 인코딩 정렬
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(
        columns=train_columns, fill_value=0
    )

    # 예측 수행
    predicted_energy = model.predict(input_encoded)[0]

    st.markdown("---")
    st.subheader("💡 실시간 시뮬레이션 결과")
    st.success(f"### ⚡ 예상 에너지 소비량: **{predicted_energy:,.2f} kWh**")
