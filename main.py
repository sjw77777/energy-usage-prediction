import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# --- 1. 페이지 설정 및 디자인 ---
st.set_page_config(
    page_title="스마트 빌딩 에너지 소비량 분석 및 예측 시스템",
    page_icon="⚡",
    layout="wide",
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

st.title("🏢 스마트 빌딩 에너지 소비량 전처리, EDA 및 예측 시스템")
st.markdown("""
결측치 확인, 이상치 정제(클리핑), 독립변수 전수 EDA를 거쳐, **`SquareFootage`를 제외한 변수들**로 다중 선형 회귀 모델을 구축하고 실시간 시뮬레이션 및 최종 모델 평가를 수행합니다.
""")
st.markdown("---")


# --- 2. 데이터 로딩 ---
@st.cache_data
def load_data():
  return pd.read_csv("Energy_consumption.csv")


try:
  df_raw = load_data()

  num_cols = [
      "Temperature",
      "Humidity",
      "SquareFootage",
      "Occupancy",
      "RenewableEnergy",
      "EnergyConsumption",
  ]
  cat_cols = ["HVACUsage", "LightingUsage", "DayOfWeek", "Holiday"]

  # --- 사이드바 메뉴 ---
  st.sidebar.title("🧭 분석 및 시뮬레이션 단계")
  menu = st.sidebar.radio(
      "이동할 단계를 선택하세요:",
      [
          "1단계: 결측치 확인",
          "2단계: 이상치 분석 및 Boxplot",
          "3단계: 이상치 정제(클리핑) 및 다운로드",
          "4단계: 독립변수 전수 EDA 시각화",
          "5단계: 실시간 예측 시뮬레이터",
          "6단계: 다중 선형 회귀 모델 최종 평가",
      ],
  )

  # =====================================================================
  # [1단계] 결측치 확인
  # =====================================================================
  if menu == "1단계: 결측치 확인":
    st.header("📌 1단계: 결측치(Missing Value) 확인")
    null_counts = df_raw.isnull().sum().reset_index()
    null_counts.columns = ["컬럼명", "결측치 개수"]
    total_nulls = df_raw.isnull().sum().sum()

    st.dataframe(null_counts, use_container_width=True)
    st.info(f"💡 데이터셋 내 총 결측치 개수는 **{total_nulls}개**입니다.")

  # =====================================================================
  # [2단계] 이상치 분석 및 Boxplot 시각화
  # =====================================================================
  elif menu == "2단계: 이상치 분석 및 Boxplot":
    st.header("📌 2단계: 이상치(Outlier) 분석 및 상자 그림(Boxplot)")
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

    st.subheader("📦 수치형 변수 개별 Boxplot 시각화")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
      sns.boxplot(y=df_raw[col], ax=axes[idx], color="skyblue")
      axes[idx].set_title(f"Boxplot of {col}")
    plt.tight_layout()
    st.pyplot(fig)

  # =====================================================================
  # [3단계] 이상치 정제(클리핑) 및 파일 다운로드
  # =====================================================================
  elif menu == "3단계: 이상치 정제(클리핑) 및 다운로드":
    st.header("📌 3단계: 이상치 정제(클리핑) 및 정제 파일 생성")
    df_processed = df_raw.copy()
    for col in num_cols:
      Q1 = df_processed[col].quantile(0.25)
      Q3 = df_processed[col].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

    st.session_state["df_cleaned"] = df_processed
    st.success("🎉 IQR 클리핑을 통한 이상치 정제가 완료되었습니다!")

    csv_data = df_processed.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 정제된 데이터(Energy_consumption_cleaned.csv) 다운로드",
        data=csv_data,
        file_name="Energy_consumption_cleaned.csv",
        mime="text/csv",
    )
    st.dataframe(df_processed.head(), use_container_width=True)

  # =====================================================================
  # [4단계] 독립변수 전수 EDA 시각화
  # =====================================================================
  elif menu == "4단계: 독립변수 전수 EDA 시각화":
    st.header("📌 4단계: 독립변수 전체에 대한 탐색적 분석(EDA) 및 시각화")
    df_eda = st.session_state.get("df_cleaned", df_raw)

    st.subheader("📈 [수치형 독립변수] vs 에너지 소비량 산점도")
    numeric_features = [
        "Temperature",
        "Humidity",
        "SquareFootage",
        "Occupancy",
        "RenewableEnergy",
    ]

    for i in range(0, len(numeric_features), 2):
      c1, c2 = st.columns(2)
      with c1:
        feat1 = numeric_features[i]
        st.write(f"**{feat1} vs EnergyConsumption**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df_eda,
            x=feat1,
            y="EnergyConsumption",
            ax=ax,
            alpha=0.6,
            color="royalblue",
        )
        sns.regplot(
            data=df_eda,
            x=feat1,
            y="EnergyConsumption",
            scatter=False,
            ax=ax,
            color="red",
        )
        st.pyplot(fig)
      with c2:
        if i + 1 < len(numeric_features):
          feat2 = numeric_features[i + 1]
          st.write(f"**{feat2} vs EnergyConsumption**")
          fig, ax = plt.subplots(figsize=(6, 4))
          sns.scatterplot(
              data=df_eda,
              x=feat2,
              y="EnergyConsumption",
              ax=ax,
              alpha=0.6,
              color="seagreen",
          )
          sns.regplot(
              data=df_eda,
              x=feat2,
              y="EnergyConsumption",
              scatter=False,
              ax=ax,
              color="red",
          )
          st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 [범주형 독립변수]별 평균 에너지 소비량 (막대그래프)")
    for i in range(0, len(cat_cols), 2):
      c3, c4 = st.columns(2)
      with c3:
        cat1 = cat_cols[i]
        st.write(f"**{cat1}에 따른 평균 에너지 소비량**")
        fig, ax = plt.subplots(figsize=(6, 4))
        cat_mean = (
            df_eda.groupby(cat1)["EnergyConsumption"].mean().reset_index()
        )
        sns.barplot(
            data=cat_mean,
            x=cat1,
            y="EnergyConsumption",
            palette="Set2",
            ax=ax,
        )
        st.pyplot(fig)
      with c4:
        if i + 1 < len(cat_cols):
          cat2 = cat_cols[i + 1]
          st.write(f"**{cat2}에 따른 평균 에너지 소비량**")
          fig, ax = plt.subplots(figsize=(6, 4))
          cat_mean2 = (
              df_eda.groupby(cat2)["EnergyConsumption"].mean().reset_index()
          )
          sns.barplot(
              data=cat_mean2,
              x=cat2,
              y="EnergyConsumption",
              palette="Pastel1",
              ax=ax,
          )
          st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔥 수치형 변수 종합 상관관계 히트맵")
    fig, ax = plt.subplots(figsize=(8, 5))
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
  # [5단계] 실시간 예측 시뮬레이터 (SquareFootage 제외 모델 학습 적용)
  # =====================================================================
  elif menu == "5단계: 실시간 예측 시뮬레이터":
    st.header("📌 5단계: 실시간 건물 에너지 소비량 예측 시뮬레이션")
    st.write(
        "💡 안내: 모델 학습 시 요구사항에 따라 **`SquareFootage` 변수는"
        " 제외**되었습니다."
    )

    df_sim = st.session_state.get("df_cleaned", df_raw)

    base_df = df_sim.drop(columns=["SquareFootage"], errors="ignore")
    if "Timestamp" in base_df.columns:
      base_df = base_df.drop(columns=["Timestamp"])

    df_encoded = pd.get_dummies(base_df, drop_first=True)
    target_col = "EnergyConsumption"
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.session_state["trained_model"] = model
    st.session_state["X_columns"] = X.columns
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
      st.subheader("🌡️ 수치형 환경 변수 조절")
      in_temp = st.slider(
          "Temperature (온도)",
          float(df_raw["Temperature"].min()),
          float(df_raw["Temperature"].max()),
          float(df_raw["Temperature"].mean()),
      )
      in_hum = st.slider(
          "Humidity (습도)",
          float(df_raw["Humidity"].min()),
          float(df_raw["Humidity"].max()),
          float(df_raw["Humidity"].mean()),
      )
      in_occ = st.slider(
          "Occupancy (재실 인원)",
          int(df_raw["Occupancy"].min()),
          int(df_raw["Occupancy"].max()),
          int(df_raw["Occupancy"].mean()),
      )
      in_reno = st.slider(
          "RenewableEnergy (신재생에너지)",
          float(df_raw["RenewableEnergy"].min()),
          float(df_raw["RenewableEnergy"].max()),
          float(df_raw["RenewableEnergy"].mean()),
      )

    with sim_col2:
      st.subheader("⚙️ 범주형 설비 및 일정 상태 조절")
      in_hvac = st.selectbox(
          "HVACUsage (냉난방기 사용 여부)", df_raw["HVACUsage"].unique()
      )
      in_light = st.selectbox(
          "LightingUsage (조명 사용 여부)", df_raw["LightingUsage"].unique()
      )
      in_day = st.selectbox("DayOfWeek (요일)", df_raw["DayOfWeek"].unique())
      in_holiday = st.selectbox(
          "Holiday (휴일 여부)", df_raw["Holiday"].unique()
      )

    user_input_raw = pd.DataFrame({
        "Temperature": [in_temp],
        "Humidity": [in_hum],
        "Occupancy": [in_occ],
        "RenewableEnergy": [in_reno],
        "HVACUsage": [in_hvac],
        "LightingUsage": [in_light],
        "DayOfWeek": [in_day],
        "Holiday": [in_holiday],
    })

    user_input_encoded = pd.get_dummies(user_input_raw)
    user_input_encoded = user_input_encoded.reindex(
        columns=X.columns,
