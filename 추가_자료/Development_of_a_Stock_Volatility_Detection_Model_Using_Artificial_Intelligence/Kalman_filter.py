# 칼만 필터 적용 및 데이터 저장 함수 정의
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def apply_kalman_filter_numpy(data):
    """
    NumPy를 사용하여 1차원 칼만 필터를 적용하는 함수
    """
    n = len(data)
    x_est = np.zeros(n)  # 필터링된 상태 추정값
    P_est = np.zeros(n)  # 오차 공분산
    x_est[0] = data[0]  # 초기 상태값
    P_est[0] = 1

    # 칼만 필터 파라미터 설정
    A = 1  # 상태 전이 행렬
    H = 1  # 측정 행렬
    Q = 1e-5  # 프로세스 노이즈 공분산
    R = 0.1  # 측정 노이즈 공분산

    # 칼만 필터 적용 (예측 및 갱신 과정 반복)
    for k in range(1, n):
        x_pred = A * x_est[k - 1]
        P_pred = A * P_est[k - 1] * A + Q

        # 칼만 이득 계산
        K = P_pred * H / (H * P_pred * H + R)

        # 갱신 단계
        x_est[k] = x_pred + K * (data[k] - H * x_pred)
        P_est[k] = (1 - K * H) * P_pred

    return x_est

# 칼만 필터 적용 및 데이터 저장 함수 정의 (정규화 및 선형 보간 없이)
def preprocess_and_apply_kalman(file_path, output_path, selected_features):
    """
    주어진 데이터 파일에서 selected_features에 대해 칼만 필터를 적용하고 저장하는 함수

    Parameters:
    file_path (str): 원본 데이터 파일 경로
    output_path (str): 필터링된 데이터를 저장할 파일 경로
    selected_features (list): 칼만 필터를 적용할 열 리스트
    """
    # 데이터 로드
    df = pd.read_csv(file_path)

    # 날짜 변환 및 정렬
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    # 수치 데이터 문자열을 float로 변환 (콤마 제거 후 변환)
    for feature in selected_features:
        df[feature] = df[feature].replace(',', '', regex=True).astype(float)

    def convert_to_numeric(value):
        if isinstance(value, (float, int)):  # 이미 숫자인 경우 변환 없이 반환
            return value
        
        value = value.replace(',', '')  # 쉼표 제거
        if 'K' in value:
            return float(value.replace('K', '')) * 1_000  # K(천 단위) 변환
        elif 'M' in value:
            return float(value.replace('M', '')) * 1_000_000  # M(백만 단위) 변환
        else:
            return float(value)  # 그냥 숫자인 경우 변환


    # 거래량(Vol.) 데이터 변환 (비어있는 경우 처리)
    if "Vol." in df.columns and df["Vol."].notna().sum() > 0:
        df["Vol."] = df["Vol."].apply(convert_to_numeric)

    # 필터링된 데이터를 저장할 데이터프레임 생성
    filtered_data = df.copy()

    # 모든 선택된 변수에 칼만 필터 적용
    for feature in selected_features:
        filtered_data[feature] = apply_kalman_filter_numpy(df[feature].values)

    # 결과 시각화 (원본 데이터 vs 칼만 필터 적용 데이터)
    plt.figure(figsize=(12, 6))
    for feature in selected_features:
        plt.plot(df["Date"], df[feature], alpha=0.3, label=f"Original {feature}")
        plt.plot(df["Date"], filtered_data[feature], label=f"Filtered {feature}", linewidth=2)

    plt.legend()
    plt.title("Filtered Data with Kalman Filter (NumPy Implementation)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

    # CSV 파일로 저장 (정규화 및 선형 보간 없이)
    filtered_data.to_csv(output_path, index=False)

    return filtered_data

if __name__ == "__main__":
    # 함수 실행 예제
    file_path = "/mnt/data/Gold Futures Historical Data.csv"
    output_path = "/mnt/data/filtered_gold_futures.csv"
    selected_features = ["Price", "Open", "High", "Low"]

    # 실행
    filtered_result = preprocess_and_apply_kalman(file_path, output_path, selected_features)


