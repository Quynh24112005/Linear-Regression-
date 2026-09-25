import numpy as np
import pandas as pd
from dataset_loader_p2 import StandardScalerScratch


class BikeBasisExpansion:

    def __init__(self, level=2):
        self.level = level
        self.scaler = StandardScalerScratch()
        self.feature_names = []

    def _expand(self, df):
        features = {}

        # 1. Các đặc trưng số cơ bản
        base_cols = [
            "season",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed",
        ]
        for col in base_cols:
            if col in df.columns:
                features[col] = df[col].values.astype(np.float64)

        if self.level >= 1:
            # 2. Level 1: HÀM CƠ SỞ TUẦN HOÀN (CYCLIC / PERIODIC BASIS)
            # Biến đổi giờ trong ngày (Chu kỳ 24h & Harmonic 12h cho 2 đỉnh cao điểm đi làm 8h & 17h)
            hr = df["hr"].values.astype(np.float64)
            features["hr_sin_24"] = np.sin(2.0 * np.pi * hr / 24.0)
            features["hr_cos_24"] = np.cos(2.0 * np.pi * hr / 24.0)
            features["hr_sin_12"] = np.sin(4.0 * np.pi * hr / 24.0)
            features["hr_cos_12"] = np.cos(4.0 * np.pi * hr / 24.0)

            # Biến đổi tháng trong năm (Chu kỳ 12 tháng)
            mnth = df["mnth"].values.astype(np.float64)
            features["mnth_sin"] = np.sin(2.0 * np.pi * mnth / 12.0)
            features["mnth_cos"] = np.cos(2.0 * np.pi * mnth / 12.0)

            # Biến đổi thứ trong tuần (Chu kỳ 7 ngày)
            wday = df["weekday"].values.astype(np.float64)
            features["wday_sin"] = np.sin(2.0 * np.pi * wday / 7.0)
            features["wday_cos"] = np.cos(2.0 * np.pi * wday / 7.0)

        if self.level >= 2:
            # 3. Level 2: ĐA THỨC VÀ TƯƠNG TÁC ĐẶC TRƯNG (POLYNOMIAL & INTERACTIONS)
            temp = df["temp"].values.astype(np.float64)
            hum = df["hum"].values.astype(np.float64)
            wind = df["windspeed"].values.astype(np.float64)
            work = df["workingday"].values.astype(np.float64)

            # Bậc 2 & Bậc 3 của các biến thời tiết liên tục
            features["temp_sq"] = temp**2
            features["hum_sq"] = hum**2
            features["wind_sq"] = wind**2
            features["temp_cube"] = temp**3

            # Tương tác thời tiết (Weather Interactions)
            features["temp_x_hum"] = temp * hum  # Chỉ số oi bức nhiệt độ - độ ẩm
            features["temp_x_wind"] = temp * wind  # Chỉ số gió lạnh
            features["hum_x_wind"] = hum * wind

            # Tương tác chu kỳ giờ x ngày làm việc (Commute pattern differences)
            features["hr_sin24_x_work"] = features["hr_sin_24"] * work
            features["hr_cos24_x_work"] = features["hr_cos_24"] * work
            features["hr_sin12_x_work"] = features["hr_sin_12"] * work
            features["hr_cos12_x_work"] = features["hr_cos_12"] * work

            # Tương tác nhiệt độ x mùa
            season = df["season"].values.astype(np.float64)
            features["temp_x_season"] = temp * season
            features["temp_x_work"] = temp * work

        expanded_df = pd.DataFrame(features)
        self.feature_names = list(expanded_df.columns)
        return expanded_df.values

    def fit_transform(self, X_df):
        X_expanded = self._expand(X_df)
        return self.scaler.fit_transform(X_expanded)

    def transform(self, X_df):
        X_expanded = self._expand(X_df)
        return self.scaler.transform(X_expanded)


if __name__ == "__main__":
    from dataset_loader_p2 import load_and_preprocess_bike_data

    data = load_and_preprocess_bike_data()
    for lvl in [0, 1, 2]:
        be = BikeBasisExpansion(level=lvl)
        X_tr = be.fit_transform(data["X_train_df"])
        print(
            f"Level {lvl} expanded shape: {X_tr.shape} ({len(be.feature_names)} features)"
        )
