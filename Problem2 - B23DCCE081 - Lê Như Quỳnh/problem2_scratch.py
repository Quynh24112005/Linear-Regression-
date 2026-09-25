import numpy as np
import time


class BaseRegressorScratch:
    def __init__(self):
        self.w = None
        self.execution_time_ms = 0

    def _add_intercept(self, X):
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack((ones, X))

    def predict(self, X):
        X_bar = self._add_intercept(X)
        return np.dot(X_bar, self.w).flatten()

    def get_sparsity(self, tol=1e-5):
        if self.w is None:
            return 0
        w_features = self.w[1:].flatten()
        return int(np.sum(np.abs(w_features) < tol))

    def get_nonzero_indices(self, tol=1e-5):
        if self.w is None:
            return []
        w_features = self.w[1:].flatten()
        return list(np.where(np.abs(w_features) >= tol)[0])


class OLSLinearRegression(BaseRegressorScratch):
    def fit(self, X, y):
        start_time = time.time()
        X_bar = self._add_intercept(X)
        y_col = y.reshape(-1, 1)
        self.w = np.dot(np.linalg.pinv(X_bar), y_col)
        self.execution_time_ms = (time.time() - start_time) * 1000.0
        return self


class RidgeRegressionScratch(BaseRegressorScratch):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)

    def fit(self, X, y):
        start_time = time.time()
        X_bar = self._add_intercept(X)
        y_col = y.reshape(-1, 1)
        N, d = X_bar.shape

        I = np.eye(d, dtype=np.float64)
        I[0, 0] = 0.0  # Không phạt bias

        XT_X = np.dot(X_bar.T, X_bar)
        XT_y = np.dot(X_bar.T, y_col)

        # Sử dụng giải hệ phương trình tuyến tính ổn định
        self.w = np.linalg.solve(XT_X + self.alpha * N * I, XT_y)
        self.execution_time_ms = (time.time() - start_time) * 1000.0
        return self


class LassoRegressionScratch(BaseRegressorScratch):
    def __init__(self, alpha=0.01, epochs=2000, tol=1e-5):
        super().__init__()
        self.alpha = float(alpha)
        self.epochs = epochs
        self.tol = tol

    def _soft_thresholding(self, rho, lam):
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0.0

    def fit(self, X, y):
        start_time = time.time()
        X_bar = self._add_intercept(X)
        y_flat = y.flatten()
        N, d = X_bar.shape

        # Khởi tạo trọng số bằng 0
        self.w = np.zeros(d, dtype=np.float64)
        z = np.sum(X_bar**2, axis=0)  # Chuẩn bình phương của từng cột đặc trưng

        l1_penalty = self.alpha * N

        for epoch in range(self.epochs):
            w_old = self.w.copy()

            for j in range(d):
                # Tính phần dư loại trừ thành phần đóng góp của w_j
                y_pred = np.dot(X_bar, self.w)
                residual = y_flat - y_pred + self.w[j] * X_bar[:, j]
                rho_j = np.dot(X_bar[:, j], residual)

                if j == 0:
                    # w0 (bias) không bị phạt L1
                    self.w[j] = rho_j / z[j]
                else:
                    # Trọng số đặc trưng bị phạt L1
                    self.w[j] = self._soft_thresholding(rho_j, l1_penalty) / z[j]

            # Kiểm tra hội tụ qua norm thay đổi
            if np.linalg.norm(self.w - w_old, ord=np.inf) < self.tol:
                break

        self.w = self.w.reshape(-1, 1)
        self.execution_time_ms = (time.time() - start_time) * 1000.0
        return self
