import numpy as np
import time


class BaseLinearRegressionScratch:
    def __init__(self):
        self.w = None  # Vector trọng số [w0, w1, ..., wd]^T bao gồm cả bias w0
        self.loss_history = []  # Lịch sử MSE loss qua các epochs
        self.execution_time_ms = 0  # Thời gian huấn luyện (ms)

    def _add_intercept(self, X):
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack((ones, X))

    def predict(self, X):
        X_bar = self._add_intercept(X)
        return np.dot(X_bar, self.w).flatten()

    def compute_residuals(self, X_bar, y):
        y_pred = np.dot(X_bar, self.w)
        return y_pred - y.reshape(-1, 1)

    def compute_mse_loss(self, X_bar, y):
        N = len(y)
        residuals = self.compute_residuals(X_bar, y)
        loss = (1.0 / (2.0 * N)) * np.sum(residuals**2)
        return loss

    def compute_gradient(self, X_bar, residuals):
        N = len(residuals)
        return (1.0 / N) * np.dot(X_bar.T, residuals)


class LinearRegressionNormalEq(BaseLinearRegressionScratch):
    def fit(self, X, y):
        start_time = time.time()
        X_bar = self._add_intercept(X)
        y_col = y.reshape(-1, 1)

        # Cài đặt từ nguyên lý cơ bản qua Giả nghịch đảo (Pseudoinverse)
        self.w = np.dot(np.linalg.pinv(X_bar), y_col)

        self.loss_history.append(self.compute_mse_loss(X_bar, y_col))
        self.execution_time_ms = (time.time() - start_time) * 1000.0
        return self


class LinearRegressionBGD(BaseLinearRegressionScratch):

    def __init__(self, learning_rate=0.01, epochs=1000, tol=1e-7):
        super().__init__()
        self.lr = learning_rate
        self.epochs = epochs
        self.tol = tol

    def fit(self, X, y):
        start_time = time.time()
        X_bar = self._add_intercept(X)
        y_col = y.reshape(-1, 1)
        N, d = X_bar.shape

        # Khởi tạo trọng số bằng 0
        self.w = np.zeros((d, 1), dtype=np.float64)

        for epoch in range(self.epochs):
            residuals = self.compute_residuals(X_bar, y_col)
            grad = self.compute_gradient(X_bar, residuals)
            # Gradient clipping để chống bùng nổ gradient
            grad = np.clip(grad, -1e3, 1e3)

            w_old = self.w.copy()
            self.w -= self.lr * grad

            loss = self.compute_mse_loss(X_bar, y_col)
            self.loss_history.append(loss)

            # Kiểm tra hội tụ
            if np.linalg.norm(self.w - w_old) < self.tol:
                break

        self.execution_time_ms = (time.time() - start_time) * 1000.0
        return self


class LinearRegressionMBGD(BaseLinearRegressionScratch):

    def __init__(self, learning_rate=0.01, epochs=200, batch_size=64, random_state=42):
        super().__init__()
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        start_time = time.time()
        np.random.seed(self.random_state)
        X_bar = self._add_intercept(X)
        y_col = y.reshape(-1, 1)
        N, d = X_bar.shape

        self.w = np.zeros((d, 1), dtype=np.float64)

        for epoch in range(self.epochs):
            indices = np.random.permutation(N)
            X_shuffled = X_bar[indices]
            y_shuffled = y_col[indices]

            for i in range(0, N, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]

                residuals_batch = np.dot(X_batch, self.w) - y_batch
                grad_batch = (1.0 / len(y_batch)) * np.dot(X_batch.T, residuals_batch)
                grad_batch = np.clip(grad_batch, -1e3, 1e3)

                self.w -= self.lr * grad_batch

            loss = self.compute_mse_loss(X_bar, y_col)
            self.loss_history.append(loss)

        self.execution_time_ms = (time.time() - start_time) * 1000.0
        return self


class LinearRegressionSGD(BaseLinearRegressionScratch):

    def __init__(self, learning_rate=0.005, epochs=30, decay=0.0001, random_state=42):
        super().__init__()
        self.lr = learning_rate
        self.epochs = epochs
        self.decay = decay
        self.random_state = random_state

    def fit(self, X, y):
        start_time = time.time()
        np.random.seed(self.random_state)
        X_bar = self._add_intercept(X)
        y_col = y.reshape(-1, 1)
        N, d = X_bar.shape

        self.w = np.zeros((d, 1), dtype=np.float64)
        t = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(N)
            X_shuffled = X_bar[indices]
            y_shuffled = y_col[indices]

            for i in range(N):
                xi = X_shuffled[i : i + 1]
                yi = y_shuffled[i : i + 1]

                pred_i = np.dot(xi, self.w)
                grad_i = np.dot(xi.T, (pred_i - yi))
                grad_i = np.clip(grad_i, -1e3, 1e3)

                current_lr = self.lr / (1.0 + self.decay * t)
                self.w -= current_lr * grad_i
                t += 1

            loss = self.compute_mse_loss(X_bar, y_col)
            self.loss_history.append(loss)

        self.execution_time_ms = (time.time() - start_time) * 1000.0
        return self
