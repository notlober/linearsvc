import random

class LinearSVM:
    def __init__(self, C=1.0, eps=0.001, max_passes=10):
        self.C = C
        self.eps = eps
        self.max_passes = max_passes
        self.w = None
        self.b = None

    def fit(self, y, x):
        prob = self._create_svm_problem(y, x)
        Q = self._compute_Q_matrix(prob)
        alpha = self._solve_dual_problem(Q, prob)
        self.w = self._compute_weight_vector(alpha, prob)
        self.b = self._compute_bias_term(alpha, prob)

    def predict(self, x):
        predictions = []
        for x_i in x:
            decision_value = sum([self.w[j] * x_i[j] for j in range(len(self.w))]) + self.b
            predictions.append(1 if decision_value > 0 else -1)
        return predictions

    def _create_svm_problem(self, y, x):
        return [(y_i, x_i) for y_i, x_i in zip(y, x)]

    def _compute_Q_matrix(self, prob):
        Q = [[0.0 for _ in range(len(prob))] for _ in range(len(prob))]
        for i in range(len(prob)):
            for j in range(len(prob)):
                Q[i][j] = prob[i][0] * prob[j][0] * self._dot_product(prob[i][1], prob[j][1])
        return Q

    def _dot_product(self, x_i, x_j):
        return sum([x_i[k] * x_j[k] for k in range(len(x_i))])

    def _solve_dual_problem(self, Q, prob):
        l = len(prob)
        alpha = [0.0 for _ in range(l)]
        b = 0
        passes = 0

        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(l):
                E_i = self._predict_with_alpha(alpha, prob, b, i) - prob[i][0]
                if (E_i * prob[i][0] < -self.eps and alpha[i] < self.C) or (E_i * prob[i][0] > self.eps and alpha[i] > 0):
                    j = random.randint(0, l - 1)
                    while j == i:
                        j = random.randint(0, l - 1)

                    E_j = self._predict_with_alpha(alpha, prob, b, j) - prob[j][0]
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    if prob[i][0] != prob[j][0]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    eta = 2 * self._dot_product(prob[i][1], prob[j][1]) - self._dot_product(prob[i][1], prob[i][1]) - self._dot_product(prob[j][1], prob[j][1])
                    if eta >= 0:
                        continue

                    alpha[j] = alpha_j_old - prob[j][0] * (E_i - E_j) / eta

                    alpha[j] = max(alpha[j], L)
                    alpha[j] = min(alpha[j], H)

                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    alpha[i] = alpha_i_old + prob[i][0] * prob[j][0] * (alpha_j_old - alpha[j])

                    b1 = b - E_i - prob[i][0] * (alpha[i] - alpha_i_old) * self._dot_product(prob[i][1], prob[i][1]) - prob[j][0] * (alpha[j] - alpha_j_old) * self._dot_product(prob[i][1], prob[j][1])
                    b2 = b - E_j - prob[i][0] * (alpha[i] - alpha_i_old) * self._dot_product(prob[i][1], prob[j][1]) - prob[j][0] * (alpha[j] - alpha_j_old) * self._dot_product(prob[j][1], prob[j][1])
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        return alpha

    def _predict_with_alpha(self, alpha, prob, b, i):
        return sum([alpha[j] * prob[j][0] * self._dot_product(prob[j][1], prob[i][1]) for j in range(len(prob))]) + b

    def _compute_weight_vector(self, alpha, prob):
        n = len(prob[0][1])
        w = [0.0 for _ in range(n)]
        for i in range(len(prob)):
            for j in range(len(prob[i][1])):
                w[j] += alpha[i] * prob[i][0] * prob[i][1][j]
        return w

    def _compute_bias_term(self, alpha, prob):
        b_values = []
        for i in range(len(prob)):
            if alpha[i] > 0:
                b_values.append(prob[i][0] - sum([self.w[j] * prob[i][1][j] for j in range(len(self.w))]))
        if b_values:
            return sum(b_values) / len(b_values)
        else:
            return 0.0