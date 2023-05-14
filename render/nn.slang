
struct LargeMatrix<let M: int, let N: int> {
    float value[M][N];

    __subscript(int x, int y)->float
    {
        get { return value[x][y]; }
        set { value[x][y] = newValue; }
    }

    matrix<float, M, N> to_matrix() {
        matrix<float, M, N> output;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                output[i][j] = value[i][j];
            }
        }
        return output;
    }

    static LargeMatrix<M, N> from_matrix(matrix<float, M, N> input) {
        LargeMatrix<M, N> output;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                output.value[i][j] = input[i][j];
            }
        }
        return output;
    }
};

struct LargeVector<let N: int> {
    float value[N];

    __subscript(int x)->float
    {
        get { return value[x]; }
        set { value[x] = newValue; }
    }

    vector<float, N> to_vector() {
        vector<float, N> output;
        for (int i = 0; i < N; i++) {
            output[i] = value[i];
        }
        return output;
    }

    static LargeVector<N> from_vector(vector<float, N> input) {
        LargeVector<N> output;
        for (int i = 0; i < N; i++) {
            output[i] = input[i];
        }
        return output;
    }
};

LargeVector<M> mul<let M: int, let N: int>(LargeMatrix<M, N> lhs, LargeVector<N> rhs) {
    LargeVector<M> output;
    for (int i = 0; i < M; i++) {
        output[i] = 0;
        for (int j = 0; j < N; j++) {
            output[i] += lhs.value[i][j] * rhs[j];
        }
    }
    return output;
}

LargeMatrix<M, C> mul<let M: int, let N: int, let C: int>(LargeMatrix<M, N> lhs, LargeMatrix<N, C> rhs) {
    LargeMatrix<M, C> output;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < C; j++) {
            output.value[i][j] = 0;
            for (int k = 0; k < N; k++) {
                output.value[i][j] += lhs.value[i][k] * rhs.value[k][j];
            }
        }
    }
    return output;
}

LargeVector<N> operator+<let N: int>(LargeVector<N> lhs, LargeVector<N> rhs) {
    LargeVector<N> output;
    for (int i = 0; i < N; i++) {
        output[i] = lhs[i] + rhs[i];
    }
    return output;
}

LargeMatrix<M, N> operator+<let M: int, let N: int>(LargeMatrix<M, N> lhs, LargeMatrix<M, N> rhs) {
    LargeMatrix<M, N> output;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output.value[i][j] = lhs.value[i][j] + rhs.value[i][j];
        }
    }
    return output;
}

struct LinearLayer<let N_IN : int, let N_OUT : int> {
    LargeMatrix<N_OUT, N_IN> weight;
    LargeVector<N_OUT> bias;

    LargeVector<N_OUT> forward(LargeVector<N_IN> input) {
        return mul(weight, input) + bias;
    }
};

LargeVector<N> relu<let N : int>(LargeVector<N> value) {
    LargeVector<N> output;
    for (int i = 0; i < N; i++) {
        output[i] = max(0.0, value[i]);
    }
    return output;
}

LargeVector<N> tanh<let N : int>(LargeVector<N> value) {
    LargeVector<N> output;
    for (int i = 0; i < N; i++) {
        output[i] = tanh(value[i]);
    }
    return output;
}

LargeVector<N> exp<let N : int>(LargeVector<N> value) {
    LargeVector<N> output;
    for (int i = 0; i < N; i++) {
        output[i] = exp(value[i]);
    }
    return output;
}
