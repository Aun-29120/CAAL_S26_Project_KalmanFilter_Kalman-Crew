// ─────────────────────────────────────────────────────────────────────────────
// lkf.cpp  —  Linear Kalman Filter for full-body 3D gait
// Self-contained: no external headers required except STL.
// ─────────────────────────────────────────────────────────────────────────────

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int    NUM_JOINTS     = 23;
static constexpr int    STATE_PER_JOINT = 12;
static constexpr int    STATE_DIM      = NUM_JOINTS * STATE_PER_JOINT; // 276
static constexpr int    MEAS_PER_JOINT = 3;
static constexpr int    MEAS_DIM       = NUM_JOINTS * MEAS_PER_JOINT;  // 69
static constexpr double DT             = 0.01;   // 100 Hz

// LKF tuning parameters
static constexpr double SIGMA_J = 0.5;    // process noise jerk std-dev (m/s^3)
static constexpr double SIGMA_R = 0.3593; // measurement noise std-dev (m)

static const std::vector<std::string> JOINT_NAMES = {
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
};

// ─────────────────────────────────────────────────────────────────────────────
// Matrix class — dense, row-major, heap-allocated
// ─────────────────────────────────────────────────────────────────────────────
class Matrix {
public:
    int rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, double fill = 0.0)
        : rows(r), cols(c), data(r * c, fill) {}

    double& operator()(int r, int c)       { return data[r * cols + c]; }
    double  operator()(int r, int c) const { return data[r * cols + c]; }

    static Matrix identity(int n) {
        Matrix I(n, n, 0.0);
        for (int i = 0; i < n; ++i) I(i, i) = 1.0;
        return I;
    }

    Matrix operator+(const Matrix& B) const {
        assert(rows == B.rows && cols == B.cols);
        Matrix C(rows, cols);
        for (int i = 0; i < rows * cols; ++i) C.data[i] = data[i] + B.data[i];
        return C;
    }
    Matrix operator-(const Matrix& B) const {
        assert(rows == B.rows && cols == B.cols);
        Matrix C(rows, cols);
        for (int i = 0; i < rows * cols; ++i) C.data[i] = data[i] - B.data[i];
        return C;
    }
    Matrix operator*(double s) const {
        Matrix C(rows, cols);
        for (int i = 0; i < rows * cols; ++i) C.data[i] = data[i] * s;
        return C;
    }
    Matrix operator*(const Matrix& B) const {
        assert(cols == B.rows);
        Matrix C(rows, B.cols, 0.0);
        for (int i = 0; i < rows; ++i)
            for (int k = 0; k < cols; ++k) {
                double a = data[i * cols + k];
                if (a == 0.0) continue;
                for (int j = 0; j < B.cols; ++j)
                    C.data[i * B.cols + j] += a * B.data[k * B.cols + j];
            }
        return C;
    }
    Matrix transpose() const {
        Matrix T(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                T(j, i) = (*this)(i, j);
        return T;
    }

    // LDL^T solve: A*X = B  (A must be symmetric positive-definite)
    static Matrix solve(const Matrix& A, const Matrix& B) {
        int n = A.rows, m = B.cols;
        assert(A.cols == n && B.rows == n);
        std::vector<double> D(n, 0.0);
        Matrix L = Matrix::identity(n);
        std::vector<double> v(n, 0.0);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < j; ++i) v[i] = L(j, i) * D[i];
            D[j] = A(j, j);
            for (int i = 0; i < j; ++i) D[j] -= L(j, i) * v[i];
            if (std::abs(D[j]) < 1e-14) D[j] = 1e-14;
            for (int k = j + 1; k < n; ++k) {
                L(k, j) = A(k, j);
                for (int i = 0; i < j; ++i) L(k, j) -= L(k, i) * v[i];
                L(k, j) /= D[j];
            }
        }
        Matrix y(n, m, 0.0);
        for (int col = 0; col < m; ++col)
            for (int i = 0; i < n; ++i) {
                y(i, col) = B(i, col);
                for (int k = 0; k < i; ++k) y(i, col) -= L(i, k) * y(k, col);
            }
        Matrix z(n, m, 0.0);
        for (int i = 0; i < n; ++i)
            for (int col = 0; col < m; ++col)
                z(i, col) = y(i, col) / D[i];
        Matrix x(n, m, 0.0);
        for (int col = 0; col < m; ++col)
            for (int i = n - 1; i >= 0; --i) {
                x(i, col) = z(i, col);
                for (int k = i + 1; k < n; ++k) x(i, col) -= L(k, i) * x(k, col);
            }
        return x;
    }
};
inline Matrix operator*(double s, const Matrix& M) { return M * s; }

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────
struct JointMeasurement { double x, y, z; };
struct Frame { std::vector<JointMeasurement> joints; };

// ─────────────────────────────────────────────────────────────────────────────
// CSV loader
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<Frame> load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<Frame> frames;
    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string token;
        Frame f;
        f.joints.resize(NUM_JOINTS);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            std::getline(ss, token, ','); f.joints[j].x = std::stod(token);
            std::getline(ss, token, ','); f.joints[j].y = std::stod(token);
            std::getline(ss, token, ','); f.joints[j].z = std::stod(token);
        }
        frames.push_back(f);
    }
    return frames;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build system matrices
// ─────────────────────────────────────────────────────────────────────────────
static Matrix make_F_full(double dt) {
    double dt2 = dt*dt, dt3 = dt2*dt;
    Matrix Fj(12, 12, 0.0);
    for (int b = 0; b < 3; ++b) {
        int o = b * 4;
        Fj(o+0,o+0)=1; Fj(o+0,o+1)=dt; Fj(o+0,o+2)=0.5*dt2; Fj(o+0,o+3)=dt3/6.0;
        Fj(o+1,o+1)=1; Fj(o+1,o+2)=dt; Fj(o+1,o+3)=0.5*dt2;
        Fj(o+2,o+2)=1; Fj(o+2,o+3)=dt;
        Fj(o+3,o+3)=1;
    }
    Matrix F(STATE_DIM, STATE_DIM, 0.0);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int o = j * STATE_PER_JOINT;
        for (int r = 0; r < 12; ++r)
            for (int c = 0; c < 12; ++c)
                F(o+r, o+c) = Fj(r, c);
    }
    return F;
}

static Matrix make_Q_full(double dt, double sigma_j) {
    double dt2=dt*dt, dt3=dt2*dt, dt4=dt3*dt, dt5=dt4*dt, dt6=dt5*dt;
    double s2 = sigma_j * sigma_j;
    double q[4][4] = {
        {dt6/36.0, dt5/12.0, dt4/6.0, dt3/6.0},
        {dt5/12.0, dt4/4.0,  dt3/2.0, dt2/2.0},
        {dt4/6.0,  dt3/2.0,  dt2,     dt      },
        {dt3/6.0,  dt2/2.0,  dt,      1.0     }
    };
    Matrix Qj(12, 12, 0.0);
    for (int b = 0; b < 3; ++b) {
        int o = b * 4;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                Qj(o+r, o+c) = s2 * q[r][c];
    }
    Matrix Q(STATE_DIM, STATE_DIM, 0.0);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int o = j * STATE_PER_JOINT;
        for (int r = 0; r < 12; ++r)
            for (int c = 0; c < 12; ++c)
                Q(o+r, o+c) = Qj(r, c);
    }
    return Q;
}

static Matrix make_H_full() {
    Matrix H(MEAS_DIM, STATE_DIM, 0.0);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int ro = j * 3, co = j * STATE_PER_JOINT;
        H(ro+0, co+0) = 1.0; // px
        H(ro+1, co+4) = 1.0; // py
        H(ro+2, co+8) = 1.0; // pz
    }
    return H;
}

static Matrix make_R_full(double sigma_r) {
    Matrix R(MEAS_DIM, MEAS_DIM, 0.0);
    double s2 = sigma_r * sigma_r;
    for (int i = 0; i < MEAS_DIM; ++i) R(i, i) = s2;
    return R;
}

static Matrix frame_to_z(const Frame& f) {
    Matrix z(MEAS_DIM, 1, 0.0);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        z(j*3+0, 0) = f.joints[j].x;
        z(j*3+1, 0) = f.joints[j].y;
        z(j*3+2, 0) = f.joints[j].z;
    }
    return z;
}

static Matrix init_state(const Frame& f) {
    Matrix x(STATE_DIM, 1, 0.0);
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int o = j * STATE_PER_JOINT;
        x(o+0, 0) = f.joints[j].x;
        x(o+4, 0) = f.joints[j].y;
        x(o+8, 0) = f.joints[j].z;
    }
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// LKF predict and update
// ─────────────────────────────────────────────────────────────────────────────
static void lkf_predict(Matrix& x, Matrix& P,
                        const Matrix& F, const Matrix& Q) {
    x = F * x;
    P = F * P * F.transpose() + Q;
}

static void lkf_update(Matrix& x, Matrix& P,
                       const Matrix& z,
                       const Matrix& H, const Matrix& R) {
    Matrix y   = z - H * x;
    Matrix PHt = P * H.transpose();
    Matrix S   = H * PHt + R;
    Matrix KT  = Matrix::solve(S, PHt.transpose());
    Matrix K   = KT.transpose();
    x = x + K * y;
    int n = P.rows;
    Matrix IKH = Matrix::identity(n) - K * H;
    P = IKH * P * IKH.transpose() + K * R * K.transpose();
}

// ─────────────────────────────────────────────────────────────────────────────
// Write output CSV
// ─────────────────────────────────────────────────────────────────────────────
static void write_csv(const std::string& path,
                      const std::vector<Matrix>& states) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot write: " + path);
    f << "frame";
    for (int j = 0; j < NUM_JOINTS; ++j) {
        std::string n = JOINT_NAMES[j];
        f << "," << n << "_px," << n << "_vx," << n << "_ax," << n << "_jx";
        f << "," << n << "_py," << n << "_vy," << n << "_ay," << n << "_jy";
        f << "," << n << "_pz," << n << "_vz," << n << "_az," << n << "_jz";
    }
    f << "\n";
    for (int t = 0; t < (int)states.size(); ++t) {
        f << t;
        const Matrix& x = states[t];
        for (int i = 0; i < STATE_DIM; ++i) f << "," << x(i, 0);
        f << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string noisy_path  = (argc > 1) ? argv[1] : "3D Full Body Humain Gait Walking Dataset (Noisy Values).csv";
    std::string output_path = (argc > 2) ? argv[2] : "lkf_output.csv";

    std::cout << "[LKF] Loading data from: " << noisy_path << "\n";
    std::vector<Frame> frames = load_csv(noisy_path);
    int T = (int)frames.size();
    std::cout << "[LKF] Frames loaded: " << T << "\n";

    std::cout << "[LKF] Building matrices (276x276)...\n";
    Matrix F = make_F_full(DT);
    Matrix Q = make_Q_full(DT, SIGMA_J);
    Matrix H = make_H_full();
    Matrix R = make_R_full(SIGMA_R);

    Matrix x = init_state(frames[0]);
    Matrix P = Matrix::identity(STATE_DIM) * 1.0;

    std::vector<Matrix> states;
    states.reserve(T);

    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < T; ++t) {
        if (t % 500 == 0)
            std::cout << "[LKF] Processing frame " << t << "/" << T << "\n";
        lkf_predict(x, P, F, Q);
        Matrix z = frame_to_z(frames[t]);
        lkf_update(x, P, z, H, R);
        states.push_back(x);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[LKF] Done. Time: " << std::chrono::duration<double>(t1-t0).count() << " s\n";

    std::cout << "[LKF] Writing: " << output_path << "\n";
    write_csv(output_path, states);
    std::cout << "[LKF] Complete.\n";
    return 0;
}
