#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

// cost functor 函子/仿函数/函数对象 f(x)的计算模型
struct AutoCurveFittingCostFunctor
{
    AutoCurveFittingCostFunctor(double x, double y) : _x(x), _y(y) {}

    // 残差的计算
    template <typename T>
    bool operator()(
        const T *const abc, // 模型参数，有4维
        T *residual) const
    {
        // y-[exp(ax^2+bx+c)+dx]
        residual[0] = T(_y) - (ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]) + abc[3] * T(_x));
        return true;
    }

    const double _x, _y;
};

// 解析求导直接定义cost function，不用像自动求导，要先定义cost functor
class AnalyticCurveFittingCostFunction : public ceres::SizedCostFunction<1, 4>
{
public:
    AnalyticCurveFittingCostFunction(const double x, const double y) : _x(x), _y(y) {}
    virtual ~AnalyticCurveFittingCostFunction() {}
    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        const double a = parameters[0][0];
        const double b = parameters[0][1];
        const double c = parameters[0][2];
        const double d = parameters[0][3];
        double exp_num = ceres::exp(a * _x * _x + b * _x + c);
        residuals[0] = _y - (exp_num + d * _x) ;

        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            jacobians[0][0] = -_x * _x * exp_num;
            jacobians[0][1] = -_x * exp_num;
            jacobians[0][2] = -exp_num;
            jacobians[0][3] = -_x;
        }
        return true;
    }
private:
    const double _x, _y;
};

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0, dr = 2.0;  // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0, de = -1.0; // 估计参数值初值
    int N = 100;                          // 数据点数
    double w_sigma = 1.0;                 // 噪声sigma值

    cv::RNG rng;                   // OpenCV的随即数生成器
    vector<double> x_data, y_data; // 数据
    for (int i = 0; i < N; i++)
    {
        double xi = i / 100.0;
        x_data.push_back(xi);
        y_data.push_back((exp(ar * xi * xi + br * xi + cr) + dr * xi) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc_auto[4] = {ae, be, ce, de};

    // 使用Ceres自动求导
    // 构建自动求导最小二乘问题
    ceres::Problem problem_auto;
    for (int i = 0; i < N; i++)
    {
        // 向问题中添加残差块
        problem_auto.AddResidualBlock(
            // 使用自动求导，模板参数是误差函数对象类，残差维度，参数维度，和CurveFittingCost类中定义的一致
            new ceres::AutoDiffCostFunction<AutoCurveFittingCostFunctor, 1, 4>(
                new AutoCurveFittingCostFunctor(x_data[i], y_data[i])),
            nullptr, // 核函数，这里不使用，为空
            abc_auto      // 待估计的参数
        );
    }

    // 配置求解器
    ceres::Solver::Options options_auto;                            // 有很多配置项可选
    options_auto.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // 增量方程如何求解
    options_auto.minimizer_progress_to_stdout = true;               // 输出到stdout

    ceres::Solver::Summary summary_auto; // 优化信息
    auto t1 = chrono::steady_clock::now();
    ceres::Solve(options_auto, &problem_auto, &summary_auto); // 开始优化
    auto t2 = chrono::steady_clock::now();
    auto time_used_auto = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "auto diff solve time cost: " << time_used_auto.count() << "seconds." << endl;

    // 输出结果

    cout << summary_auto.BriefReport() << endl;
    cout << "auto diff estimated a, b, c,d = ";
    for (auto a : abc_auto)
    {
        cout << a << " ";
    }
    cout << endl;

    // 使用Ceres解析求导
    // 构建解析求导最小二乘问题
    double abc_ana[4] = {ae, be, ce, de};
    ceres::Problem problem_analytic;
    for (int i = 0; i < N; i++)
    {
        problem_analytic.AddResidualBlock(
            new AnalyticCurveFittingCostFunction(x_data[i], y_data[i]),
            nullptr,
            abc_ana);
    }

    // 配置求解器
    ceres::Solver::Options options_analytic;
    options_analytic.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options_analytic.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary_analytic;
    t1 = chrono::steady_clock::now();
    ceres::Solve(options_analytic, &problem_analytic, &summary_analytic);
    t2 = chrono::steady_clock::now();
    auto time_used_analytic = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "analytic diff solve time cost: " << time_used_analytic.count() << endl;

    // 输出结果
    cout << summary_analytic.BriefReport() << endl;
    cout << "analytic diff estimated a, b, c, d = ";
    for (auto a : abc_ana)
    {
        cout << a << " ";
    }
    cout << endl;

    return 0;
}