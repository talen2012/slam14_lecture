#include <iostream>
using namespace std;

#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 重置
    virtual void setToOriginImpl() override
    {
        // _estimate是g2o::BaseVertex的成员变量，通过estimate()来访问
        _estimate << 0, 0, 0;
    }

    // 更新
    virtual void oplusImpl(const double *update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    // 存盘和读盘，留空
    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};

// 误差模型 模板参数：观测值维度、类型、连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // 计算曲线模型误差
    virtual void computeError() override
    {
        // addVertex()后，会给图结构中的_vertices赋值
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate(); // BaseVertex类的estimate()方法获取当前优化参数值
        // _measurement通过setMeasurement()传入，_measurement和_error是BaseEdge的成员变量
        // setMeasurement()是BaseEdge的成员函数
        _error(0, 0) = _measurement - exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    // 计算雅可比矩阵
    virtual void linearizeOplus() override
    {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        // _jacobianOpulsXi是BaseUnaryEdge的成员变量，尺寸为误差维度 * 顶点维度(优化变量维度)
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}

public:
    double _x; // x值， y值为_measurement
};

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;    // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;   // 估计参数值
    int N = 100;                            // 数据点数
    double w_sigma = 1.0;                   // 噪声sigma值
    double inv_w_sigma = 1.0 / w_sigma;
    cv::RNG rng; // OpenCV随机数产生器

    vector<double> x_data, y_data;          // 数据
    for (int i = 0; i < N; i++)
    {
        double xi = i / 100.0;
        x_data.push_back(xi);
        y_data.push_back(exp(ar * xi * xi + br * xi + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // 构建图优化，先设定g2o
    // 每个误差项优化变量维度3，误差值维度为1
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
    // 线性求解器类型
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    // explicit OptimizationAlgorithmGaussNewton::OptimizationAlgorithmGaussNewton(std::unique_ptr<solver>)
    // 第一步：通过make_unique创建一个线性求解器LinearSolverDense(Cholesky分解，下三角和其转置的乘积)
    // 第二步：通过创建的线性求解器来创建BlockSolver，该线性求解器是BlockSolver的一部分
    // 第三步：从BlockSolver派生总求解器solver。Solver是optimizationWithHessian的一部分
    //        optimizationWithHessian再进一步派生出GN，LM，DogLeg算法
    auto optimation_GM = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // 第四步：创建稀疏优化器(SparseOptimizer)
    g2o::SparseOptimizer optimizer;        // SparseOptimizer是图模型，内部有一个优化算法
    optimizer.setAlgorithm(optimation_GM); // 设置优化算法
    optimizer.setVerbose(true);            // 打开调试输出

    // 往图中增加顶点
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    // 往途中增加边
    for (int i = 0; i < N; i++)
    {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);             // 设置连接的顶点
        edge->setMeasurement(y_data[i]);   // 设置观测数值
        // 设置信息矩阵，协方差矩阵之逆
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }

    // 执行优化
    cout << "start optimization" << endl;
    auto t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << "seconds." << endl;

    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}