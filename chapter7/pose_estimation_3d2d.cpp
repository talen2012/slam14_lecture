#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>              // OpenCV提取特征点，绘制关键点，绘制匹配点
#include <opencv2/highgui/highgui.hpp>                    // OpenCV图像显示相关
#include <opencv2/calib3d/calib3d.hpp>                    // OpenCV本质矩阵、基础矩阵、单应矩阵和位姿恢复，解PnP及标定
#include <Eigen/Core>                                     // Eigen的Matrix和Array类，基础的线性代数运算和数组操作
#include <g2o/core/base_vertex.h>                         // G2O顶点基类，自定义顶点需要继承该类，并重载setToOriginImpl()重置顶点函数和oplusImpl()更新顶点函数
#include <g2o/core/base_unary_edge.h>                     // G2O单边基类，自定义单边需要继承该类，并添加有参构造函数、重载computeError()计算残差函数和lineari_ZeOplus()计算梯度函数
#include <g2o/core/sparse_optimizer.h>                    // G2O核心优化器
#include <g2o/core/block_solver.h>                        // G2O块求解器，包含线性求解器和系数求解器
#include <g2o/core/solver.h>                              // G2O求解器，继承自块求解器
#include <g2o/core/optimization_algorithm_gauss_newton.h> // G2O优化算法 高斯牛顿
#include <g2o/solvers/dense/linear_solver_dense.h>        // G2O稠密矩阵线性求解器
#include "sophus/se3.hpp"                                 // Sophus李群李代数，SE3群
#include <chrono>

using namespace std;
using namespace cv;

string img1_path = "1.png";
string img2_path = "2.png";
string img_depth1_path = "1_depth.png";
string img_depth2_path = "2_depth.png";

void find_feature_matches(
    const Mat &img_1,
    const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);

// BA by gauss-newton
void bundleAdjustmentSelfGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);

int main(int argc, char **argv)
{
    //-- 读取图像
    Mat img_1 = imread(img1_path, CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(img2_path, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat d1 = imread(img_depth1_path, CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch match : matches)
    {
        Point2f p1_pixel = keypoints_1[match.queryIdx].pt;
        Point2f p2_pixel = keypoints_2[match.trainIdx].pt;

        ushort d = d1.ptr<unsigned short>(int(p1_pixel.y))[int(p1_pixel.x)]; // 浮点数要转换成整数
        if (d == 0)                                                          // bad depth
        {
            continue;
        }
        float dd = d / 5000.0; // 数据来源于TUM数据集，深度有5000的因子，5000代表1m
        Point2d p1_cam = pixel2cam(p1_pixel, K);
        pts_3d.push_back(Point3f(p1_cam.x * dd, p1_cam.y * dd, dd));
        pts_2d.push_back(p2_pixel);
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;
    cout << endl;

    cout << "solve pnp in opencv..." << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "R=" << endl
         << R << endl;
    cout << "t=" << endl
         << t << endl;
    
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
    cout << endl;

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i)
    {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "calling bundle adjustment by self gauss newton..." << endl;
    Sophus::SE3d pose_gn; // 不需要初始化？
    t1 = chrono::steady_clock::now();
    bundleAdjustmentSelfGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by self gauss newton cost time: " << time_used.count() << " seconds." << endl;
    cout << endl;

    cout << "calling bundle adjustment by g2o..." << endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
    return 0;
}

void find_feature_matches(
    const Mat &img_1,
    const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches)
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
        {
            min_dist = dist;
        }
        if (dist > max_dist)
        {
            max_dist = dist;
        }
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void bundleAdjustmentSelfGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose)
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++)
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0; // 每次迭代需将cost置0
        // compute cost
        for (int i = 0; i < points_3d.size(); i++)
        {
            Eigen::Vector3d posi_1 = pose * points_3d[i];
            double _X = posi_1[0];
            double _Y = posi_1[1];
            double _Z = posi_1[2];
            double inv_Z = 1.0 / _Z;
            double inv_Z2 = inv_Z * inv_Z;
            Eigen::Vector2d proj(fx * _X / _Z + cx, fy * _Y / _Z + cy);

            Eigen::Vector2d e = points_2d[i] - proj;

            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_Z,
                0,
                fx * _X * inv_Z2,
                fx * _X * _Y * inv_Z2,
                -fx - fx * _X * _X * inv_Z2,
                fx * _Y * inv_Z,
                0,
                -fy * inv_Z,
                fy * _Y * inv_Z2,
                fy + fy * _Y * _Y * inv_Z2,
                -fy * _X * _Y * inv_Z2,
                -fy * _X * inv_Z;
            // 十四讲里的是标量变元函数的例子，那里的J实际上是梯度向量，一阶泰勒展开为f(x)+J^T*dx
            // 对于向量变元，J指的是雅各比矩阵，一阶泰勒展开为F(x)+J*dx
            // 因此对于向量变元，十四讲的结论需要对J取转置
            H += J.transpose() * J; 
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b); // 使用ldlt方法进行矩阵分解，等价于cholesky分解，适用于正定的对称矩阵

        // 1. 方程无解，停止迭代，本次迭代无效
        if (isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }
        // 2. 损失增加，停止迭代，本次迭代无效
        if (iter > 0 && cost >= lastCost)
        {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // 若没有1，2两种错误
        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;
        cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;

        // 3.梯度增量模长小于1e-6, 停止迭代
        // 但本次迭代是有效的，因此需在记录上述结果之后再停止迭代
        if (dx.norm() < 1e-6)
        {
            // converge
            cout << "norm of dx is smaller than 1e-6, stop iteration." << endl;
            break;
        }
    }

    cout << "pose by self G-N: \n"
         << pose.matrix() << endl;
}

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> // 模板参数<优化变量维度，优化变量数据类型>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 解决Eigen库数据结构内存对齐问题

    // 1. 重载setToOriginImpl()重置顶点函数
    virtual void setToOriginImpl() override
    {
        // _estimate是g2o::BaseVertex的成员变量，通过estimate()来访问
        _estimate = Sophus::SE3d();
    }

    // 2. 重载oplusImpl()更新顶点函数
    // // left multiplication on SE3
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    // 存盘和读盘，留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> // 模板参数<观测值维度，类型，连接顶点数据类型>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 1. 创建有参构造函数
    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

    // 2. 重载computeError()计算残差函数
    virtual void computeError() override
    {
        // addVertex()后，会给图结构中的_vertices赋值
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]); // 基类指针转换为派生类指针，可能会越界访问，除非你知道自己在干什么
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        // _measurement和_error是BaseEdge的成员变量
        // _measurement通过setMeasurement()传入，setMeasurement()是BaseEdge的成员函数
        _error = _measurement - pos_pixel.head<2>();
    }

    // 3. 重载linearizeOplus()计算雅各比矩阵函数，不重载则采用默认的数值求导方式
    virtual void linearizeOplus() override
    {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi
            << -fx / Z,
            0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
            0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}

private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose)
{

    // -- 第一步：设定优化算法
    // 1. 定义一个块求解器类型
    // template <int _PoseDim, int _LandmarkDim> struct BlockSolverTraits
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;           // pose is 6, landmark is 3
    // 2. 基于块求解器类型，定义相应的线性求解器类型
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; 
    // 3. 选择梯度下降方法，可以从GN, LM, DogLeg 中选，并将求解器指针作为参数传入
    auto optim_method = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    // -- 第二步：设定核心优化器
    g2o::SparseOptimizer optimizer; 
    // 1. 绑定优化算法
    optimizer.setAlgorithm(optim_method); 
    // 2. 打开优化过程输出
    optimizer.setVerbose(true);     

    // -- 第三步：设定优化图结构
    // 1. 添加顶点
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d()); // 优化变量赋初值
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // 2. 添加边
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i)
    {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d); // 设定观测值
        edge->setInformation(Eigen::Matrix2d::Identity()); // 信息矩阵
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n"
         << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate();
}
