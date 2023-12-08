#include <iostream>
#include <opencv2/core/core.hpp>             // OpenCV基础数据类型Mat，Vec和Point等，以及算数运算
#include <opencv2/features2d/features2d.hpp> // OpenCV提取特征点、绘制关键点和匹配点
#include <opencv2/highgui/highgui.hpp>       // OpenCV图像显示相关
#include <opencv2/calib3d/calib3d.hpp>       // OpenCV本质矩阵、基础矩阵、单应矩阵和位姿恢复，解PnP及标定
#include <Eigen/Core>                        // Eigen的Matrix类和Array类，基础的线性代数运算和数组操作
#include <Eigen/SVD>                         // Eigen的SVD分解算法
#include <chrono>
#include <g2o/core/base_vertex.h>                         // G2O顶点基类，自定义顶点需要继承该类，并重载setToOriginImpl()重置顶点函数和oplusImpl()更新顶点函数
#include <g2o/core/base_unary_edge.h>                     // G2O单边基类，自定义单边需要继承该类，并创建有参构造函数，重载computeError()计算残差函数和linearizeOplus()计算雅各比矩阵函数
#include <g2o/core/sparse_block_matrix.h>                 // G2O核心优化器
#include <g2o/core/block_solver.h>                        // G2O块求解器，包含一个线性求解器和一个稀疏求解器
#include <g2o/solvers/dense/linear_solver_dense.h>        // G2O稠密矩阵线性求解器
#include <g2o/core/solver.h>                              // G2O求解器，继承自块求解器
#include <g2o/core/optimization_algorithm_gauss_newton.h> // G2O优化算法，高斯牛顿
#include "sophus/se3.hpp"                                 // Sophus李群李代数包，SE3群

using namespace std;
using namespace cv;

string img1_path = "1.png";
string img2_path = "2.png";
string img_depth1_path = "1_depth.png";
string img_depth2_path = "2_depth.png";

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// 定义3d点的数据类型, 由于ICP的SVD和非线性优化方法都基于Eigen
// 因此放弃采用十四讲源码里的vector<Point3f>
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void pose_estimation_3d3d(
    const VecVector3d &pts1,
    const VecVector3d &pts2,
    Mat &R, Mat &t);

// 自定义G2O顶点类和单边类
// Vertex
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
    // 注意，可以看到这里的更新方式是原估计量左乘dx的李群，并不是直接在原估计量上加上dx
    // 说明本案例中并非直接对待优化量求的导，即，不是直接到位姿的李代数求的导
    // 其实，这就是欧式变换群李代数求导的左乘扰动模型
    // 细节上来说，就是将误差在指定点关于扰动为0展开，也就是f(0+dx)
    // 由此也可以看出非线性优化中，并不一定是直接对待优化量求导获得下一次迭代的增量
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    // 读盘和存盘，留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};
// unary edge
class EdgePosition : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> // 模板参数<观测值维度，观测值类型，连接顶点类型>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 解决Eigen库数据结构内存对齐问题

    // 1. 创建有参构造函数
    EdgePosition(Eigen::Vector3d pt_3d) : _pt_3d(pt_3d) {}

    // 2. 重载computeError()计算残差函数
    virtual void computeError() override
    {
        // _vertices是hyper_graph中Edge类的成员，由setVertex(index, vertex指针)成员函数指定，最终被BaseUnaryEdge继承
        VertexPose *v = static_cast<VertexPose *>(_vertices[0]); // 基类指针转换为派生类指针，可能会越界访问，除非你知道自己在干什么
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pt_3d_aftertrans = T * _pt_3d;
        // _measurement和_error是BaseEdge的成员变量
        // _measurement通过setMeasurement()传入，setMeasurement()是BaseEdge的成员函数
        _error = _measurement - pt_3d_aftertrans;
    }

    // 3. 重载linearizeOplus()计算雅各比矩阵函数，不重载则默认使用数值求导方式
    virtual void linearizeOplus() override
    {
        VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pt_3d_aftertrans = T * _pt_3d;
        double _X = pt_3d_aftertrans[0];
        double _Y = pt_3d_aftertrans[1];
        double _Z = pt_3d_aftertrans[2];
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(pt_3d_aftertrans);
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}

private:
    Eigen::Vector3d _pt_3d;
};

void bundleAdjustment(
    const VecVector3d &pts1,
    const VecVector3d &pts2,
    Mat &R, Mat &t);

int main(int argc, char **argv)
{
    //-- 读取图像
    Mat img_1 = imread(img1_path, CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(img2_path, CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat img_depth_1 = imread(img_depth1_path, CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
    Mat img_depth_2 = imread(img_depth2_path, CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    VecVector3d pts1_3d, pts2_3d;
    for (DMatch match : matches)
    {
        Point2f pt1_pixel = keypoints_1[match.queryIdx].pt;
        Point2f pt2_pixel = keypoints_2[match.trainIdx].pt;
        ushort d1 = img_depth_1.ptr<ushort>(int(pt1_pixel.y))[int(pt1_pixel.x)]; // 关键点的坐标是float格式的，当作索引时需转换为整形
        ushort d2 = img_depth_2.ptr<ushort>(int(pt2_pixel.y))[int(pt2_pixel.x)];
        if (d1 == 0 || d2 == 0) // bad depth
        {
            continue;
        }
        Point2d pt1_cam = pixel2cam(pt1_pixel, K);
        Point2d pt2_cam = pixel2cam(pt2_pixel, K);
        double dd1 = double(d1) / 5000.0;
        double dd2 = double(d2) / 5000.0;
        pts1_3d.push_back(Eigen::Vector3d(pt1_cam.x * dd1, pt1_cam.y * dd1, dd1));
        pts2_3d.push_back(Eigen::Vector3d(pt2_cam.x * dd2, pt2_cam.y * dd2, dd2));
    }
    cout << "3d-3d pairs: " << pts1_3d.size() << endl;

    // SVD分解法
    Mat R, t;
    cout << "ICP via SVD..." << endl;
    pose_estimation_3d3d(pts1_3d, pts2_3d, R, t);
    cout << "R12 = " << R << endl; // 第二帧到第一帧的变换
    cout << "t12 = " << t << endl;
    cout << "R21 = " << R.t() << endl; // 第一帧到第二帧的变换
    cout << "t21 = " << -R.t() * t << endl;
    cout << endl;

    // 非线性优化法
    cout << "calling bundle adjustment..." << endl;
    bundleAdjustment(pts1_3d, pts2_3d, R, t);
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << endl;

    // // 验证pt1 = R * pt2 + t
    cout << "验证pt1 = R * pt2 + t: " << endl;
    for (int i = 0; i < 5; i++)
    {
        cout << "pt1 = " << pts1_3d[i].transpose() << endl;
        cout << "pt2 = " << pts2_3d[i].transpose() << endl;
        cout << "R * pt2 + t = " << (R * Vec3d(pts2_3d[i][0], pts2_3d[i][1], pts2_3d[i][2]) + t).t() << endl;
        cout << endl;
    }

}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
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
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
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

void pose_estimation_3d3d(const VecVector3d &pts1,
                          const VecVector3d &pts2,
                          Mat &R, Mat &t)
{
    // center of mass
    Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d p2 = Eigen::Vector3d::Zero();
    int N = pts1.size();
    for (int i = 0; i < N; i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 /= N;
    p2 /= N;

    // compute q2*q1^T
    // 注意，十四讲源码里计算的是q1*q2^T，即第二帧到第一帧的变换，并不是之前习惯的从相机1的坐标转换到相机2的坐标
    VecVector3d q1(N), q2(N); // 开辟N个元素空间，并初始化为0，如要指定初始化值，则传入第二个参数，q1和q2为原3d点减去质心
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();

    for (int i = 0; i < N; i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
        W += q1[i] * q2[i].transpose();
    }

    cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    Eigen::Matrix3d _R = U * (V.transpose());
    if (_R.determinant() < 0)
    {
        _R = -_R;
    }
    Eigen::Vector3d _t = p1 - _R * p2;

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) << _R(0, 0), _R(0, 1), _R(0, 2),
         _R(1, 0), _R(1, 1), _R(1, 2),
         _R(2, 0), _R(2, 1), _R(2, 2));
    t = (Mat_<double>(3, 1) << _t(0, 0), _t(1, 0), _t(2, 0));
}

void bundleAdjustment(
    const VecVector3d &pts1,
    const VecVector3d &pts2,
    Mat &R, Mat &t)
{
    // -- 第一步：设定优化算法
    // 1. 定义一个块求解器类型
    // template <int _PoseDim, int _LandmarkDim> struct BlockSolverTraits
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    // 2. 基于定义的块求解器类型，定义相应的线性求解器类型
    // typedef Eigen::Matrix<number_t, PoseDim, PoseDim, Eigen::ColMajor> PoseMatrixType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 模板参数：H矩阵的数据类型
    // 3. 选择梯度下降方法，可从G-N，L-M和Dog-Leg中选择，并将求解器指针作为参数传入
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
    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d()); // 优化变量赋初值
    optimizer.addVertex(vertex_pose);
    // 2. 添加边
    for (size_t i = 0; i < pts1.size(); i++)
    {
        auto pt1 = pts1[i];
        auto pt2 = pts2[i];
        EdgePosition *edge_positon = new EdgePosition(pt2);
        edge_positon->setVertex(0, vertex_pose);
        edge_positon->setMeasurement(pt1); // 设定观测值
        edge_positon->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge_positon);
    }

    // -- 第四步：执行优化
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // 1. 初始化
    optimizer.initializeOptimization();
    // 2. 指定迭代次数
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    cout << "optimization cost time: " << time_used.count() << " seconds." << endl;
    cout << endl;
    cout << "after optimization: " << endl;
    cout << "T = " << endl;
    cout << vertex_pose->estimate().matrix() << endl;

    // convert to cv::Mat
    Eigen::Matrix3d _R = vertex_pose->estimate().rotationMatrix();
    Eigen::Vector3d _t = vertex_pose->estimate().translation();
    R = (Mat_<double>(3, 3) << _R(0, 0), _R(0, 1), _R(0, 2),
         _R(1, 0), _R(1, 1), _R(1, 2),
         _R(2, 0), _R(2, 1), _R(2, 2));
    t = (Mat_<double>(3, 1) << _t[0], _t[1], _t[2]);
}