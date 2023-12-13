#include <iostream>
#include <opencv2/core/core.hpp>                          // OpenCV基础数据类型Mat，Vec和Point等，以及算数运算
#include <opencv2/features2d/features2d.hpp>              // OpenCV提取特征点，绘制关键点，绘制匹配点
#include <opencv2/highgui/highgui.hpp>                    // OpenCV图像显示相关
#include <opencv2/calib3d/calib3d.hpp>                    // OpenCV本质矩阵、基础矩阵、单应矩阵和位姿恢复，解PnP及标定
#include <Eigen/Core>                                     // Eigen的Matrix和Array类，基础的线性代数运算和数组操作
#include "sophus/se3.hpp"                                 // Sophus李群李代数包，SE3群
#include <ceres/ceres.h>                                  // 优化库Ceres
#include <ceres/rotation.h>                               // Ceres在BA时，位姿用旋转向量和平移向量表示，坐标转换需要rotaion.h中的ceres::angleAxisRotatePoint(rotation_vector, p_origin, p_transformed)函数

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

// 使用ceres自动求导，构造CostFunctor
struct PnpReprojectionCostFunctor
{
    // 已知量为世界坐标系下的3d特征点坐标和其在相机2坐标系下的像素坐标
    // 有参构造函数
    PnpReprojectionCostFunctor(Point2f pt_2d, Point3f pt_3d)
        : _pt_3d_1(pt_2d), _pt_3d_2(pt_3d) {}

    // 定义仿函数,传入待优化参数和残差
    template <typename T>
    bool operator()(const T* const rotation_vector,
                    const T* const translation_vector,
                    T* residuals) const
    {
        T p_origin[3], p_transformed[3];
        p_origin[0] = T(_pt_3d_2.x);
        p_origin[1] = T(_pt_3d_2.y);
        p_origin[2] = T(_pt_3d_2.z);

        // 计算相机2坐标系下的3d点坐标
        // // 先旋转
        ceres::AngleAxisRotatePoint(rotation_vector, p_origin, p_transformed);
        // // 再平移
        p_transformed[0] += translation_vector[0];
        p_transformed[1] += translation_vector[1];
        p_transformed[2] += translation_vector[2];

        // 投影到相机2归一化平面
        T x_cam2 = p_transformed[0] / p_transformed[2];
        T y_cam2 = p_transformed[1] / p_transformed[2];
        // 相机内参
        double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
        // 重投影
        T x_pix2 = fx * x_cam2 + cx;
        T y_pix2 = fy * y_cam2 + cy;
        // 计算残差(重投影误差)
        residuals[0] = T(_pt_3d_1.x) - x_pix2;
        residuals[1] = T(_pt_3d_1.y) - y_pix2;   
        return true; 
    }

    // 定义create()函数，创建相应的自动求导CostFunction
    static ceres::CostFunction * Create(const Point2f pt_2d, const Point3f pt_3d)
    {
        return (new ceres::AutoDiffCostFunction<PnpReprojectionCostFunctor, 2, 3, 3>(
            new PnpReprojectionCostFunctor(pt_2d, pt_3d)));
    }

    Point2f _pt_3d_1;
    Point3f _pt_3d_2;
};

void bundleAdjustmentCeres(
    const vector<Point2f> &points_2d,
    const vector<Point3f> &points_3d,
    Mat &R,
    Mat &t);

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

    cout << "calling bundle adjustment by ceres..." << endl;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentCeres(pts_2d, pts_3d, R, t);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by ceres cost time: " << time_used.count() << " seconds." << endl;
    cout << "R=" << endl
         << R << endl;
    cout << "t=" << endl
         << t << endl;
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

void bundleAdjustmentCeres(
    const vector<Point2f> &points_2d,
    const vector<Point3f> &points_3d,
    Mat &R,
    Mat &t)
{
    // 1. 定义参数块并赋初值
    double r_ceres[3] = {0, 0, 0};
    double t_ceres[3] = {0, 0, 0};

    // 2. 定义ceres优化问题
    ceres::Problem problem;
    // 3. 添加残差块
    for(int i = 0; i < points_2d.size(); i++)
    {
        ceres::CostFunction * cost_function = PnpReprojectionCostFunctor::Create(points_2d[i], points_3d[i]);
        problem.AddResidualBlock(cost_function, nullptr, r_ceres, t_ceres);
    }

    // 4. 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    
    //5. 求解
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << summary.BriefReport() << endl;
    cout << "optimation cost time: " << time_used.count() << " seconds. " << endl;

    // 旋转向量通过罗德里格斯公式转换为旋转矩阵
    Mat r_ceres_cv = (Mat_<double>(3, 1) << r_ceres[0], r_ceres[1], r_ceres[2]);
    cv::Rodrigues(r_ceres_cv, R);

    t = (Mat_<double>(3, 1) << t_ceres[0], t_ceres[1], t_ceres[2]);
}

