#include <iostream>
#include <opencv2/core/core.hpp>             // OpenCV基础数据类型Mat，Vec和Point等，以及算数运算
#include <opencv2/features2d/features2d.hpp> // OpenCV提取特征点、绘制关键点和匹配点
#include <opencv2/highgui/highgui.hpp>       // OpenCV图像显示相关
#include <opencv2/calib3d/calib3d.hpp>       // OpenCV本质矩阵、基础矩阵、单应矩阵和位姿恢复，解PnP及标定
#include <Eigen/Core>                        // Eigen的Matrix类和Array类，基础的线性代数运算和数组操作
#include <Eigen/SVD>                         // Eigen的SVD分解算法
#include <chrono>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

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

// 使用ceres自动求导，构造CostFunctor
struct PnpReprojectionCostFunctor
{
    // 已知量为世界坐标系下的3d特征点坐标和其在相机2坐标系下的像素坐标
    // 有参构造函数
    PnpReprojectionCostFunctor(Point3f pt_3d_1, Point3f pt_3d_2)
        : _pt_3d_1(pt_3d_1), _pt_3d_2(pt_3d_2) {}

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

        // 计算残差
        residuals[0] = T(_pt_3d_1.x) - p_transformed[0];
        residuals[1] = T(_pt_3d_1.y) - p_transformed[1];   
        residuals[2] = T(_pt_3d_1.z) - p_transformed[2];   
        return true; 
    }

    // 定义create()函数，创建相应的自动求导CostFunction
    static ceres::CostFunction * Create(const Point3f pt_3d_1, const Point3f pt_3d_2)
    {
        return (new ceres::AutoDiffCostFunction<PnpReprojectionCostFunctor, 3, 3, 3>(
            new PnpReprojectionCostFunctor(pt_3d_1, pt_3d_2)));
    }

    Point3f _pt_3d_1;
    Point3f _pt_3d_2;
};

void bundleAdjustmentCeres(
    const vector<Point3f> &points_2d,
    const vector<Point3f> &points_3d,
    Mat &R,
    Mat &t);

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
    vector<Point3f> pts1_3d_cv, pts2_3d_cv;
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
        pts1_3d_cv.push_back(Point3f(pt1_cam.x * dd1, pt1_cam.y * dd1, dd1));
        pts2_3d.push_back(Eigen::Vector3d(pt2_cam.x * dd2, pt2_cam.y * dd2, dd2));
        pts2_3d_cv.push_back(Point3f(pt2_cam.x * dd2, pt2_cam.y * dd2, dd2));
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
    cout << "calling bundle adjustment by ceres..." << endl;
    bundleAdjustmentCeres(pts1_3d_cv, pts2_3d_cv, R, t);
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

void bundleAdjustmentCeres(
    const vector<Point3f> &points_3d_1,
    const vector<Point3f> &points_3d_2,
    Mat &R,
    Mat &t)
{
    // 1. 定义参数块并赋初值
    double r_ceres[3] = {0, 0, 0};
    double t_ceres[3] = {0, 0, 0};

    // 2. 定义ceres优化问题
    ceres::Problem problem;
    // 3. 添加残差块
    for(int i = 0; i < points_3d_1.size(); i++)
    {
        ceres::CostFunction * cost_function = PnpReprojectionCostFunctor::Create(points_3d_1[i], points_3d_2[i]);
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