#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// #include "extra.h" // use this if in OpenCV2

using namespace std;
using namespace cv;

/*******************************************
 * 本程序演示了如何使用2D-2D特征匹配估计相机运动
 *******************************************/
void find_feature_matches(const Mat &img_1,
                          const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

// 相较于十四讲源码，额外传入后三个参数，记录通过内参矩阵解算的结果
// 第4～6个参数，记录通过光心和焦距解算的结果
void pose_estimation_2d2d(vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches,
                          Mat &R,
                          Mat &t,
                          Mat &essential_matrix,
                          Mat &R_K,
                          Mat &t_K,
                          Mat &essential_matrix_K);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "error! proper usage: pose_esimation_2d2d img_1 img_2" << endl;
        return 1;
    }

    // -- 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // -- 检测和匹配关键点
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // -- 估计两张图像之间的运动
    // 由于本质矩阵的尺度等价性，R自身具有约束，所以认为t具有一个尺度
    // 在单目SLAM中，对两张图像的t进行归一化处理，相当于固定了尺度，
    // 虽然不知道实际距离是多少，但以这个t为1，这就是单目相机的初始化，因此下面获得的t模长为1
    // 初始化之后，就可以用3D-2D计算相机运动了
    Mat R, R_K, t, t_K, essential_matrix, essential_matrix_K;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t, essential_matrix, R_K, t_K, essential_matrix_K);

    // -- 验证E = t^R*scale
    Mat t_hat = (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
              t.at<double>(2, 0), 0, -t.at<double>(0, 0),
              -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    cout << "t^R: " << endl;
    cout << t_hat * R << endl;
    cout << " t^R / essential_matrix: " << endl;
    cout << (t_hat * R) / essential_matrix << endl;

    Mat t_K_hat = (Mat_<double>(3, 3) << 0, -t_K.at<double>(2, 0), t_K.at<double>(1, 0),
                 t_K.at<double>(2, 0), 0, -t_K.at<double>(0, 0),
                 -t_K.at<double>(1, 0), t_K.at<double>(0, 0), 0);
    cout << "t_K^R_K: " << endl;
    cout << t_K_hat * R_K << endl;
    cout << " t_K^R_K / essential_matrix_K: " << endl;
    cout << (t_K_hat * R_K) / essential_matrix_K << endl;

    // -- 验证对极约束
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    double d_total = 0, d_K_total = 0;
    for (DMatch m : matches)
    {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        // 归一化成像平面齐次坐标如下
        Mat x1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat x2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        // 对极约束，值应为0
        Mat d1 = x2.t() * t_hat * R * x1;  // 注意d是个标量，但是其类型还需写成Mat, 因为=后边是一个矩阵表达式
        cout << "essential_matrix epipolar constraint = " << d1 << endl;
        d_total += abs(d1.at<double>(0, 0)); // 不能用d1.data[0], 因为data是uchar*的指针，而Mat的计算结果是double型的
        Mat d2 = x2.t() * t_K_hat * R_K * x1; // 注意d是个标量，但是其类型还需写成Mat, 因为=后边是一个矩阵表达式
        cout << "essential_matrix_K epipolar constraint = " << d2 << endl;
        d_K_total += abs(d2.at<double>(0, 0));
    }

    cout << "essential_matrix total epipolar constraint = " << d_total << endl;
    cout << "essential_matrix_K total epipolar constraint = " << d_K_total << endl;

    return 0;
}

void find_feature_matches(const Mat &img_1,
                          const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches)
{
    // -- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // -- 第一步：检测Oriented FAST角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // -- 第二步：计算角点位置处的BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    cout << "descriptor的高和宽: " << descriptors_1.rows << " " << descriptors_2.cols * descriptors_1.elemSize() << endl;
    // 可见，描述子矩阵是以关键点数为高，以描述子长度为宽

    // -- 第三步：对两幅图像中的BRIEF描述子进行初步匹配
    vector<DMatch> preliminary_matches;
    matcher->match(descriptors_1, descriptors_2, preliminary_matches);

    // -- 第四步：匹配点筛选
    double min_dist = 10000, max_dist = 0;
    // // 找出所有匹配点的最小距离和最大距离
    for (auto match : preliminary_matches)
    {
        double dist = match.distance;
        if (match.distance < min_dist)
        {
            min_dist = match.distance;
        }
        if (match.distance > max_dist)
        {
            max_dist = match.distance;
        }
    }

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    // // 当描述子的距离大于两倍的最小距离时，即认为匹配有误
    // // 但有时最小距离会非常小，设定一个经验值作为下限
    for(auto match: preliminary_matches)
    {
        if (match.distance <= max( 2 * min_dist, 30.0))
        {
            matches.push_back(match);
        }
    }
}

// 坐标转换：像素平面坐标，转换到归一化成像平面坐标
Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                   (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_2d2d(vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches,
                          Mat &R,
                          Mat &t,
                          Mat &essential_matrix,
                          Mat &R_K,
                          Mat &t_K,
                          Mat &essential_matrix_K)
{
    // 相机内参，TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // 把匹配点转换为vector(Point2f)的形式
    vector<Point2f> points_1;
    vector<Point2f> points_2;

    for (auto match: matches)
    {
        points_1.push_back(keypoints_1[match.queryIdx].pt);
        points_2.push_back(keypoints_2[match.trainIdx].pt);
    }

    // 计算基础矩阵
    Mat fundamental_matrix = findFundamentalMat(points_1, points_2, FM_8POINT);
    cout << "fundamental matrix: " << endl;
    cout << fundamental_matrix << endl;

    // 计算本质矩阵
    Point2d principal_point(325.1, 249.7); // 相机光心， TUM dataset标定值
    double focal_length = 521;             // 相机焦距， TUM dataset标定值
    // 焦距对E, R，t的解算影响巨大，曾将焦距改为520，只差了1，结果截然不同
    essential_matrix = findEssentialMat(points_1, points_2, focal_length, principal_point);
    cout << "传入光心和焦距, essential matrix: " << endl;
    cout << essential_matrix << endl;
    // // 上文不是给出了内参矩阵么，为什么不直接用呢
    essential_matrix_K = findEssentialMat(points_1, points_2, K);
    cout << "传入内参矩阵, essential matrix_K:" << endl;
    cout << essential_matrix_K << endl;
    cout << "传入光心和焦距求得的本质矩阵 / 传入内参矩阵求得的: " << endl;
    cout << essential_matrix / essential_matrix_K << endl;

    // 验证F和E相差了内参矩阵, 输出结果并不满足，奇怪
    cout << "F / (K^(-T)*E*K^(-1)): " << endl;
    cout << fundamental_matrix / ((K.inv().t()) * essential_matrix_K * (K.inv())) << endl;

    // -- 从本质矩阵中恢复旋转和平移信息
    // 此函数仅在OpenCV3中提供
    recoverPose(essential_matrix, points_1, points_2, R, t,focal_length, principal_point);
    recoverPose(essential_matrix_K, points_1, points_2, K, R_K, t_K);
    cout << "R的数据类型: " << R.type() << endl;
    cout << "t的数据类型: " << t.type() << endl;
    cout << "R is: " << endl;
    cout << R << endl;
    cout << "R_K is: " << endl;
    cout << R_K << endl;
    cout << "R / R_K is: " << endl;
    cout << R / R_K << endl;
    cout << "t is: " << endl;
    cout << t << endl;
    cout << "t_K is: " << endl;
    cout << t_K << endl;
    cout << "t / t_K is: " << endl;
    cout << t / t_K << endl;

    // 计算单应矩阵
    Mat homography_matrix = findHomography(points_1, points_2, RANSAC, 3);
    cout << "homography matrix: " << endl;
    cout << homography_matrix << endl;


}
