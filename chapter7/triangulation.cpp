#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// 寻找匹配点
void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

// Dmatch数据结构拆解成两个Point2f向量（由于KeyPoint的pt属性是Points2f类型）
void convertDmatchsToPoint2fs(const vector<KeyPoint> &keypoints_1,
                              const vector<KeyPoint> &keypoints_2,
                              const vector<DMatch> &matches,
                              vector<Point2f> &points_1,
                              vector<Point2f> &points_2);

// 用本质矩阵估计位姿R，t
void pose_estimation_2d2d(
    const vector<Point2f> &points_1,
    const vector<Point2f> &points_2,
    Mat &R, Mat &t);

// 三角化
void triangulation(
    const vector<Point2f> &points_1,
    const vector<Point2f> &points_2,
    const Mat &R, const Mat &t,
    vector<Point3d> &points);

// 像素坐标转相机归一化坐标
Point2f pixel2cam(const Point2d &p, const Mat &K);

/// 作图用
inline cv::Scalar get_color(double depth);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "error! proper usage: trianulation img_1 img_2" << endl;
        return 1;
    }

    // -- 读取图像
    Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 绘制匹配图
    Mat img_matches;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1), Scalar::all(0));
    cv::imshow("matches", img_matches);

    // -- 将匹配好的点对，拆解成Point2f构成的向量，一共拆成了两个
    // -- 像素坐标都是保存为float的，6位有效数字足够了
    vector<Point2f> points_1, points_2;
    convertDmatchsToPoint2fs(keypoints_1, keypoints_2, matches, points_1, points_2);

    // -- 估计两幅图像之间的相机运动， 注意，这里获得的t模长为1
    Mat R, t;
    pose_estimation_2d2d(points_1, points_2, R, t);

    // -- 三角化
    vector<Point3d> points3d_worldCord; // 每一对匹配点产生一个points3d，代表世界坐标系中的点
    triangulation(points_1, points_2, R, t, points3d_worldCord);

    // -- 验证三角化点与特征点的重投影关系
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat img_1_plot = img_1.clone();
    Mat img_2_plot = img_2.clone();

    for (int i = 0; i < points3d_worldCord.size(); i++)
    {
        // 第一个图
        // 在第一个图上用颜色表示特征点深度
        double depth_1 = points3d_worldCord[i].z; // 本实验中第一个相机坐标系和世界坐标系相同，故不用转换坐标
        cout << i << ". depth1 = " << depth_1;
        cv::circle(img_1_plot, points_1[i], 2, get_color(depth_1), 2);

        // 第二个图
        // // 先将世界坐标系中的点，转换到相机2的坐标系下
        Mat points3d_2cord = R * (Mat_<double>(3, 1) << points3d_worldCord[i].x, points3d_worldCord[i].y, points3d_worldCord[i].z) + t;
        double depth_2 = points3d_2cord.at<double>(2, 0); // 注意，上式中的t为double型，因此获得的points3d_2cord也是double型的
        cout << " depth2 = " << depth_2 << endl;
        cv::circle(img_2_plot, points_2[i], 2, get_color(depth_2), 2);
    }

    cv::imshow("img 1", img_1_plot);
    cv::imshow("img 2", img_2_plot);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches)
{
    // 初始化
    Mat descriptors1, descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // --第一步：检测oriented FAST角点
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // --第二步：根据角点位置计算BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors1);
    descriptor->compute(img_2, keypoints_2, descriptors2);

    // --第三步：匹配描述子
    vector<DMatch> preliminary_matches;
    matcher->match(descriptors1, descriptors2, preliminary_matches);

    // --第四步：筛选描述子
    double min_dist = 10000, max_dist = 0;
    for (DMatch match : preliminary_matches)
    {
        double dist = match.distance;
        if (dist < min_dist)
        {
            min_dist = dist;
        }
        if (dist > max_dist)
        {
            max_dist = dist;
        }
    }
    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    for (DMatch match : preliminary_matches)
    {
        if (match.distance <= max(min_dist * 2, 30.0))
        {
            matches.push_back(match);
        }
    }
}

void convertDmatchsToPoint2fs(const vector<KeyPoint> &keypoints_1,
                              const vector<KeyPoint> &keypoints_2,
                              const vector<DMatch> &matches,
                              vector<Point2f> &points_1,
                              vector<Point2f> &points_2)
{
    for (DMatch match : matches)
    {
        points_1.push_back(keypoints_1[match.queryIdx].pt);
        points_2.push_back(keypoints_2[match.trainIdx].pt);
    }
}

void pose_estimation_2d2d(
    const vector<Point2f> &points_1,
    const vector<Point2f> &points_2,
    Mat &R, Mat &t)
{
    // 相机内参,TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    Point2d principal_point(325.1, 249.7); // 相机主点, TUM dataset标定值
    int focal_length = 521;                // 相机焦距, TUM dataset标定值

    // --计算本质矩阵
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points_1, points_2, focal_length, principal_point);

    // --从本质矩阵中恢复R， t
    // 利用SVD分解E，需要匹配点对坐标，去筛选出正确的解
    recoverPose(essential_matrix, points_1, points_2, R, t, focal_length, principal_point);
}

Point2d pixel2cam(const Point2f &p, const Mat &K)
{
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                   (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void triangulation(
    const vector<Point2f> &points_1,
    const vector<Point2f> &points_2,
    const Mat &R, const Mat &t,
    vector<Point3d> &points)
{
    // 设定相机1，2的位姿矩阵，注意，两者都是世界坐标下下的
    Mat T1 = (Mat_<double>(3, 4) << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);

    Mat T2 = (Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    // 将像素坐标转换到归一化成像平面坐标
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2d> pts_cam_1, pts_cam_2;
    for (int i = 0; i < points_1.size(); i++)
    {
        pts_cam_1.push_back(pixel2cam(points_1[i], K));
        pts_cam_2.push_back(pixel2cam(points_2[i], K));
    }

    Mat pts_4d; // cv::triangulatePoints, 在传入的参数中按列记录世界坐标系下点的齐次坐标
    // void cv::triangulatePoints(cv::InputArray projMatr1, cv::InputArray projMatr2, cv::InputArray projPoints1, cv::InputArray projPoints2, cv::OutputArray points4D)
    cv::triangulatePoints(T1, T2, pts_cam_1, pts_cam_2, pts_4d);

    // 转化成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<double>(3, 0); // 按最后一个元素归一化
        Point3d p(x.at<double>(0, 0),
                  x.at<double>(1, 0),
                  x.at<double>(2, 0));
        points.push_back(p);
    }
}

inline cv::Scalar get_color(double depth) // cv::Scalar是一个double型的1*4向量
{
    float up_th = 16, low_th = 7, th_range = up_th - low_th;
    if (depth > up_th)
    {
        depth = up_th;
    }
    if (depth < low_th)
    {
        depth = low_th;
    }
    // 返回点的颜色，越深越蓝，越浅越红
    return cv::Scalar(255 * (depth - low_th) / th_range, 0, 255 * (1 - (depth - low_th) / th_range));
}
