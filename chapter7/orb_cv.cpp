#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <chrono>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "usage: orb_cv img1 img2" << endl;
        return 1;
    }

    // -- 读取图像
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // -- 初始化
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();       // Ptr是OpenCV提供的指针对象模板，类似C++11的智能指针，无需手动释放指针指向的内存；
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(); // cv::ORB是实施ORB关键点检测和描述子提取的类
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // --第一步：检测oriented FAST角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // virtual void cv::Feature2D::detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask = noArray())
    detector->detect(img_1, keypoints_1);
    cout << "图1的关键点个数: " << keypoints_1.size() << endl;
    detector->detect(img_2, keypoints_2);

    // --第二步：根据角点位置计算BRIEF描述子
    // void compute(cv::InputArrayOfArrays images, std::vector<std::vector<cv::KeyPoint>> &keypoints, cv::OutputArrayOfArrays descriptors)
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "exract ORB cost = " << time_used.count() << " seconds." << endl;

    cv::Mat outimg_1;
    // cv::Scalar_是OpenCV中一个四元素列向量模板，常用于传递像素值，cv::Scalar为其double型
    // 在drawKeyPoint函数中，cv::Scalar::all(-1)表示特征点绘制颜色为随机
    cv::drawKeypoints(img_1, keypoints_1, outimg_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("img_1 ORB features", outimg_1);

    // -- 第三步：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    // cv::DMatch是一个类，用于记录描述子匹配结果，DMatch(int _queryIdx, int _trainIdx, int _imgIdx, float _distance);
    vector<cv::DMatch> matches;
    t1 = chrono::steady_clock::now();
    // match( InputArray queryDescriptors, InputArray trainDescriptors, CV_OUT std::vector<DMatch>& matches, InputArray mask=noArray() )
    // 对每一个queryDescriptors，在trainDescriptors中寻找最近邻，每个matches[i]描述其中的一对
    // knnMatch函数还需一个参数k,指定返回的近邻的个数，不仅仅返回最近邻
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds." << endl;
    // -- 匹配点筛选
    // 计算最小距离和最大距离
    // minmax_element是STL算法库里的函数
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const cv::DMatch &m1, const cv::DMatch &m2)
                                  { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。
    // 但有时最小距离会非常小，所以要设置一个经验值30作为下限
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if(matches[i].distance <= max(min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // 第五步：绘制匹配结果
    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    cv::namedWindow("all matches", cv::WINDOW_NORMAL);
    cv::imshow("all matches", img_match);
    cv::imshow("good matches", img_goodmatch);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}