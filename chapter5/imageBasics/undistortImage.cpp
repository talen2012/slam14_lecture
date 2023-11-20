#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
#include <string>
string image_file = "./imageBasics/distorted.png";

int main(int argc, char **argv)
{
    // 去畸变的部分代码，也可直接调用OpenCV的去畸变
    // 畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file, 0); // imread函数第二个参数为ImreadModes, -1为按图像本身的属性，0为灰度，1为BGR(默认值），2为深度图
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1); // 用来存储去畸变后的图

    // 计算去畸变后的图像内容
    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            /**
             * 按照公式计算点(u, v)对应到畸变图像中的坐标(u_distorted, v_distorted);
             * 畸变发生在归一化成像平面，先转换到归一化成像平面坐标
             * 在归一化成相平面畸变后，再转换回像素平面坐标
            */
            double x = (u - cx) / fx, y = (v - cy) / cy;
            double r = sqrt(x * x + y * y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            double u_distorted = x_distorted * fx + cx;
            double v_distorted = y_distorted * fy + cy;

            // 赋值（最近邻插值）
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows)
            {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int)v_distorted, (int)u_distorted);
            }
            else
            {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    }

    // 画出去畸变后的图像
    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey(0);

    return 0;
}