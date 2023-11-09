#include <iostream>
#include <chrono>
using namespace std;

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace Eigen;

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值初值
    int N = 100;                          // 数据点数
    double w_sigma = 1.0;                 // 噪声Sigma值
    double inv_w_sigma = 1.0 / w_sigma;
    cv::RNG rng; // OpenCV随机数产生器

    vector<double> x_data, y_data; // 数据
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // 开始Gauss-Newton迭代
    int iterations = 100;           // 迭代次数
    double cost = 0, last_cost = 0; // 本次迭代和上一次迭代的cost

    auto t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++)
    {
        Matrix3d H = Matrix3d::Zero(); // H = J * J^T 其中J为梯度向量
        Vector3d b = Vector3d::Zero(); // b = -J * error 其中error为观测值和模型预估值的误差
        cost = 0; // 每次迭代前，将记录损失的变量置0

        for (int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i]; // 第i个数据点
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J;                                         // 参数的梯度向量
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce); // De/Da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);      // De/Db
            J[2] = -exp(ae * xi * xi + be * xi + ce);           // De/Dc

            H += J * J.transpose();
            b += -error * J;
            cost += error * error;
        }

        // 求解线性方程 Hx = b
        Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= last_cost)
        {
            cout << "cost: " << cost << ">= last cost: " << last_cost << ", break." << endl;
            break;
        }

        // 更新参数
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        last_cost = cost;

        cout << iter << " total cost: " << cost << ", \t\tupdate: " << dx.transpose()
             << "\t\t estimated params: " << ae << " " << be << " " << ce << endl;
    }
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    cout << "solve time cost = " << time_used.count() << "seconds" << endl;
    cout << "estimated abc =" << ae << " " << be << " " << ce << endl;

    return 0;
}