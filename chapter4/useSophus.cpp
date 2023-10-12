#include <iostream>
using namespace std;

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"
using namespace Eigen;

// 本程序演示Sophus的基本用法
int main(int argc, char **argv)
{
    // 沿Z轴转90度的旋转矩阵
    Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
    // 或者四元数
    Quaterniond q(R);
    Sophus::SO3d SO3_R(R); // Sophus::SO3可以直接从旋转矩阵构造
    Sophus::SO3d SO3_q(q); // 也可以通过四元数构造
    // 二者是等价的
    cout << "SO(3) from matrix: \n"
         << SO3_R.matrix() << endl;
    cout << "SO(3) from quaternion: \n"
         << SO3_q.matrix() << endl;
    cout << "they are equal" << endl;

    // 使用对数映射获得它的李代数
    Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;
    // hat为向量到反对称矩阵，这是Sophus::SO3提供的一个类成员函数
    cout << "so3 hat = \n"
         << Sophus::SO3d::hat(so3) << endl;
    // 相对的，vee为反对称矩阵到向量，这是Sophus::SO3提供的一个类成员函数
    cout << "so3 hat vee = \n"
         << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

     // 增量扰动模型的更新
    Vector3d update_so3(1e-4, 0, 0); // 假设更新量为这么多
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    cout << "SO3左乘扰动量后 = \n"
         << SO3_updated.matrix() << endl;

    cout << "**********************************" << endl;
    // SE(3)操作大同小异
    Vector3d t(1, 0, 0); //沿X轴平移1
    Sophus::SE3d SE3_Rt(R, t); // 从R和t构造SE3
    Sophus::SE3d SE3_qt(q, t); // 从q和t构造SE3
    cout << "SE3 from R, t = \n"
         << SE3_Rt.matrix() << endl;
    cout << "SE3 from q, t = \n"
         << SE3_qt.matrix() << endl;

    // 李代数se(3)是一个六维向量，为方便起见先typedef一下
    typedef Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl;
    // 观察输出，可发现在Sophus中，se(3)平移在前，旋转在后
    // 同样地，有hat和vee两个算符
    cout << "se3 hat = " << Sophus::SE3d::hat(se3) << endl;
    cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

    // 最后，演示更新
    Vector6d update_se3; // 更新量
    update_se3.setZero();
    update_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3左乘扰动后 = \n"
         << SE3_updated.matrix() << endl;

    cout << "**********************************" << endl;
    // 定义尺度因子s为2
    double s = 2;
    Sophus::RxSO3d sR(s * R); // Sim3不支持从R，t, s构造，需要先构造一个RxSO3对象，内容是s*R
    Sophus::Sim3d Sim3_sRt(sR, t); // 从sR和t构造Sim(3)
    cout << "Sim3_sRt from sR, t = \n"
         << Sim3_sRt.matrix() << endl;

    // 李代数sim(3)是一个7维向量，为方便起见先typedef一下
    typedef Matrix<double, 7, 1> Vector7d;
    Vector7d sim3 = Sim3_sRt.log();
    cout << "sim3 = " << sim3.transpose() << endl;
    // 观察输出，可发现Sophus中，sim(3)平移在前，旋转在后,sigma最后

    return 0;
}