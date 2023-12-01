#include <iostream>
using namespace std;

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry> // Eigen的旋转变换、欧式变换、仿射变换和射影变换
using namespace Eigen;

// 本程序演示了Eigen几何模块的使用方法
int main(int argc, char **argv)
{
    // Eigen/Geometry模块提供了各种旋转和平移的表示
    // 3D旋转矩阵直接使用Matrix3d或Matrix3f

    Matrix3d rotation_matrix = Matrix3d::Identity();
    // 旋转向量使用AngleAxis，它底层不直接是Matrix，但运算可以当作矩阵（因为重载了运算符）
    AngleAxisd rotation_vector(M_PI_4, Vector3d(0, 0, 1)); // 沿Z轴旋转45度
    cout.precision(3);
    cout << "roatation matrix = " << endl;
    cout << rotation_vector.matrix() << endl; // 用matrix()转换成矩阵，也可以直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix(); // toRotationMatrix()和matirx()是一样的
    cout << endl;

    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1, 0, 0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;
    // 或者用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout << "(1, 0, 0) after rotation (by rotation matrix) = " << v_rotated.transpose() << endl;
    cout << endl;

    // 欧拉角：可以将旋转矩阵直接转换为欧拉角
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX顺序，即yaw-pitch-roll顺序
    cout << "yaw-pitch-roll = " << euler_angles.transpose() << endl;
    cout << endl;

    // 欧式变换矩阵使用Eigen:Isometry
    Isometry3d T = Isometry3d::Identity(); //虽然称为3D,实质上是4 * 4矩阵
    T.rotate(rotation_vector); // 按照ratation_vector进行旋转

    // T.pretranslate(Vector3d(1, 3, 4)); // 把平移向量设为(1，3，4)
    cout << "Transform matrix from rotaion_verctor= " << endl;
    cout << T.matrix() << endl;
    cout << "(1, 0, 0) after transform matrix =" << (T * v).transpose() << endl;
    cout << endl;

    T = Isometry3d::Identity(); // 清除上一个rotate()的效果
    T.rotate(rotation_matrix); // 按照rotation_matrix进行旋转
    cout << "Transform matrix from rotation_matirx = " << endl;
    cout << T.matrix() << endl;
    cout << "(1, 0, 0) after transform matrix =" << (T * v).transpose() << endl;
    cout << endl;

    // 对于仿射变换和射影变换，使用Eigen::Affine3d和Eigen::Projective3d即可

    // 四元数
    // 可以直接把AngleAxis赋值给四元数，反之亦然;也可以在有参构造中传入彼此用于初始化
    Quaterniond q(rotation_vector); // Quaterniond的有参构造，参数是AngleAxisd类对象
    cout << "quaternion from rotaiton vector = "
         << q.coeffs().transpose() << endl; // 请注意coeffs()的顺序是(x, y, z, w)，w为实部，前三者为虚部
    // 也可以把旋转矩阵赋给它
    q = Quaterniond(rotation_matrix); // Quaterniond的有参构造哦，参数是Matrix3d类对象
    cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << endl;

    // 使用四元数旋转一个向量，使用重载的乘法即可
    v_rotated = q * v; // 注意数学上是qvq^{-1}，返回3维向量
    cout << "(1, 0, 0) afer quaternion rotation = " << v_rotated.transpose() << endl;
    // 用常规四元数乘法表示，则计算如下，返回四元数
    cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;

    return 0;
}