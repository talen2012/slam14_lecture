#include <iostream>
using namespace std;

#include <ctime>
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>
using namespace Eigen;

#define MATRIX_SIZE 50

int main(int argc, char **argv)
{
    // Eigen中所有向量和矩阵都是Eigen::Matrix，它是一个模板类，前三个参数为数据类型，行，列
    // 声明一个2*3的float矩阵
    Matrix<float, 2, 3> matrix_23f;
    Vector3d v_3d;

    Matrix<float, 3, 1> v_3f;

    Matrix3d matrix_33d = Matrix3d::Zero();

    // 如果不确定矩阵大小，可以使用动态大小的矩阵，Dynamic是Eigen定义的一个宏，值是-1
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;

    matrix_23f << 1, 2, 3, 4, 5, 6;

    cout << "matrix 2x3 form 1 to 6: \n"
         << matrix_23f << endl;

    // 用()访问矩阵中的元素
    cout << "print matrix 2x3: \n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << matrix_23f(i, j) << "\t";
        cout << endl;
    }

    // 矩阵和向量相乘（实际上仍是矩阵和矩阵）
    v_3d << 3, 2, 1;
    v_3f << 4, 5, 6;

    // 但是在Eigen里不能混合两种不同类型的矩阵，像下边这样是错的
    // Matrix<double, 2, 1> result_wrong_type = matrix_23f * v_3d;

    // 应该显式转换,cast函数会返回一个新的对象，并非修改原矩阵元素
    Matrix<double, 2, 1> result = matrix_23f.cast<double>() * v_3d;
    cout << "[1, 2, 3; 4, 5, 6] * [3, 2, 1] = " << result.transpose() << endl;

    Matrix<float, 2, 1> result2 = matrix_23f * v_3f;
    cout << "[1, 2, 3; 4, 5, 6] * [4, 5, 6] = " << result2.transpose() << endl;

    // 同样不能搞错矩阵的维度
    // Matrix<double, 2, 3> result_wrong_dimension = matrix_23f.cast<double>() * v_3d;

    // 一些矩阵运算
    matrix_33d = Matrix3d::Random(); // 随机数矩阵
    cout << "random matrix: \n"
         << matrix_33d << endl;
    cout << "transpose: \n"
         << matrix_33d.transpose() << endl; // 转置
    cout << "sum: \n"
         << matrix_33d.sum() << endl; // 各元素和
    cout << "trace: \n"
         << matrix_33d.trace() << endl; // 迹
    cout << "times 10: \n"
         << matrix_33d * 10 << endl; // 数乘
    cout << "inverse: \n"
         << matrix_33d.inverse() << endl; // 逆
    cout << "det: \n"
         << matrix_33d.determinant() << endl; // 行列式

    // 特征值
    // 实对称阵一定可以相似对角化
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33d.transpose() * matrix_33d);
    cout << "Eigen values = \n"
         << eigen_solver.eigenvalues() << endl;

    cout << "Eigen vectors = \n"
         << eigen_solver.eigenvectors() << endl;

    // 解方程
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_Nd; // MATRIX_SIZE = 50
    matrix_Nd = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_Nd = matrix_Nd * matrix_Nd.transpose();
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();  // 计时,返回程序占用的CPU时钟数
    // 直接求逆

    Matrix<double, MATRIX_SIZE, 1> x = matrix_Nd.inverse() * v_Nd;
    cout << "time of normal inverse is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    time_stt = clock();
    // 通常用矩阵分解来求解，比如QR分解，速度会快很多
    x = matrix_Nd.colPivHouseholderQr().solve(v_Nd);
    cout << "time of Qr decompostion is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    // 对于正定矩阵，还可以用cholesky分解来解方程
    time_stt = clock();
    x = matrix_Nd.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    return 0;
}
