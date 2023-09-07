#include <iostream>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

int main(int argc, char **argv)
{
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    q1.normalize();
    q2.normalize();
    cout << "quaterniond q1 =" << q1.coeffs().transpose() << endl;
    cout << "quaterniond q2 =" << q2.coeffs().transpose() << endl;
    cout << endl;

    Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);
    Vector3d p1(0.5, 0, 0.2);

    Isometry3d t1w(q1), t2w(q2);
    t1w.pretranslate(t1);
    t2w.pretranslate(t2);

    Vector3d p2 = t2w * t1w.inverse() * p1;
    cout << p2.transpose() << endl;

    return 0;
}