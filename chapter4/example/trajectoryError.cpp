#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace std;

string groundtruth_file = "example/groundtruth.txt";
string estimated_file = "example/estimated.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

TrajectoryType ReadTrajectory(const string &path);

int main(int argc, char **argv) {
  const int MAXPATH = 250;
  char buffer[MAXPATH];
  getcwd(buffer, MAXPATH);
  printf("The current directory is: %s\n", buffer);

  TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
  TrajectoryType estimated = ReadTrajectory(estimated_file);
  assert(!groundtruth.empty() && !estimated.empty());
  assert(groundtruth.size() == estimated.size());

  // compute rmse
  double rmse = 0;
  for (size_t i = 0; i < estimated.size(); i++) {
    Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
    double error = (p2.inverse() * p1).log().norm();
    rmse += error * error;
  }
  rmse = rmse / double(estimated.size());
  rmse = sqrt(rmse);
  cout << "RMSE = " << rmse << endl;

  // compute ate_trans(绝对平移误差)
  double ate_trans = 0;
  for (size_t i = 0; i < estimated.size(); i++)
  {
    Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
    double error1 = (p2.inverse() * p1).translation().norm();// translation()方法获得平移部分，rotationMatrix()方法获得旋转矩阵
    ate_trans += error1 * error1;
  }
  ate_trans /= double(estimated.size());
  ate_trans = sqrt(ate_trans);
  cout << "ATE_trans = " << ate_trans << endl;

  // compute rpe_all(相对位姿误差)
  double rpe_all = 0;
  int delta_t = 10;
  for (size_t i = 0; i < estimated.size() - delta_t; i++)
  {
    Sophus::SE3d p1 = estimated[i], p2 = estimated[i + delta_t];
    Sophus::SE3d p3 = groundtruth[i], p4 = groundtruth[i + delta_t];
    double error2 = ((p3.inverse() * p4).inverse() * (p1.inverse() * p2)).log().norm();
    rpe_all += error2 * error2;
  }
  rpe_all /= double(estimated.size() - delta_t);
  rpe_all = sqrt(rpe_all);
  cout << "RPE_all = " << rpe_all << endl;

  // compute rpe_trans(相对平移误差)
  double rpe_trans = 0;
  delta_t = 10;
  for (size_t i = 0; i < estimated.size() - delta_t; i++)
  {
    Sophus::SE3d p1 = estimated[i], p2 = estimated[i + delta_t];
    Sophus::SE3d p3 = groundtruth[i], p4 = groundtruth[i + delta_t];
    double error3 = ((p3.inverse() * p4).inverse() * (p1.inverse() * p2)).translation().norm();
    rpe_trans += error3 * error3;
  }
  rpe_trans /= double(estimated.size() - delta_t);
  rpe_trans = sqrt(rpe_trans);
  cout << "RPE_trans = " << rpe_trans << endl;

  DrawTrajectory(groundtruth, estimated);
  return 0;
}

TrajectoryType ReadTrajectory(const string &path)
{
  ifstream fin(path);
  TrajectoryType trajectory;
  if (!fin) {
    cerr << "trajectory " << path << " not found." << endl;
    return trajectory;
  }

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
    trajectory.push_back(p1);
  }
  return trajectory;
}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));


  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}
