// std
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

// open3d
#include <open3d/core/EigenConverter.h>
#include <open3d/t/geometry/Image.h>
#include <open3d/t/geometry/VoxelBlockGrid.h>
#include <open3d/t/io/ImageIO.h>
#include <open3d/utility/Timer.h>
#include <open3d/visualization/utility/DrawGeometry.h>

// 3rd
#include <Eigen/Core>
#include <Eigen/Geometry>

// global
namespace o3c = open3d::core;
using std::cerr;
using std::cout;
using std::endl;

std::map<double, std::tuple<std::string, std::string, Eigen::Isometry3d>>
loadTumRgbdPoseDataset(const std::string& datasetRootPath,
                       const std::string& associationRgbDepthPath,
                       const std::string& associationRgbPosePath) {
  namespace fs = std::filesystem;
  std::unordered_map<double, std::pair<std::string, std::string>> tRgbDepthMap;
  {
    std::ifstream fAssociation(associationRgbDepthPath);
    while (!fAssociation.eof()) {
      std::string s;
      getline(fAssociation, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;
        double t, t2;
        std::string sRGB, sD;
        ss >> t;
        ss >> sRGB;
        ss >> t2;
        ss >> sD;
        fs::path rootPath(datasetRootPath);
        tRgbDepthMap[t] = {(rootPath / sRGB).string(),
                           (rootPath / sD).string()};
      }
    }
  }
  std::map<double, std::tuple<std::string, std::string, Eigen::Isometry3d>>
      tRgbDepthPoseMap;
  {
    std::ifstream fAssociation(associationRgbPosePath);
    while (!fAssociation.eof()) {
      std::string s;
      getline(fAssociation, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;
        double t, t2;
        std::string sRGB;
        ss >> t;
        ss >> sRGB;
        ss >> t2;
        std::array<double, 7> xyzxyzw{};
        ss >> xyzxyzw[0] >> xyzxyzw[1] >> xyzxyzw[2] >> xyzxyzw[3] >>
            xyzxyzw[4] >> xyzxyzw[5] >> xyzxyzw[6];
        if (tRgbDepthMap.find(t) == tRgbDepthMap.end()) {
          continue;
        }
        Eigen::Quaterniond qwc(xyzxyzw[6], xyzxyzw[3], xyzxyzw[4], xyzxyzw[5]);
        Eigen::Vector3d twc(xyzxyzw[0], xyzxyzw[1], xyzxyzw[2]);
        Eigen::Isometry3d T(qwc);
        T.pretranslate(twc);
        tRgbDepthPoseMap[t] = {tRgbDepthMap[t].first, tRgbDepthMap[t].second,
                               T};
      }
    }
  }
  return tRgbDepthPoseMap;
}

int main() {
  auto device = o3c::Device("CPU:0");
  open3d::t::geometry::VoxelBlockGrid vbg(
      {"tsdf", "weight", "color"}, {o3c::Float32, o3c::UInt16, o3c::UInt16},
      {{1}, {1}, {3}}, 0.02, 16, 80000, device);
  auto tRgbDepthPoseMap = loadTumRgbdPoseDataset(
      "/media/drone/Seagate/dataset/TUM-RGBD/rgbd_dataset_freiburg1_room/",
      "/media/drone/Seagate/dataset/TUM-RGBD/rgbd_dataset_freiburg1_room/"
      "associate.txt",
      "/media/drone/Seagate/dataset/TUM-RGBD/rgbd_dataset_freiburg1_room/"
      "associate_rgb_gt.txt");
  cout << "dataset size: " << tRgbDepthPoseMap.size() << endl;

  float fx(517.30), fy(516.46), cx(318.64), cy(255.31), depthScale(5000),
      depthMax(5);

  int i = 0;
  open3d::utility::Timer timer;
  timer.Start();
  for (auto& it : tRgbDepthPoseMap) {
    std::string& rgbImgPath = std::get<0>(it.second);
    std::string& depthImgPath = std::get<1>(it.second);
    Eigen::Isometry3d& pose = std::get<2>(it.second);
    open3d::t::geometry::Image imgRgb, imgD;
    open3d::t::io::ReadImage(rgbImgPath, imgRgb);
    open3d::t::io::ReadImage(depthImgPath, imgD);
    imgRgb = imgRgb.To(device);
    imgD = imgD.To(device);
    o3c::Tensor intrinsic =
        o3c::Tensor::Init<double>({{fx, 0, cx}, {0, fy, cy}, {0, 0, 1}});
    o3c::Tensor extrinsic =
        o3c::eigen_converter::EigenMatrixToTensor(pose.inverse().matrix());
    o3c::Tensor frustumBlockCoords = vbg.GetUniqueBlockCoordinates(
        imgD, intrinsic, extrinsic, depthScale, depthMax);
    vbg.Integrate(frustumBlockCoords, imgD, imgRgb, intrinsic, extrinsic,
                  depthScale, depthMax);
    cout << i++ << endl;
  }
  timer.Stop();
  cout << timer.GetDurationInMillisecond() / 1000 << endl;

  std::vector<std::shared_ptr<const open3d::geometry::Geometry>> v;

  // auto mesh = std::make_shared<open3d::geometry::TriangleMesh>(
  //     vbg.ExtractTriangleMesh().ToLegacy());
  // v.push_back(mesh);

  auto pc = std::make_shared<open3d::geometry::PointCloud>(
      vbg.ExtractPointCloud().ToLegacy());
  v.push_back(pc);

  open3d::visualization::DrawGeometries(v);
  return 0;
}
