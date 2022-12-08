// std
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

// open3d
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/EigenConverter.h>
#include <open3d/core/Tensor.h>
#include <open3d/core/TensorKey.h>
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
  auto device = o3c::Device("CUDA:0");
  float voxelSize = 0.02;
  float trunc = voxelSize * 4;

  open3d::t::geometry::VoxelBlockGrid vbg(
      {"tsdf", "weight", "color"}, {o3c::Float32, o3c::Float32, o3c::Float32},
      {{1}, {1}, {3}}, voxelSize, 16, 80000, device);
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
    // vbg.Integrate(frustumBlockCoords, imgD, imgRgb, intrinsic, extrinsic,
    //               depthScale, depthMax);
    auto vbgHashMap = vbg.GetHashMap();
    vbgHashMap.Activate(frustumBlockCoords);  // Don't use return value.
    auto [bufIndices, masks] = vbgHashMap.Find(frustumBlockCoords);
    o3c::cuda::Synchronize();
    auto [voxelCoords, voxelIndices] =
        vbg.GetVoxelCoordinatesAndFlattenedIndices(bufIndices);
    o3c::cuda::Synchronize();
    auto intrinsicDev = intrinsic.To(device, o3c::Float32);
    auto extrinsicDev = extrinsic.To(device, o3c::Float32);
    auto rotation = extrinsicDev.GetItem(
        {o3c::TensorKey::Slice(0, 3, 1), o3c::TensorKey::Slice(0, 3, 1)});
    auto translation =
        extrinsicDev.GetItem({o3c::TensorKey::Slice(0, 3, 1),
                              o3c::TensorKey::Slice(3, o3c::None, 1)});
    auto xyz = rotation.Matmul(voxelCoords.T()) + translation;
    auto uvd = intrinsicDev.Matmul(xyz);
    auto d = uvd[2];
    auto u = (uvd[0] / d).Round().To(o3c::Int64);
    auto v = (uvd[1] / d).Round().To(o3c::Int64);
    o3c::cuda::Synchronize();

    auto maskProj =
        o3c::TensorKey::IndexTensor(d.Gt(0)
                                        .LogicalAnd(u.Ge(0))
                                        .LogicalAnd(v.Ge(0))
                                        .LogicalAnd(u.Lt(imgD.GetCols()))
                                        .LogicalAnd(v.Lt(imgD.GetRows())));
    auto vProj = v.GetItem(maskProj);
    auto uProj = u.GetItem(maskProj);
    auto dProj = d.GetItem(maskProj);
    auto depthReading =
        imgD.AsTensor()
            .IndexGet({vProj, uProj, o3c::Tensor::Init<int64_t>({0}, device)})
            .To(o3c::Float32) /
        depthScale;
    auto sdf = depthReading - dProj;

    auto maskInlier = o3c::TensorKey::IndexTensor(
        depthReading.Gt(0) && depthReading.Lt(depthMax) && sdf.Ge(-trunc));

    sdf.SetItem(o3c::TensorKey::IndexTensor(sdf.Ge(trunc)),
                o3c::Tensor::Init<float>({trunc}, device));
    sdf /= trunc;
    o3c::cuda::Synchronize();

    auto weight = vbg.GetAttribute("weight").Reshape({-1, 1});
    auto tsdf = vbg.GetAttribute("tsdf").Reshape({-1, 1});
    auto validVoxelIndices = o3c::TensorKey::IndexTensor(
        voxelIndices.GetItem(maskProj).GetItem(maskInlier));
    auto w = weight.GetItem(validVoxelIndices);
    auto wp = w + 1;

    tsdf.SetItem(validVoxelIndices,
                 (tsdf.GetItem(validVoxelIndices) * w +
                  sdf.GetItem(maskInlier).Reshape(w.GetShape())) /
                     wp);

    auto color = vbg.GetAttribute("color").Reshape({-1, 3});
    auto colorReading =
        imgRgb.AsTensor().IndexGet({vProj, uProj}).To(o3c::Float32);

    color.SetItem(validVoxelIndices, (color.GetItem(validVoxelIndices) * w +
                                      colorReading.GetItem(maskInlier)) /
                                         wp);

    weight.SetItem(validVoxelIndices, wp);
    o3c::cuda::Synchronize();
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
