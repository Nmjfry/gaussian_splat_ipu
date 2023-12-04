// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdlib>

#include <ipu/ipu_utils.hpp>
#include <glm/mat4x4.hpp>
#include <opencv2/imgproc.hpp>

namespace splat {

// Fwd decls:
class Point3f;
typedef std::vector<Point3f> Points;
typedef glm::vec4 Pixel;
typedef std::vector<Pixel> Pixels;


class IpuSplatter : public ipu_utils::BuilderInterface {
public:
  IpuSplatter(const Points& pts, bool noAMP);
  IpuSplatter(const Pixels& pxs, bool noAMP);
  virtual ~IpuSplatter() {}

  void updateModelViewProjection(const glm::mat4& mvp);
  void updateFrameBuffer(cv::Mat& frame);
  void getProjectedPoints(std::vector<glm::vec4>& pts) const;
  void getTransformedFrame(cv::Mat& frame) const;

private:
  void build(poplar::Graph& graph, const poplar::Target& target) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  ipu_utils::StreamableTensor modelViewProjection;
  ipu_utils::StreamableTensor inputVertices;
  ipu_utils::StreamableTensor outputVertices;
  std::vector<float> transformMatrix;
  std::vector<float> hostVertices;
  std::atomic<bool> initialised;
  const bool disableAMPVertices;
};

} // end of namespace splat
