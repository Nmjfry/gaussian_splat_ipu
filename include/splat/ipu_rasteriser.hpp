// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdlib>

#include <ipu/ipu_utils.hpp>
#include <glm/mat4x4.hpp>
#include <tileMapping/tileConfig.hpp>
#include <opencv2/imgproc.hpp>

namespace splat {

// Fwd decls:
class Point3f;
typedef std::vector<Point3f> Points;

class IpuSplatter : public ipu_utils::BuilderInterface {
public:
  IpuSplatter(const Points& pts, TiledFramebuffer& fb, bool noAMP);
  virtual ~IpuSplatter() {}

  void updateModelViewProjection(const glm::mat4& mvp);
  void getProjectedPoints(std::vector<glm::vec4>& pts) const;
  void getFrameBuffer(cv::Mat &frame) const;

private:
  void build(poplar::Graph& graph, const poplar::Target& target) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  ipu_utils::StreamableTensor modelViewProjection;
  ipu_utils::StreamableTensor inputVertices;
  ipu_utils::StreamableTensor outputFramebuffer;
  std::vector<float> transformMatrix;
  std::vector<float> hostVertices;
  TiledFramebuffer fbMapping;
  std::vector<float> frameBuffer;
  std::atomic<bool> initialised;
  const bool disableAMPVertices; 
};

} // end of namespace splat
