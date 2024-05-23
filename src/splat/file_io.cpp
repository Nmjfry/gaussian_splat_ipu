// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/file_io.hpp>

#include <sstream>
#include <algorithm>
#include <random>

namespace splat {

Points loadXyz(std::istream&& s) {
  Points pts;
  pts.reserve(1000000);

  glm::vec3 ones(1.f, 1.f, 1.f);

  for (std::string line; std::getline(s, line); ) {
    std::stringstream ss(line);
    glm::vec3 p;
    ss >> p.x >> p.y >> p.z;
    pts.push_back({p, ones});
  }

  auto rng = std::default_random_engine {};
  std::shuffle(std::begin(pts), std::end(pts), rng);

  return pts;
}

Points loadPlyFile(const std::string& filename, Ply& ply) {
    happly::PLYData plyData(filename);
    fillPlyProperties(plyData, ply);
    
    Points points;
    for (size_t i = 0; i < ply.x.values.size(); ++i) {
        Point3f point;
        point.p = glm::vec3(ply.x.values[i], ply.z.values[i], ply.y.values[i]);
        // Add other properties if needed
        points.push_back(point);
    }
    return points;
}

Points loadPoints(const std::string& filename, Ply& ply) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "xyz") {
        return loadXyz(std::ifstream(filename));
    } else if (ext == "ply") {
        return loadPlyFile(filename, ply);
    } else {
        throw std::runtime_error("Unsupported file extension");
    }
}

void fillProperty(happly::PLYData& plyData, const std::string& elementName, const std::string& propertyName, Ply::Property& property) {
    auto values = plyData.getElement(elementName).getProperty<float>(propertyName);
    property.values.insert(property.values.end(), values.begin(), values.end());
}

void fillPlyProperties(happly::PLYData& plyData, Ply& ply) {
    fillProperty(plyData, "vertex", "x", ply.x);
    fillProperty(plyData, "vertex", "y", ply.y);
    fillProperty(plyData, "vertex", "z", ply.z);
    fillProperty(plyData, "vertex", "f_dc_0", ply.f_dc[0]);
    fillProperty(plyData, "vertex", "f_dc_1", ply.f_dc[1]);
    fillProperty(plyData, "vertex", "f_dc_2", ply.f_dc[2]);
    fillProperty(plyData, "vertex", "opacity", ply.opacity);
    fillProperty(plyData, "vertex", "scale_0", ply.scale[0]);
    fillProperty(plyData, "vertex", "scale_1", ply.scale[1]);
    fillProperty(plyData, "vertex", "scale_2", ply.scale[2]);
    fillProperty(plyData, "vertex", "rot_0", ply.rot[0]);
    fillProperty(plyData, "vertex", "rot_1", ply.rot[1]);
    fillProperty(plyData, "vertex", "rot_2", ply.rot[2]);
    fillProperty(plyData, "vertex", "rot_3", ply.rot[3]);
}

} // end of namespace splat
