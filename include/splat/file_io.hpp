// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <splat/geometry.hpp>

#include <iostream>
#include <string>
#include <happly.h>


namespace splat {

class Ply {
public:
    struct Property {
        std::vector<float> values;
    };

    Property x, y, z;
    Property f_dc[3];
    Property opacity;
    Property scale[3];
    Property rot[4];
};

/// Very basic (and limited) funciton to load a pointcloud xyz text file:
Points loadXyz(std::istream&& s);
Points loadPlyFile(const std::string& filename, Ply& ply);
Points loadPoints(const std::string& filename, Ply& ply);


void fillProperty(happly::PLYData& plyData, const std::string& elementName, const std::string& propertyName, Ply::Property& property);
void fillPlyProperties(happly::PLYData& plyData, Ply& ply);

} // end of namespace splat
