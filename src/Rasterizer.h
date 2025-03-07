#pragma once
#ifndef _RASTERIZER_H_
#define _RASTERIZER_H_

#include "Image.h"
#include <vector>

class Rasterizer {
public:
    static void drawBB(const std::vector<float>& posBuf, Image& img, const double colors[7][3]);
    static void drawTris(const std::vector<float>& posBuf, Image& img, const double colors[7][3]);
    static void Rasterizer::interpolateColors(const std::vector<float>& posBuf, Image& img, const double colors[7][3]);
    static void Rasterizer::verticalColor(const std::vector<float>& posBuf, Image& img);
    static void Rasterizer::zBuffer(const std::vector<float>& posBuf, Image& img);
    static void Rasterizer::normalColoring(const std::vector<float>& posBuf, const std::vector<float>& norBuf, Image& img);
    static void Rasterizer::simpleLighting(const std::vector<float>& posBuf, const std::vector<float>& norBuf, Image& img);
    static void Rasterizer::rotate(std::vector<float>& posBuf, std::vector<float>& norBuf);
};

#endif