/*
NOTES
All of the functions for assignment tasks have been stored in Rasterizer.cpp!
Commands for running this program:
    mkdir
    cd build
    cmake ..
    cmake --build . --config Release
    .\Release\A1.exe ..\resources\bunny.obj output.png 512 512 1
The output should be in the build folder!
*/

#include <iostream>
#include <string>
#include <vector>
#include "tiny_obj_loader.h"    // for loading meshes
#include "Image.h"
#include "Rasterizer.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace std;

double RANDOM_COLORS[7][3] = {
    {0.0000, 0.4470, 0.7410},
    {0.8500, 0.3250, 0.0980},
    {0.9290, 0.6940, 0.1250},
    {0.4940, 0.1840, 0.5560},
    {0.4660, 0.6740, 0.1880},
    {0.3010, 0.7450, 0.9330},
    {0.6350, 0.0780, 0.1840},
};

int main(int argc, char** argv) {
    if (argc < 6) {
        cout << "Usage: A1 meshfile output.png width height taskNum" << endl;
        return 0;
    }

    string meshName(argv[1]);
    string outputFile(argv[2]);
    int width = stoi(argv[3]);
    int height = stoi(argv[4]);
    int taskNum = stoi(argv[5]);

    vector<float> posBuf;
    vector<float> norBuf;
    vector<float> texBuf;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    string warnStr, errStr;

    bool rc = tinyobj::LoadObj(&attrib, &shapes, &materials, &warnStr, &errStr, meshName.c_str());
    if (!rc) {
        cerr << errStr << endl;
    }
    else {
        for (size_t s = 0; s < shapes.size(); s++) {
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                size_t fv = shapes[s].mesh.num_face_vertices[f];
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                    posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
                    posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
                    posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 2]);
                    if (!attrib.normals.empty()) {
                        norBuf.push_back(attrib.normals[3 * idx.normal_index + 0]);
                        norBuf.push_back(attrib.normals[3 * idx.normal_index + 1]);
                        norBuf.push_back(attrib.normals[3 * idx.normal_index + 2]);
                    }
                    if (!attrib.texcoords.empty()) {
                        texBuf.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
                        texBuf.push_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
                    }
                }
                index_offset += fv;
            }
        }

        Image img(width, height);

        // Choosing the task number that the user inputs
        switch (taskNum) {
        case 1:
            Rasterizer::drawBB(posBuf, img, RANDOM_COLORS);
            break;
        case 2:
            Rasterizer::drawTris(posBuf, img, RANDOM_COLORS);
            break;
        case 3:
            Rasterizer::interpolateColors(posBuf, img, RANDOM_COLORS);
            break;
        case 4:
            Rasterizer::verticalColor(posBuf, img);
            break;
        case 5:
            Rasterizer::zBuffer(posBuf, img);
            break;
        case 6:
            Rasterizer::normalColoring(posBuf, norBuf, img);
            break;
        case 7:
            Rasterizer::simpleLighting(posBuf, norBuf, img);
            break;
        case 8:
            Rasterizer::rotate(posBuf, norBuf);
            Rasterizer::simpleLighting(posBuf, norBuf, img);
            break;
        default:
            cout << "Invalid task number" << endl;
            return 1;
        }

        // Don't forget to write to the output!
        img.writeToFile(outputFile);
    }

    cout << "Number of vertices: " << posBuf.size() / 3 << endl;

    return 0;
}