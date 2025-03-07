/*
This is a helper file for all of the tasks in main().
The functions are listed in chronological order.
*/

#include "Rasterizer.h"
#include <algorithm>
#include <cmath>

/*
****************
HELPER FUNCTIONS
****************
*/

// Function for mapping a point (x,y) to image pixel coordinates
void mapToImageCoords(float x, float y,
                      float min_x, float min_y,
                      float max_x, float max_y,
                      int width, int height,
                      int &out_x, int &out_y)
{
    // Compute scaling factors
    float scale_x = width / (max_x - min_x);
    float scale_y = height / (max_y - min_y);
    float scale = std::min(scale_x, scale_y);

    // Convert world coordinates to image coordinates
    out_x = static_cast<int>((x - min_x) * scale);
    out_y = static_cast<int>((y - min_y) * scale);

    // Constrict the coordinates to the valid range [0, width-1] and [0, height-1]
    // Prevents out of bound errors
    out_x = std::max(0, std::min(out_x, width - 1));
    out_y = std::max(0, std::min(out_y, height - 1));
}

// Function for computing (u, v, w) of a point (x, y) inside a triangle
void computeBarycentric(float x, float y,
                        float x0, float y0,
                        float x1, float y1,
                        float x2, float y2,
                        float &u, float &v, float &w)
{
    // If u, v, and w are between 0 and 1, the point is inside the triangle
    // If any value is negative or greater than 1, the point is outside the triangle
    float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    u = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom;
    v = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom;
    w = 1 - u - v;
}

// Function for rotation matrix for y-axis
void applyRotation(float &x, float &y, float &z)
{
    // Initialization for defining rotation angle
    const float theta = 0.7853981633974483f; // pi/4 in radians
    const float cosTheta = std::cos(theta);
    const float sinTheta = std::sin(theta);

    float x0 = x, z0 = z;
    x = cosTheta * x0 + sinTheta * z0;
    // y remains unchanged because rotation is around the y-axis
    z = -sinTheta * x0 + cosTheta * z0;
}

// Helper function to compute the bounding box for a set of vertices
// Used in every function that only requires x and y values
void computeBoundingBox(const std::vector<float> &posBuf,
                        float &min_x, float &max_x,
                        float &min_y, float &max_y)
{
    // Initialize bounding box limits
    min_x = std::numeric_limits<float>::max();
    max_x = std::numeric_limits<float>::lowest();
    min_y = std::numeric_limits<float>::max();
    max_y = std::numeric_limits<float>::lowest();

    // Iterate through posBuf
    for (size_t i = 0; i < posBuf.size(); i += 3)
    {
        // Extract x and y values
        float x = posBuf[i];
        float y = posBuf[i + 1];

        // Keep track of the smallest and largest x and y values
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }
}

/*
*******************
Functions For Tasks
*******************
*/

/*
TASK 1: DRAWING BOUNDING BOXES
Main concept: Transforming 3D coordinates into 2D image coordinates (step 1 of rasterization)
*/
void Rasterizer::drawBB(const std::vector<float> &posBuf, Image &img, const double colors[7][3])
{
    // Compute the bounding box for the entire mesh
    // This helps with scaling!
    float min_x, max_x, min_y, max_y;
    computeBoundingBox(posBuf, min_x, max_x, min_y, max_y);

    // Draw the bounding box for each triangle
    // Helps in later processing
    for (size_t i = 0; i < posBuf.size(); i += 9)
    {
        // Extract the 3 vertices of the triangle using posBuf (formated as (x, y, z))
        float x0 = posBuf[i]; float y0 = posBuf[i + 1];
        float x1 = posBuf[i + 3]; float y1 = posBuf[i + 4]; // Why not 2? Because we're skipping z
        float x2 = posBuf[i + 6]; float y2 = posBuf[i + 7]; // Again, skipping z

        // Compute the bounding box for this triangle by finding the smallest and largest x and y values
        float triMin_x = std::min({x0, x1, x2});
        float triMax_x = std::max({x0, x1, x2});
        float triMin_y = std::min({y0, y1, y2});
        float triMax_y = std::max({y0, y1, y2});

        // Map the bounding box coordinates to image space
        int imgMin_x, imgMin_y, imgMax_x, imgMax_y;
        mapToImageCoords(triMin_x, triMin_y, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), imgMin_x, imgMin_y);
        mapToImageCoords(triMax_x, triMax_y, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), imgMax_x, imgMax_y);

        // Pick a color for the triangle from RANDOM_COLORS
        int colorIndex = (i / 9) % 7;
        unsigned char r = static_cast<unsigned char>(colors[colorIndex][0] * 255);
        unsigned char g = static_cast<unsigned char>(colors[colorIndex][1] * 255);
        unsigned char b = static_cast<unsigned char>(colors[colorIndex][2] * 255);

        // Loop through all pixels inside the bounding box and fill them with the chosen color
        for (int x = imgMin_x; x <= imgMax_x; ++x)
        {
            for (int y = imgMin_y; y <= imgMax_y; ++y)
            {
                img.setPixel(x, y, r, g, b);
            }
        }
    }
}

/*
TASK 2: DRAWING TRIANGLES
Main Concept: Add the barycentric test to write out the triangles
*/
void Rasterizer::drawTris(const std::vector<float> &posBuf, Image &img, const double colors[7][3])
{
    // Compute overall bounding box for the mesh as done before
    float min_x, max_x, min_y, max_y;
    computeBoundingBox(posBuf, min_x, max_x, min_y, max_y);

    // Loop through each triangle, skipping z because it's not needed for 2D
    for (size_t i = 0; i < posBuf.size(); i += 9) // (x, y, z) --> 3 * 3 = 9 floats
    {
        // Extract triangle vertices as done before
        float x0 = posBuf[i], y0 = posBuf[i + 1];
        float x1 = posBuf[i + 3], y1 = posBuf[i + 4];
        float x2 = posBuf[i + 6], y2 = posBuf[i + 7];

        // Convert to image coordinates (ensure float conversion)
        int img_x0, img_y0, img_x1, img_y1, img_x2, img_y2;
        mapToImageCoords(x0, y0, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x0, img_y0);
        mapToImageCoords(x1, y1, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x1, img_y1);
        mapToImageCoords(x2, y2, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x2, img_y2);

        // Compute the triangle's bounding box by determining the smallest and largest pixel coordinates
        int triMin_x = std::min({img_x0, img_x1, img_x2});
        int triMax_x = std::max({img_x0, img_x1, img_x2});
        int triMin_y = std::min({img_y0, img_y1, img_y2});
        int triMax_y = std::max({img_y0, img_y1, img_y2});

        // Constrict bounding box to stay inside the imahe
        triMin_x = std::max(0, triMin_x);
        triMax_x = std::min(img.getWidth() - 1, triMax_x);
        triMin_y = std::max(0, triMin_y);
        triMax_y = std::min(img.getHeight() - 1, triMax_y);

        // Choose color for this triangle using RANDOM_COLORS
        int colorIndex = (i / 9) % 7;
        unsigned char r = static_cast<unsigned char>(colors[colorIndex][0] * 255);
        unsigned char g = static_cast<unsigned char>(colors[colorIndex][1] * 255);
        unsigned char b = static_cast<unsigned char>(colors[colorIndex][2] * 255);

        // Iterate over pixels in the triangle's bounding box
        for (int x = triMin_x; x <= triMax_x; ++x)
        {
            for (int y = triMin_y; y <= triMax_y; ++y)
            {
                // Convert pixel position to float explicitly
                float fx = static_cast<float>(x);
                float fy = static_cast<float>(y);

                // Use barycentric coordinates to check if the pixel (fx, fy) is inside the triangle
                float u, v, w;
                computeBarycentric(fx, fy, static_cast<float>(img_x0), static_cast<float>(img_y0),
                                   static_cast<float>(img_x1), static_cast<float>(img_y1),
                                   static_cast<float>(img_x2), static_cast<float>(img_y2), u, v, w);

                // Check if the pixel is inside the triangle
                /*
                USED CHATGPT HERE:
                I used >= 0 at first and was getting some weird miscolorations, so I asked ChatGPT what was wrong with it.
                The AI suggested I use -1e-5 instead of 0, and it worked. It was because there were some floating-point
                precision errors.
                */
                if (u >= -1e-5 && v >= -1e-5 && w >= -1e-5) // If all coordinates are non-negative
                {
                    img.setPixel(x, y, r, g, b); // Valid
                }
            }
        }
    }
}

/*
TASK 3: INTERPOLATING PER-VERTEX COLORS
Main Concept: Use random per-vertex colors to blend the colors smoothly across the triangle's surface
*/
void Rasterizer::interpolateColors(const std::vector<float> &posBuf, Image &img, const double colors[7][3])
{
    // Compute the global bounding box for the entire mesh
    float min_x, max_x, min_y, max_y;
    computeBoundingBox(posBuf, min_x, max_x, min_y, max_y);

    // Looop through each triangle
    for (size_t i = 0; i < posBuf.size(); i += 9)
    {
        // Extract triangle vertices again
        float x0 = posBuf[i], y0 = posBuf[i + 1];
        float x1 = posBuf[i + 3], y1 = posBuf[i + 4];
        float x2 = posBuf[i + 6], y2 = posBuf[i + 7];

        // Assign per-vertex colors using RANDOM_COLORS
        const double *color0 = colors[(i / 3) % 7];
        const double *color1 = colors[(i / 3 + 1) % 7];
        const double *color2 = colors[(i / 3 + 2) % 7];

        // Convert triangle coordinates to image space by mapping
        int img_x0, img_y0, img_x1, img_y1, img_x2, img_y2;
        mapToImageCoords(x0, y0, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x0, img_y0);
        mapToImageCoords(x1, y1, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x1, img_y1);
        mapToImageCoords(x2, y2, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x2, img_y2);

        // Compute the triangle's bounding box in image space
        // Only check pixels inside this box instead of the entire image
        int triMin_x = std::min({img_x0, img_x1, img_x2});
        int triMax_x = std::max({img_x0, img_x1, img_x2});
        int triMin_y = std::min({img_y0, img_y1, img_y2});
        int triMax_y = std::max({img_y0, img_y1, img_y2});

        // Constrict bounding box to image bounds
        triMin_x = std::max(0, triMin_x);
        triMax_x = std::min(img.getWidth() - 1, triMax_x);
        triMin_y = std::max(0, triMin_y);
        triMax_y = std::min(img.getHeight() - 1, triMax_y);

        // Iterate over the bounding box pixels
        for (int x = triMin_x; x <= triMax_x; ++x)
        {
            for (int y = triMin_y; y <= triMax_y; ++y)
            {
                // Convert pixel position to float for interpolation
                float fx = static_cast<float>(x);
                float fy = static_cast<float>(y);

                // Compute barycentric coordinates
                float u, v, w;
                computeBarycentric(fx, fy, static_cast<float>(img_x0), static_cast<float>(img_y0),
                                   static_cast<float>(img_x1), static_cast<float>(img_y1),
                                   static_cast<float>(img_x2), static_cast<float>(img_y2), u, v, w);

                // Check if the pixel is inside the triangle
                if (u >= -1e-5 && v >= -1e-5 && w >= -1e-5)
                {
                    // Interpolate the color
                    unsigned char r = static_cast<unsigned char>(std::max(0.0, std::min(255.0,
                                                                                        u * color0[0] * 255 + v * color1[0] * 255 + w * color2[0] * 255)));

                    unsigned char g = static_cast<unsigned char>(std::max(0.0, std::min(255.0,
                                                                                        u * color0[1] * 255 + v * color1[1] * 255 + w * color2[1] * 255)));

                    unsigned char b = static_cast<unsigned char>(std::max(0.0, std::min(255.0,
                                                                                        u * color0[2] * 255 + v * color1[2] * 255 + w * color2[2] * 255)));

                    // Set the pixel color
                    img.setPixel(x, y, r, g, b);
                }
            }
        }
    }
}
/*
REFERENCE FOR SELF
If we had a triangle with these vertex colors:
- Vertex 1 (Red)
- Vertex 2 (Green)
- Vertex 3 (Blue)
The rasterized triangle would look like this:
   R
  RGB
 RGGB
RGBGBG
 GGGG
  B
*/

/*
TASK 4: VERTICAL COLOR
Main Concept: Implement vertical color interpolation, where the y-value determines the color
*/
void Rasterizer::verticalColor(const std::vector<float> &posBuf, Image &img)
{
    // Compute the global bounding box for the entire mesh
    // This will be NEEDED for color mapping
    float min_x, max_x, min_y, max_y;
    computeBoundingBox(posBuf, min_x, max_x, min_y, max_y);

    // Loop through each triangle
    for (size_t i = 0; i < posBuf.size(); i += 9)
    {
        // Extract triangle vertices
        float x0 = posBuf[i], y0 = posBuf[i + 1];
        float x1 = posBuf[i + 3], y1 = posBuf[i + 4];
        float x2 = posBuf[i + 6], y2 = posBuf[i + 7];

        // Convert to image coordinates
        int img_x0, img_y0, img_x1, img_y1, img_x2, img_y2;
        mapToImageCoords(x0, y0, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x0, img_y0);
        mapToImageCoords(x1, y1, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x1, img_y1);
        mapToImageCoords(x2, y2, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x2, img_y2);

        // Compute the triangle's bounding box
        int triMin_x = std::min({img_x0, img_x1, img_x2});
        int triMax_x = std::max({img_x0, img_x1, img_x2});
        int triMin_y = std::min({img_y0, img_y1, img_y2});
        int triMax_y = std::max({img_y0, img_y1, img_y2});

        // Constrict bounding box to image bounds
        triMin_x = std::max(0, triMin_x);
        triMax_x = std::min(img.getWidth() - 1, triMax_x);
        triMin_y = std::max(0, triMin_y);
        triMax_y = std::min(img.getHeight() - 1, triMax_y);

        // Iterate over the bounding box pixels
        for (int x = triMin_x; x <= triMax_x; ++x)
        {
            for (int y = triMin_y; y <= triMax_y; ++y)
            {
                // Convert pixel position to float for interpolation
                float fx = static_cast<float>(x);
                float fy = static_cast<float>(y);

                // Compute barycentric coordinates
                float u, v, w;
                computeBarycentric(fx, fy, static_cast<float>(img_x0), static_cast<float>(img_y0),
                                   static_cast<float>(img_x1), static_cast<float>(img_y1),
                                   static_cast<float>(img_x2), static_cast<float>(img_y2), u, v, w);

                // Check if the pixel is inside the triangle
                if (u >= -1e-5 && v >= -1e-5 && w >= -1e-5)
                {
                    // Interpolate the y-value using barycentric coordinates
                    float world_y = u * y0 + v * y1 + w * y2;

                    // Map world_y to the range [0, 1] (0 = blue, 1 = red)
                    float t = (world_y - min_y) / (max_y - min_y);
                    t = std::max(0.0f, std::min(1.0f, t)); // Constrict to [0,1]

                    // Compute interpolated color (linear blend from blue to red)
                    /*
                    USED CHATGPT HERE:
                    I honestly did not know how to interpolate between the two colors based on the range,
                    so I asked AI. The AI explained that (t * 255) would control the red component, and
                    (1 - t) controls the blue component.
                    */
                    unsigned char r = static_cast<unsigned char>(t * 255);
                    unsigned char g = 0;
                    unsigned char b = static_cast<unsigned char>((1 - t) * 255);

                    // Set the pixel color
                    img.setPixel(x, y, r, g, b);
                }
            }
        }
    }
}

/*
TASK 5: Z-BUFFERING
Main Concept: Create a data structure to support z-buffer tests and ensure that closer objects overwrite farther ones
*/
void Rasterizer::zBuffer(const std::vector<float> &posBuf, Image &img)
{
    // Create and initialize the Z-buffer with negative infinity
    std::vector<float> zBuffer(img.getWidth() * img.getHeight(), -std::numeric_limits<float>::max());

    // Compute the global bounding box for the entire mesh
    // Unlike previous functions, this is for x, y, and z coordinates
    // Thus, we're not calling computeBoundingBox() here
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();

    // Loop through each triangle, using the three coordinates this time
    for (size_t i = 0; i < posBuf.size(); i += 3)
    {
        float x = posBuf[i];
        float y = posBuf[i + 1];
        float z = posBuf[i + 2];
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
    }

    // Process each triangle as usual
    for (size_t i = 0; i < posBuf.size(); i += 9)
    {
        // Extract triangle vertices, this time accounting for z
        float x0 = posBuf[i], y0 = posBuf[i + 1], z0 = posBuf[i + 2];
        float x1 = posBuf[i + 3], y1 = posBuf[i + 4], z1 = posBuf[i + 5];
        float x2 = posBuf[i + 6], y2 = posBuf[i + 7], z2 = posBuf[i + 8];

        // Convert to image coordinates using x and y bounding box
        int img_x0, img_y0, img_x1, img_y1, img_x2, img_y2;
        mapToImageCoords(x0, y0, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x0, img_y0);
        mapToImageCoords(x1, y1, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x1, img_y1);
        mapToImageCoords(x2, y2, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x2, img_y2);

        // Compute the triangle's bounding box
        int triMin_x = std::max(0, std::min({img_x0, img_x1, img_x2}));
        int triMax_x = std::min(img.getWidth() - 1, std::max({img_x0, img_x1, img_x2}));
        int triMin_y = std::max(0, std::min({img_y0, img_y1, img_y2}));
        int triMax_y = std::min(img.getHeight() - 1, std::max({img_y0, img_y1, img_y2}));

        // Iterate over pixels in the bounding box
        for (int x = triMin_x; x <= triMax_x; ++x)
        {
            for (int y = triMin_y; y <= triMax_y; ++y)
            {
                // Convert pixel position to float
                float fx = static_cast<float>(x);
                float fy = static_cast<float>(y);

                // Compute barycentric coordinates
                float u, v, w;
                computeBarycentric(fx, fy, static_cast<float>(img_x0), static_cast<float>(img_y0),
                                   static_cast<float>(img_x1), static_cast<float>(img_y1),
                                   static_cast<float>(img_x2), static_cast<float>(img_y2), u, v, w);

                // Check if the pixel is inside the triangle
                if (u >= -1e-5 && v >= -1e-5 && w >= -1e-5)
                {
                    // Interpolate the depth (z-value)
                    float interpolated_z = u * z0 + v * z1 + w * z2;

                    // Compute 1D index in Z-buffer
                    int pixelIndex = y * img.getWidth() + x;

                    // Perform a depth test
                    // If the interpolated z-value is closer than the current value, update the pixel
                    if (interpolated_z > zBuffer[pixelIndex]) //  Slight debug with ChatGPT: changed < to >
                    {
                        zBuffer[pixelIndex] = interpolated_z; // Update Z-buffer

                        // Map interpolated_z to grayscale color (0 = farthest, 255 = closest)
                        float normalized_z = (interpolated_z - min_z) / (max_z - min_z);
                        normalized_z = std::max(0.0f, std::min(1.0f, normalized_z)); // Constrict to [0,1]
                        unsigned char depthColor = static_cast<unsigned char>(normalized_z * 255);

                        // Set pixel color (Red = depth, Green & Blue = 0)
                        img.setPixel(x, y, depthColor, 0, 0);
                    }
                }
            }
        }
    }
}

/*
TASK 6: NORMAL COLORING
Main concept: Store and interpolate the normal vectors in each vertex
*/
void Rasterizer::normalColoring(const std::vector<float> &posBuf, const std::vector<float> &norBuf, Image &img)
{
    // Initialize to the lowest possible depth
    std::vector<std::vector<float>> zBuffer(img.getHeight(), std::vector<float>(img.getWidth(), std::numeric_limits<float>::lowest()));

    // Compute the global bounding box for the entire mesh
    float min_x, max_x, min_y, max_y;
    computeBoundingBox(posBuf, min_x, max_x, min_y, max_y);

    // Loop through each triangle
    for (size_t i = 0; i < posBuf.size(); i += 9)
    {
        // Extract triangle vertices
        float x0 = posBuf[i], y0 = posBuf[i + 1], z0 = posBuf[i + 2];
        float x1 = posBuf[i + 3], y1 = posBuf[i + 4], z1 = posBuf[i + 5];
        float x2 = posBuf[i + 6], y2 = posBuf[i + 7], z2 = posBuf[i + 8];

        // Extract vertex normals
        float nx0 = norBuf[i], ny0 = norBuf[i + 1], nz0 = norBuf[i + 2];
        float nx1 = norBuf[i + 3], ny1 = norBuf[i + 4], nz1 = norBuf[i + 5];
        float nx2 = norBuf[i + 6], ny2 = norBuf[i + 7], nz2 = norBuf[i + 8];

        // Convert to image coordinates
        int img_x0, img_y0, img_x1, img_y1, img_x2, img_y2;
        mapToImageCoords(x0, y0, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x0, img_y0);
        mapToImageCoords(x1, y1, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x1, img_y1);
        mapToImageCoords(x2, y2, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x2, img_y2);

        // Compute bounding box of the triangle
        int min_x = std::max(0, std::min({img_x0, img_x1, img_x2}));
        int max_x = std::min(img.getWidth() - 1, std::max({img_x0, img_x1, img_x2}));
        int min_y = std::max(0, std::min({img_y0, img_y1, img_y2}));
        int max_y = std::min(img.getHeight() - 1, std::max({img_y0, img_y1, img_y2}));

        // Rasterize the triangle
        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                float u, v, w;
                computeBarycentric(static_cast<float>(x), static_cast<float>(y),
                                   static_cast<float>(img_x0), static_cast<float>(img_y0),
                                   static_cast<float>(img_x1), static_cast<float>(img_y1),
                                   static_cast<float>(img_x2), static_cast<float>(img_y2), u, v, w);

                // If inside the triangle
                if (u >= -1e-5f && v >= -1e-5f && w >= -1e-5f)
                {
                    // Interpolate z-value
                    float z = u * z0 + v * z1 + w * z2;
                    int index = y * img.getWidth() + x;
                    if (z > zBuffer[y][x])
                    {
                        zBuffer[y][x] = z;

                        // Interpolate normal
                        float nx = u * nx0 + v * nx1 + w * nx2;
                        float ny = u * ny0 + v * ny1 + w * ny2;
                        float nz = u * nz0 + v * nz1 + w * nz2;

                        // Normalize the interpolated normal
                        float length = std::sqrt(nx * nx + ny * ny + nz * nz);
                        if (length > 1e-5f)
                        {
                            nx /= length;
                            ny /= length;
                            nz /= length;
                        }

                        // Map normal to color
                        unsigned char r = static_cast<unsigned char>(255 * (0.5f * nx + 0.5f));
                        unsigned char g = static_cast<unsigned char>(255 * (0.5f * ny + 0.5f));
                        unsigned char b = static_cast<unsigned char>(255 * (0.5f * nz + 0.5f));

                        img.setPixel(x, y, r, g, b);
                    }
                }
            }
        }
    }
}

/*
TASK 7: SIMPLE LIGHTING
Main concept: Calculate the lighting at each pixel based on the normal vector at that pixel
and a directional light source
*/
void Rasterizer::simpleLighting(const std::vector<float> &posBuf, const std::vector<float> &norBuf, Image &img)
{
    // Again, initialize to the lowest possible depth
    std::vector<std::vector<float>> zBuffer(img.getHeight(), std::vector<float>(img.getWidth(), std::numeric_limits<float>::lowest()));

    // Compute global bounding box (as before)
    float min_x, max_x, min_y, max_y;
    computeBoundingBox(posBuf, min_x, max_x, min_y, max_y);

    // Define light vector according to the formula given in documentation
    const float sqrt3 = std::sqrt(3.0f);
    const float lx = 1.0f / sqrt3;
    const float ly = 1.0f / sqrt3;
    const float lz = 1.0f / sqrt3;

    // Process each triangle
    for (size_t i = 0; i < posBuf.size(); i += 9)
    {
        // Extract triangle vertices and normals as before
        float x0 = posBuf[i], y0 = posBuf[i + 1], z0 = posBuf[i + 2];
        float x1 = posBuf[i + 3], y1 = posBuf[i + 4], z1 = posBuf[i + 5];
        float x2 = posBuf[i + 6], y2 = posBuf[i + 7], z2 = posBuf[i + 8];

        // Extract vertex normals
        float nx0 = norBuf[i], ny0 = norBuf[i + 1], nz0 = norBuf[i + 2];
        float nx1 = norBuf[i + 3], ny1 = norBuf[i + 4], nz1 = norBuf[i + 5];
        float nx2 = norBuf[i + 6], ny2 = norBuf[i + 7], nz2 = norBuf[i + 8];

        // Convert to image coordinates
        int img_x0, img_y0, img_x1, img_y1, img_x2, img_y2;
        mapToImageCoords(x0, y0, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x0, img_y0);
        mapToImageCoords(x1, y1, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x1, img_y1);
        mapToImageCoords(x2, y2, min_x, min_y, max_x, max_y, img.getWidth(), img.getHeight(), img_x2, img_y2);

        // Compute bounding box of the triangle
        int min_x = std::max(0, std::min({img_x0, img_x1, img_x2}));
        int max_x = std::min(img.getWidth() - 1, std::max({img_x0, img_x1, img_x2}));
        int min_y = std::max(0, std::min({img_y0, img_y1, img_y2}));
        int max_y = std::min(img.getHeight() - 1, std::max({img_y0, img_y1, img_y2}));

        // Rasterize the triangle
        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                float u, v, w;
                computeBarycentric(static_cast<float>(x), static_cast<float>(y),
                                   static_cast<float>(img_x0), static_cast<float>(img_y0),
                                   static_cast<float>(img_x1), static_cast<float>(img_y1),
                                   static_cast<float>(img_x2), static_cast<float>(img_y2), u, v, w);

                if (u >= -1e-5f && v >= -1e-5f && w >= -1e-5f)
                {
                    // Interpolate z-value
                    float z = u * z0 + v * z1 + w * z2;
                    if (z > zBuffer[y][x])
                    {
                        zBuffer[y][x] = z;

                        // Interpolate normal
                        float nx = u * nx0 + v * nx1 + w * nx2;
                        float ny = u * ny0 + v * ny1 + w * ny2;
                        float nz = u * nz0 + v * nz1 + w * nz2;

                        // Normalize the interpolated normal
                        float length = std::sqrt(nx * nx + ny * ny + nz * nz);
                        if (length > 1e-5f)
                        {
                            nx /= length;
                            ny /= length;
                            nz /= length;
                        }

                        // Compute lighting
                        float dotProduct = nx * lx + ny * ly + nz * lz;
                        float c = std::max(dotProduct, 0.0f);

                        // Apply lighting to color
                        unsigned char color = static_cast<unsigned char>(255 * c);

                        // Set the pixel color
                        img.setPixel(x, y, color, color, color);
                    }
                }
            }
        }
    }
}

/*
TASK 8: ROTATION
Main concept: Rotate both the positions and normals of the vertices in the mesh
*/
void Rasterizer::rotate(std::vector<float> &posBuf, std::vector<float> &norBuf)
{
    // Loop through each vertex
    for (size_t i = 0; i < posBuf.size(); i += 3)
    {
        // Rotate vertex position
        applyRotation(posBuf[i], posBuf[i + 1], posBuf[i + 2]);

        // Rotate normal to maintain correct lighting
        applyRotation(norBuf[i], norBuf[i + 1], norBuf[i + 2]);
    }
}