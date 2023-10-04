# sploot

Attempting to implement 3d Gaussian Splats from scratch for fun.

A set of splats
Each splat has position, covariance, color, opacity
projection of splat into camera plane

GPU-friendly rasterization process where each thread block is assigned to render an image tile.

Given a scaling matrix 𝑆 and rotation matrix 𝑅, we can find the corresponding Σ:
Σ = 𝑅𝑆𝑆𝑇
To allow independent optimization of both factors, we store them separately: a 3D vector 𝑠 for scaling and a quaternion 𝑞 to represent rotation. These can be trivially converted to their respective matrices and combined, making sure to normalize 𝑞 to obtain a valid unit quaternion.

split the screen into tiles (16x16)
cull gaussians using view frustum

sort gaussians by view-space depth (each tile on one thread) using GPU Radix sort

alpha blend gaussians to get final pixel values for tile

![Alt text](algo1.png)

![Alt text](algo2.png)