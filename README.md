# 3D Gaussian Splatting on Graphcore IPUs

Experimental implementation of [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting) B. Kerbl and G. Kopanas, T. Leimk{\"u}hler and G. Drettakis, ACM Transactions on Graphics, July 2023.

Which bits are a good fit for IPU?:
- Should be able to perform efficient Gaussian transformations and splats using matrix and convolution unit.
- Tile based renderer is IPU friendly (each tile of framebuffer can stay pinned in SRAM of each IPU tile).
- Can hold millions of gaussians in SRAM.
  - i.e. scene and framebuffer can all stay in SRAM.

Which bits will be difficult?:
- After transformation and sorting Gaussians need to be dynamically moved to the destination tile(s) for splatting.
- Load imbalances: depending on scene and viewpoint the number of Gaussians splatted per tile could vary significantly.
 - I.e. any one gaussian might need to be splatted on 0 or all tiles!
