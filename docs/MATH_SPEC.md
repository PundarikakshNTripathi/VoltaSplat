# 3D Gaussian Splatting: Mathematical and Algorithmic Specification

## 1. Introduction
This document outlines the mathematical formulation and algorithmic requirements for the forward rasterization and backward gradient propagation passes of the 3D Gaussian Splatting (3DGS) engine.

## 2. Forward Pass: Geometry Projection

A 3D Gaussian is defined by its mean $\mu \in \mathbb{R}^3$ and its covariance matrix $\Sigma \in \mathbb{R}^{3 \times 3}$. When projecting to 2D screen space, we use a viewing transformation matrix $W \in \mathbb{R}^{4 \times 4}$ and the Jacobian $J$ of the perspective projection affine approximation.

### 2.1 Viewing Transformation
Let a 3D point (or the mean $\mu$) be transformed into camera space coordinates $t = (t_x, t_y, t_z)^T$:
$$ t = W \mu $$
where $W$ contains the view rotation and translation.

### 2.2 Jacobian of Perspective Projection
The perspective projection Jacobian $J$ evaluated at the camera-space point $t$ is:
$$ J = \begin{bmatrix}
\frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\
0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \\
0 & 0 & 0
\end{bmatrix} $$
where $f_x, f_y$ are the focal lengths.

### 2.3 2D Covariance
The 3D covariance $\Sigma$ is projected to 2D covariance $\Sigma'$ in screen space as:
$$ \Sigma' = J W \Sigma W^T J^T $$
We extract the top-left $2 \times 2$ submatrix of $\Sigma'$ as the 2D covariance for rasterization. A low-pass filter (adding an identity matrix scaled by a small constant, e.g., 0.3) is typically applied to $\Sigma'$ to prevent aliasing. The inverse of this $2 \times 2$ matrix is the *conic* parameters used in the rendering equation.

## 3. Forward Pass: Alpha-Blending Rendering Equation

The color $C$ of a pixel is computed via front-to-back alpha compositing (volumetric rendering approximation). Given $N$ sorted Gaussians overlapping the pixel, the color is:
$$ C = \sum_{i=1}^{N} c_i \alpha_i T_i $$
where:
- $c_i$ is the color of the $i$-th Gaussian (often view-dependent, derived from Spherical Harmonics).
- $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$ is the accumulated transmittance before the $i$-th Gaussian.
- $\alpha_i$ is the evaluated opacity of the $i$-th Gaussian at the pixel location $x \in \mathbb{R}^2$:
$$ \alpha_i = o_i \exp \left( -\frac{1}{2} (x - \mu_i')^T (\Sigma_i')^{-1} (x - \mu_i') \right) $$
where $o_i$ is the base opacity of the Gaussian and $\mu_i'$ is the 2D projected mean.

## 4. Backward Pass: Partial Derivatives

To train the 3DGS model, we must backpropagate the loss gradients from the final pixel color back to the 3D Gaussian parameters ($\mu, \Sigma, o, c$). Let $\frac{\partial L}{\partial C}$ be the gradient of the loss with respect to the rendered color.

### 4.1 Gradients w.r.t Color and Alpha
$$ \frac{\partial L}{\partial c_i} = \frac{\partial L}{\partial C} \alpha_i T_i $$

To compute gradients with respect to $\alpha_i$, we use the chain rule on the compositing equation. Since $C = \sum_{j=1}^N c_j \alpha_j T_j$, the derivative w.r.t $\alpha_i$ requires isolating $\alpha_i$'s contribution to its own term and all subsequent terms $j > i$ (since it affects their transmittance $T_j$):
$$ \frac{\partial L}{\partial \alpha_i} = \frac{\partial L}{\partial C} \left( c_i T_i - \frac{1}{1 - \alpha_i} \sum_{j=i+1}^N c_j \alpha_j T_j \right) $$
This is typically computed efficiently in a back-to-front (or front-to-back using cached variables) accumulation pass.

### 4.2 Gradients w.r.t 2D Mean and Covariance
Given $\frac{\partial L}{\partial \alpha_i}$, we backpropagate into the spatial components. Let $\Delta x = x - \mu_i'$ and $\Sigma'^{-1} = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$ (conic). The exponent is $p = -\frac{1}{2} (a \Delta x^2 + 2b \Delta x \Delta y + c \Delta y^2)$.
$$ \frac{\partial L}{\partial \mu_i'} = \frac{\partial L}{\partial \alpha_i} \alpha_i \left( (\Sigma_i')^{-1} (x - \mu_i') \right) $$
$$ \frac{\partial L}{\partial (\Sigma_i')^{-1}} = -\frac{1}{2} \frac{\partial L}{\partial \alpha_i} \alpha_i (x - \mu_i')(x - \mu_i')^T $$

### 4.3 Gradients w.r.t 3D Parameters
Using the chain rule on $\Sigma' = J W \Sigma W^T J^T$:
$$ \frac{\partial L}{\partial \Sigma} = W^T J^T \frac{\partial L}{\partial \Sigma'} J W $$
Similarly, for the 3D mean $\mu$, the gradient flows through both the 2D mean $\mu'$ and the Jacobian $J$ (since $J$ depends on $t_z$, which depends on $\mu$):
$$ \frac{\partial L}{\partial \mu} = W^T \frac{\partial L}{\partial \mu'} + \text{gradients from } \frac{\partial L}{\partial \Sigma'} \text{ via } J $$

## 5. Algorithmic State and Caching Requirements
To perform the exact backward pass efficiently, the forward pass must cache several intermediate variables to global memory, as the backward kernel cannot recompute the exact spatial layout and sorting without them.

1. **Gaussian Geometry Data:** 2D means ($\mu'$), 2D conics ($(\Sigma')^{-1}$), depths, and bounding radii for all Gaussians.
2. **Tile and Sorting Data:** The 64-bit keys and sorted Gaussian indices from the Radix Sort, as well as tile offsets (start and end indices for Gaussians overlapping each 16x16 tile).
3. **Pixel State:** Final accumulated color $C$ and final accumulated transmittance $T_{final}$ for every pixel, allowing the backward pass to reverse the front-to-back blending.
