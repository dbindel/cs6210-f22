### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ 97e46e8a-227b-11ed-3ad4-bf493f45d214
md"""
# HW5 for CS 6210

You may (and probably should) talk about problems with the each other, with the TA, and with me, providing attribution for any good ideas you might get. Your final write-up should be your own.
"""

# ╔═╡ 7a18f12a-94db-4b8e-907d-61038796ef1e
md"""
## 1. Interesting identity

Suppose $X, Y \in \mathbb{R}^{n \times k}$.  Show that if $\lambda \neq 0$ is an eigenvalue of $XY^T$, then

$$\begin{bmatrix} -\lambda I & X \\ Y^T & -I \end{bmatrix}$$

is singular.  Via this formulation, show that $\lambda$ must also be an eigenvalue of $Y^T X$.
"""

# ╔═╡ 6aa9a580-41f6-40dd-98e8-bb6ba4892d54
md"""
## 2. Real rotation

Suppose $A \in \mathbb{R}^{2 \times 2}$ has a complex pair of eigenvalues $\lambda_{\pm} = \rho \exp(\pm i \theta)$ (with nonzero imaginary part) and eigenvectors $z_{\pm} = u \pm i v$.
Show that

$$A = W (\rho G) W^{-1}$$

where

$$W = \begin{bmatrix} u & v \end{bmatrix}, \quad
  G = \begin{bmatrix} 
        \cos(\theta) & -\sin(\theta) \\ 
        \sin(\theta) & \cos(\theta) 
      \end{bmatrix}.$$

Why must $W$ be invertible?
"""

# ╔═╡ ca16b7cc-bf54-4176-9b84-d67c01e746b6
md"""
## 3. Vector variations

Suppose $\lambda$ is an isolated eigenvalue of $A$ with eigenvector $x$, normalized
so that $l^* x = 1$.
Differentiate the eigenvalue equation $Ax = \lambda x$ subject to this linear constraint in order to obtain a linear system for derivatives of $x$ and $\lambda$ under small variations of $A$.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─97e46e8a-227b-11ed-3ad4-bf493f45d214
# ╟─7a18f12a-94db-4b8e-907d-61038796ef1e
# ╟─6aa9a580-41f6-40dd-98e8-bb6ba4892d54
# ╟─ca16b7cc-bf54-4176-9b84-d67c01e746b6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
