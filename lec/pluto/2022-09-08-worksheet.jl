### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 4577430e-17fe-4b0f-ae14-122dc37486f6
using LinearAlgebra

# ╔═╡ 9b69b962-4973-4608-b5d5-ee539ff06672
md"""
# In-class notebook for 2022-09-08
"""

# ╔═╡ c8f6167b-caec-42e4-8af3-3866f4514b93
md"""
## 1. Evil `inv`

The `inv` command in Julia computes an LU factorization and then solves a linear system for each column of the identity.  We can write this in Julia as `A\I`.

As we will see next Tuesday, solving a linear system by Gaussian elimination with pivoting is usually backward stable; that is, there is usually a usually a "small" $E$ such that the computed solution to $Ax = b$ satisfies $(A+E)x = b$.  But `inv` is not backward stable.  How can this be?
"""

# ╔═╡ 450fa7e7-66a4-4d51-afd5-4f74f3a024cb
md"""
## 2. Determinants

Suppose $A = LU$ is an (unpivoted) LU factorization.  Argue that the determinant of $A$ is the product of the diagonal entries of $U$.
"""

# ╔═╡ ed978b53-4ef2-44d5-aaa4-330c8e2a2cd5
md"""
## 3. Tricky tridiagonal

Assuming no pivoting is needed, write a short routine to factor the tridiagonal matrix

$$A = 
\begin{bmatrix} 
\alpha_1 & \beta_1 \\ 
\gamma_2 & \alpha_2 & \beta_2 \\ 
& \gamma_3 & \alpha_3 & \beta_3 \\ 
&& \ddots & \ddots & \ddots \\
&&& \gamma_{n-1} & \alpha_{n-1} & \beta_{n-1} \\
&&&& \gamma_{n} & \alpha_n \end{bmatrix}$$

as

$$A =
\begin{bmatrix}
1 \\
l_2 & 1 \\
& l_3 & 1 \\
&& \ddots & \ddots \\
&&& l_{n} & 1
\end{bmatrix}
\begin{bmatrix}
u_1 & v_1 \\
& u_2 & v_2 \\
&& \ddots & \ddots \\
&&& u_{n-1} & v_{n-1} \\
&&&& u_n
\end{bmatrix}$$
"""

# ╔═╡ 8ad0abb9-05da-4083-ac1c-a56e2f2bd7b7
md"""
## 4. Follow the arrow

Consider the linear system

$$\begin{bmatrix} I & u \\ v^T & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$$

Show by eliminating $y$ that $(I+uv^T) x = b$.  By eliminating $x$ first, give an $O(n)$ algorithm for solving the system.  Show that this implies that $(I+uv^T)^{-1}$ has a simple form.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "ac1187e548c6ab173ac57d4e72da1620216bce54"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╟─9b69b962-4973-4608-b5d5-ee539ff06672
# ╠═4577430e-17fe-4b0f-ae14-122dc37486f6
# ╟─c8f6167b-caec-42e4-8af3-3866f4514b93
# ╟─450fa7e7-66a4-4d51-afd5-4f74f3a024cb
# ╟─ed978b53-4ef2-44d5-aaa4-330c8e2a2cd5
# ╟─8ad0abb9-05da-4083-ac1c-a56e2f2bd7b7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
