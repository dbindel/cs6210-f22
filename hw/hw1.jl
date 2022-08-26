### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ ca0206b9-e4ee-4f5f-8685-80efaf7ff0ba
using LinearAlgebra

# ╔═╡ 97e46e8a-227b-11ed-3ad4-bf493f45d214
md"""
# HW1 for CS 6210

You may (and probably should) talk about problems with the each other, with the TA, and with me, providing attribution for any good ideas you might get. Your final write-up should be your own.
"""

# ╔═╡ 6294b596-0cb1-4613-b9a6-057e0bad6d5b
md"""
## 1. All about you

- How do you prefer to be called?
- Why are you taking the class?
- Are there things you particularly hope to learn?
- Do you have any concerns (about background, schedule, etc)?
- Is there anything else I should know about your situation?
"""

# ╔═╡ 19d8e00f-2e9c-44f1-af54-3272300f4135
md"""
## 2. A little differentiation

Differentiate $\|Ax\|^2$ with respect to $x$ and $A$, and write the result in the form

$$\delta[ \|Ax\|^2 ] = g^T \delta x + \langle G, \delta A \rangle_F$$

You may use the following tester as a sanity check.
"""

# ╔═╡ 5fc27366-77d8-4ea2-8e7b-748c483a469e
function check_p2(A, x, G, g, h = 1e-4)

	# Pick a random direction
	δA = rand(size(A)...)
	δx = rand(size(x)...)

	# Compute an estimated directional derivative by finite differences
	np = norm((A+h*δA)*(x+h*δx))^2
	nm = norm((A-h*δA)*(x-h*δx))^2
	dn_fd = (np-nm)/(2*h)

	# Compute the analytical directional derivative
	dn_analytical = dot(g, δx) + dot(G, δA)

	# Return the relative error between the two computations
	abs((dn_fd-dn_analytical)/dn_fd)
end

# ╔═╡ 0ac2e51e-99e3-4f7a-b5e7-70a897968413
md"""
## 3: Norm!

For $A = xy^T$, verify the following

- $\|A\|_1 = \|x\|_1 \|y\|_\infty$
- $\|A\|_\infty = \|x\|_\infty \|y\|_1$
- $\|A\|_F = \|x\|_2 \|y\|_2$
- $\|A\|_2 = \|x\|_2 \|y\|_2$
"""

# ╔═╡ a2542e2f-a118-4473-a84a-0245ef2381f8
md"""
## 4: Seeking structure

Suppose $A = I + uv^T$ for $u, v \in \mathbb{R}^n$.  Rewrite each of the computations in the following code to take $O(n)$ time:
"""

# ╔═╡ d0344353-0e3b-4e23-84cd-9618115178b4
let
	n = 1000     # Change this to 10000 as a sanity check!
	u = rand(n)
	v = rand(n)
	x = rand(n)

	# Reference computations
	A = I + u*v'
	y1 = A*x      # Part 1
	y2 = A'*x     # Part 2
	d = diag(A)   # Part 3
	t = tr(A)     # Part 4
end

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
# ╟─97e46e8a-227b-11ed-3ad4-bf493f45d214
# ╠═ca0206b9-e4ee-4f5f-8685-80efaf7ff0ba
# ╟─6294b596-0cb1-4613-b9a6-057e0bad6d5b
# ╟─19d8e00f-2e9c-44f1-af54-3272300f4135
# ╠═5fc27366-77d8-4ea2-8e7b-748c483a469e
# ╟─0ac2e51e-99e3-4f7a-b5e7-70a897968413
# ╟─a2542e2f-a118-4473-a84a-0245ef2381f8
# ╠═d0344353-0e3b-4e23-84cd-9618115178b4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
