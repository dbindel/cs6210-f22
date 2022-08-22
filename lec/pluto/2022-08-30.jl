### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ e2b81096-20c2-11ed-19f2-619ba8119365
using LinearAlgebra

# ╔═╡ 1ab5179f-bacd-4ac0-9367-442f76ebb547
using Plots

# ╔═╡ 5c4b5dcf-47df-44c9-845c-e59c3abc9736
using SparseArrays

# ╔═╡ e68410ec-5942-4919-88f4-a2ea108d390f
md"""
# Notebook for 2022-08-30

There are four purposes to this notebook:

1. From the perspective of the main course material, we'd like to explore how implementation details like loop order and choices of building blocks lead to fairly significant differences in performance.

2. We'd also like to look at some common matrix structures that appear throughout the class, and introduce specialized Julia types for the same.

3. I'd also like to introduce you to the [Julia programming language](https://julialang.org/) and working with [Pluto notebooks](https://github.com/fonsp/Pluto.jl).  The Julia home page includes a number of resources for learning Julia, but I recommend the [links from the MIT "Computational Thinking with Julia" course](https://computationalthinking.mit.edu/Spring21/cheatsheets/).

4. Finally, I'd like to make some points about [timing with Julia](https://www.juliabloggers.com/timing-in-julia/) and some aspects of [writing performant Julia code](https://docs.julialang.org/en/v1/manual/performance-tips).
"""

# ╔═╡ 25560b2d-fd35-42e3-a4e1-4fc6228631d0
md"""
We start by including the `LinearAlgebra` package and the `Plots` package.  We will be using both of these packages for most of the code written in this class.
"""

# ╔═╡ 357169f8-b454-4a78-84ec-296a74a84ca2
md"""
## Matrix algebra vs linear algebra

>  We share a philosophy about linear algebra: we think basis-free, we write basis-free, but when the chips are down we close the office door and compute with matrices like fury. -- Irving Kaplansky on the late Paul Halmos

Linear algebra is fundamentally about the structure of vector spaces and linear maps between them (or bilinear, sesquilinear, and quadratic forms on them).  A matrix represents a linear map with respect to some bases.  Properties of the underlying linear map may be more or less obvious via the matrix representation associated with a particular basis, and much of matrix computations is about finding the right basis (or bases) to make the properties of some linear map obvious.  We also care about finding changes of basis that are "nice" for numerical work.

In some cases, we care not only about the linear map a matrix represents, but about the matrix itself.  For example, the *graph* associated with a matrix $A \in \mathbb{R}^{n \times n}$ has vertices $\{1, \ldots, n\}$ and an edge $(i,j)$ if $a_{ij} \neq 0$.  Many of the matrices we encounter in this class are special because of the structure of the associated graph, which we usually interpret as the "shape" of
a matrix (diagonal, tridiagonal, upper triangular, etc).  This structure is a property of the matrix, and not the underlying linear transformation; change the bases in an arbitrary way, and the graph changes completely.  But identifying and using special graph structures or matrix shapes is key to building efficient numerical methods for all the major problems in numerical linear algebra.

In writing, we represent a matrix concretely as an array of numbers. Inside the computer, a *dense* matrix representation is a two-dimensional array data structure, usually ordered row-by-row or column-by-column in order to accomodate the one-dimensional structure of computer memory address spaces.  While much of our work in the class will involve dense matrix layouts, it is important to realize that there are other data structures!  The "best" representation for a matrix depends on the structure of the matrix and on what we want to do with it.  For example, many of the algorithms we will discuss later in the course only require a black box function to multiply an (abstract) matrix by a vector.
"""

# ╔═╡ c230eae7-b74e-4706-84e8-285097d69fcb
md"""
## Vector basics

There is one common data structure for dense vectors: we store the vector as a sequential array of memory cells.  In this class (and in Julia), all vectors are column vectors by default.  We can construct vector expressions with closed brackets to start and end the vector and semicolons or commas for vertical concatenation.
"""

# ╔═╡ 256d0d02-295d-49a8-b4e7-c3f6969b3ca8
[1; 2; 3]  # Example of a length 3 column vector

# ╔═╡ 3014b737-389d-4db5-91d4-1825301b8a8d
md"""
Julia uses an *adjoint* type to indicate row vectors; the underlying array data structure for the vector elements remains the same.  If `x` is a vector expression, `x'` is the adjoint.  An adjoint times a vector yields a scalar.
"""

# ╔═╡ 5607b546-ba26-45cf-8932-b7fe7b7254d7
[1; 2; 3]' # Example of a length 3 row (adjoint) vector

# ╔═╡ 952b55c3-5813-408d-904d-8874ee51f5b5
[1; 2; 3]'*[1; 2; 3]

# ╔═╡ 2ee18e86-3c6c-4041-9cea-6e1f1c4df777
md"""
In a Julia array constructor, blank spaces are used to denote horizontal concatenation.  But if we horizontally concatenate some scalars, we get a $1$-by-$n$ matrix instead of getting an adjoint vector.
"""

# ╔═╡ 5d9c579a-44bb-437f-8d7d-7efcd63744ea
[1 2 3]

# ╔═╡ 4af4c093-4bf6-4bea-8515-c4c76b688ba4
md"""
The difference between a $1$-by-$n$ matrix and an adjoint vector shows up when we do multiplications.  A $1$-by-$n$ matrix times a vector is a vector of length 1, which is not the same as a scalar in Julia.
"""

# ╔═╡ 6c6d805a-74e0-4a8e-b8a7-41ccacf3ccfc
[1 2 3] * [1; 2; 3]

# ╔═╡ b548746b-8f48-4c25-b59f-5c3d1963bc29
md"""
## Dense matrix basics

There is one common data structure for vectors across a wide variety of languages.  In contrast, there are *two* common data structures for general dense matrices. In Julia (and MATLAB, Fortran, and NumPy), matrices are stored in *column-major* form. For example, an array of the first four positive integers interpreted as a two-by-two column major matrix represents the matrix

$$\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}.$$

The same array, when interpreted as a *row-major* matrix, represents

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}.$$

Unless otherwise stated, we will assume all dense matrices are represented in column-major form for this class.  As we will see, this has some concrete effects on the efficiency of different types of algorithms.

It is sometimes useful to reinterpret the same piece of memory as different types of arrays.  In Julia, we can do this with `reshape`:
"""

# ╔═╡ 771ea5b1-8af4-4ad9-88ea-3abae631669f
reshape(1:4, 2, 2)

# ╔═╡ 1bedbcb8-52c1-4752-9379-b383252a1b0a
reshape([1 3; 2 4], 4)

# ╔═╡ 7a95f0e0-1976-4df0-8211-db1f54382414
md"""
If we want to extract the elements of a multi-dimensional array in memory order, we can also do that with the syntax `X[:]`.
"""

# ╔═╡ 8211824f-b524-4f5f-98e9-a51ca08dbed8
let
	X = [1 3;
	     2 4]
	X[:]
end

# ╔═╡ ccfa2bf2-7685-4960-a0e3-50614866c77e
md"""
## The BLAS

The *Basic Linear Algebra Subroutines* (BLAS) are a standard library interface for manipulating dense vectors and matrices.  There are three *levels* of BLAS routines:

1. These routines act on vectors, and include operations
   such scaling and dot products.  For vectors of length $n$,
   they take $O(n^1)$ time.

2. These routines act on a matrix and a vector, and include operations
   such as matrix-vector multiplication and solution of triangular systems
   of equations by back-substitution.  For $n \times n$ matrices and length
   $n$ vectors, they take $O(n^2)$ time.

3. These routines act on pairs of matrices, and include operations such
   as matrix-matrix multiplication.  For $n \times n$ matrices, they
   take $O(n^3)$ time.

All of the BLAS routines are superficially equivalent to algorithms that can be written with a few lines of code involving one, two, or three nested loops (depending on the level of the routine).  Indeed, except for some refinements involving error checking and scaling for numerical stability, the reference BLAS implementations involve nothing more than these basic loop nests.  But this simplicity is deceptive --- a surprising amount of work goes into producing high performance implementations.
"""

# ╔═╡ e9048ab3-db7f-4bbf-ba29-74429fdbce94
md"""
### Level 1: Dot products

A good "hello world" numerical program in Julia is a dot product. This is a prototypical example of a level-1 BLAS routine. We will start with a relatively verbose version that looks like what you might see in any number of languages.
"""

# ╔═╡ abac878f-f78f-49bb-8635-9c6a9e291041
function mydot(x, y)
	n = length(x)
	result = 0.0
	for i = 1:n
		result += x[i]*conj(y[i])  # Two flops/iteration
	end
	result
end

# ╔═╡ c3775ef5-a8f1-42ca-81e0-7197554d68e8
md"""
This is not a long program, but there are several things to point out.

- The `function` keyword is used to start a function definition that will involve more than one expression. In this case, we take in the arguments `x` and `y`.  An implementation of the function is compiled just-in-time based on the type signature of these arguments (they will usually be `Vector{Float64}`). We could nail this down by writing `function mydot(x :: Vector{Float64}, y :: Vector{Float64})`, but this isn't necessary.
- We initialize `result = 0.0` in the anticipation that we will have a floating-point valued result.
- We use MATLAB-style `for` syntax, where the expression `1:n` indicates the range from 1 to $n$ (including both ends). Note that Julia uses one-based indexing like MATLAB and Fortran (and unlike Python and C, which are zero-based).
- If you are used to MATLAB, notice that Julia uses square brackets for indexing (like C or Python) rather than parentheses (like Fortran or MATLAB).
- The final expression in the block is the return value (in this case, `result`)

There are a number of ways that we could improve this program:

- Make it work with vectors or vector-like iterable containers of some number type `T` (and making sure `result` has type `T`)
- Check that the vectors are the same length (and throw an error if not)

We could certainly do these things by adding to the `mydot` implementation.  But a more idiomatic approach is the `mydot2` function defined below.
"""

# ╔═╡ 3d7a671f-5e45-4d61-945b-94eeece85bfb
mydot2(x, y) = sum(xi*conj(yi) for (xi,yi) in zip(x,y))

# ╔═╡ 7916fe7e-224c-4850-8cea-66e6db68b85c
md"""
This code is even shorter than the last, but again there are several points to make:

- We did not use the `function` keyword to define the function here.  Rather, we said `mydot2(x, y) = stuff` where the stuff was a single expression that is returned.
- The `sum` function sums an iterable list of results.  In this case, the expression passed to `sum` is a so-called *generator comprehension*.
- The `zip` function takes two iterable containers (things like lists or vectors that we can iterate through) and creates an iterator that marches through them together.  So the experession `for (xi,yi) in zip(x,y)` means "walk through `xi` taken from `x` and `yi` taken from `y` in pairs until we've completely traversed either `x` or `y`."
- The fact that we said "traversed *either* `x` or `y`" means that this implementation also does not throw an error when the two iterators are different length.
"""

# ╔═╡ 0495e77e-7e25-4039-a9a1-d8165c9e8e43
md"""
We can sanity check our dot product by making sure we get the same results for the reference implementation of `dot` (which can also be written as an infix expression; the "dot" symbol can be produced by typing `\cdot` and hitting tab immediately after).  The `let` block defines a local scope, like the inside of a function -- the variables defined inside the block are not visible outside.
"""

# ╔═╡ 0883c9d3-1314-487a-ab60-50e78a7d5e8f
let
	x = [1.0; 2.0; 3.0]
	y = [1.0; -1.0; 1.0]
	dref = x⋅y # Can also write as dot(x,y)
	d1 = mydot(x,y)
	d2 = mydot2(x,y)
	md"Sanity check: dot(x,y) = $dref; mydot(x,y) = $d1; mydot2(x,y) = $d2"
end

# ╔═╡ a1fff200-501e-4bdd-af50-973210a6a071
md"""
Because they take $n$ additions and $n$ multiplications, we say these dot product codes take $2n$ flops, or (a little more crudely) $O(n)$ flops.
"""

# ╔═╡ 1e83aa95-ccc9-41b4-88c3-b645ef02aef0
md"""
### Level 2: Matrix-vector products

The dot product is a prototypical example of a Level 1 BLAS (Basic Linear Algebra Subroutines) routine.  The prototypical example of a Level 2 BLAS routine is matrix-vector products.

The version you may have first learned computes each entry of $y = Ax$ as the dot product of a row of $A$ and the vector $x$.  The simplest way to compute this is with two nested loops: the outer one to iterate over rows, and the inner one to compute dot products.
"""

# ╔═╡ de7bacb6-ed1f-457d-8d2f-cabd1a1e9ce6
function matvec1_row(A, x)
	m, n = size(A)
	y = zeros(m)
	for i = 1:m
		for j = 1:n
			y[i] += A[i,j]*x[j]
		end
	end
	y
end

# ╔═╡ 86e3332b-81f7-464b-a3c7-ff05a68525e3
md"""
Alternately, we can replace the inner loop by a call to `dot`. The first argument to `dot` is a *view* of a row of the matrix $A$ -- that is, something that looks like a vector, but does not have its own storage.  We could replace `view(A,i,:)` with `A[i,:]`, but that involves making a copy of each row.
"""

# ╔═╡ dada0984-8288-4cd9-8b21-ad9adbb7e4ad
function matvec2_row(A,x)
	m, n = size(A)
	y = zeros(m)
	for i = 1:m
		y[i] = dot(view(A,i,:), x)
	end
	y
end

# ╔═╡ 604430c0-0dd4-4d36-bb37-5305944cd913
md"""
Another way to think about matrix-vector products is as computing a linear combination of the columns of the matrix.  From a coding perspective, this just involves exchanging the order of the $i$ and $j$ loops in our previous implementation.
"""

# ╔═╡ c9c8634d-c3c6-494d-bd77-4f6512dc228c
function matvec1_col(A, x)
	m, n = size(A)
	y = zeros(m)
	for j = 1:n
		for i = 1:m
			y[i] += A[i,j]*x[j]
		end
	end
	y
end

# ╔═╡ 75d07949-9106-4cba-a7e3-b748cb27270a
md"""
We can replace the inner loop by an equivalent `axpy` ($\alpha x + y$ update) line.  The dotted versions of the operations are generally used for elementwise operations (as opposed to matrix-vector or matrix-matrix products); but they are also used to produce a vectorized expression without creating intermediate temporaries, which is helpful in this context.
"""

# ╔═╡ a8bc81a8-d8f6-4e0c-91ab-d3422ff6860e
function matvec2_col(A,x)
	m, n = size(A)
	y = zeros(m)
	for j = 1:n
		y[:] .+= view(A,:,j) .* x[j]
	end
	y
end

# ╔═╡ f758f229-0615-4043-ae5e-b1d4152fd051
let
	A = rand(10,10)
	x = rand(10)
	yref = A*x
	e1c = norm(yref-matvec1_col(A, x))/norm(yref)
	e1r = norm(yref-matvec1_row(A, x))/norm(yref)
	e2c = norm(yref-matvec2_col(A, x))/norm(yref)
	e2r = norm(yref-matvec2_row(A, x))/norm(yref)
	md"""
Errors on a random test case, relative to reference version:
- `matvec1_col`: $e1c
- `matvec1_row`: $e1r
- `matvec2_col`: $e2c
- `matvec2_row`: $e2r
"""
end

# ╔═╡ eda384ac-cd36-46a3-bdd6-dd9be7dd9695
md"""
What are the performance implications of the different ways of writing matvec?  Let's find out with a simple benchmark (it is possible to do much more elaborate benchmarks using the Julia `BenchmarkTools` package).  We compare the GFlop/s rate (billions of floating point operations per second) for each of the implementations in turn, taking the best of 50 runs to mitigate performance interference effects.
"""

# ╔═╡ 7da00331-f3d0-4f01-a011-a6d033dc885a
function benchmark_matvec(sizes)
	results = []
	for s in sizes
		t0 = Inf
		t1 = Inf
		t2 = Inf
		t3 = Inf
		t4 = Inf
		for trial = 1:50
			A = rand(s, s)
			x = rand(s)
			t0 = min(t0, @elapsed A*x)
			t1 = min(t1, @elapsed matvec1_row(A, x))
			t2 = min(t2, @elapsed matvec1_col(A, x))
			t3 = min(t3, @elapsed matvec2_row(A, x))
			t4 = min(t4, @elapsed matvec2_col(A, x))
		end
		push!(results, (s, 2*s^2/t0/1e9, 2*s^2/t1/1e9, 2*s^2/t2/1e9, 2*s^2/t3/1e9, 2*s^2/t4/1e9))
	end
	results
end

# ╔═╡ b084e6db-c06b-469f-8725-bc76c0f4328a
md"""
The short version: the row-oriented codes run at less than half the speed of the column-oriented ones!  Why is this?  After all, all these program variants take $2n^2$ flops.  But on modern machines, counting flops is at best a crude way to reason about how run times scale with problem size.  This is because in many computations, the time to do arithmetic is dominated by the time to fetch the data into the processor!  As we will discuss in more detail shortly, this is the reason for the relatively high performance of the column-oriented algorithms.

The other punchline here is that the built-in matrix-vector routine (from a BLAS implementation) runs a little more than twice the speed of my fastest hand-written code.

All of these runs are on my current laptop, a Macbook M1 Pro.
"""

# ╔═╡ 1b7a257d-2204-4708-af0b-5212e5d9fc9d
let sizes=[]
	for i = 1:9
		push!(sizes, i*128-1)
		push!(sizes, i*128)
		push!(sizes, i*128+1)
	end
	r = benchmark_matvec(sizes)
	plot([t[1] for t in r], [t[2] for t in r], label="Built-in")
	plot!([t[1] for t in r], [t[3] for t in r], label="Row-oriented")
	plot!([t[1] for t in r], [t[4] for t in r], label="Col-oriented")
	plot!([t[1] for t in r], [t[5] for t in r], style=:dash, label="Row-oriented (2)")
	plot!([t[1] for t in r], [t[6] for t in r], style=:dash, label="Col-oriented (2)")
end

# ╔═╡ 7055b7aa-dad0-4c37-93f6-47754a521e6b
md"""
There are again a few features of Julia worth plotting out in this plotting code:

- The function `push!` modifies the first argument (`sizes`), adding an element to the end of it.  As a matter of convention, Julia routines that modify their arguments are usually marked with an exclamation mark (similar to the convention used in Scheme).
- The expression `[t[1] for t in r]` is an *array comprehension*.  This constructs an array by iterating over the elements in another array (`r` in this case) and applying some function (taking the first element of a tuple, in this case).
- The plot function takes a variety of *keyword arguments*.  In Julia, arguments are either indicated by position or by keyword.  Any keyword argument must have a default value, while positional arguments may or may not have a default value.  In the function definition, the positional and keyword arguments are separated by a semicolon, e.g. `function foo(p1, p2=1.0; k1=true, k2="bar")`.
- The `:dash` argument is an example of a *symbol*.  A symbol in Julia can effectively be treated like an immutable string.  Symbols can be compared to each other very quickly (constant time).
"""

# ╔═╡ a31a5d5e-392e-4033-920f-1c52e9fb15dc
md"""
### Level 3: Matrix-matrix multiply

The classical algorithm to compute $C := C + AB$ involves three nested loops.
"""

# ╔═╡ 56a4949f-5b23-45f4-931e-2e1ad369f0e2
function my_matmul!(A, B, C)
	m, n = size(A)
	n, p = size(B)
	for i = 1:m
		for j = 1:n
			for k = 1:p
				C[i,j] += A[i,k]*B[k,j]
			end
		end
	end
	C
end

# ╔═╡ b32965f1-9694-47e4-b7c9-bc01aa382e60
md"""
This is sometimes called an *inner product* variant of the algorithm, because the innermost loop is computing a dot product between a row of $A$ and a column of $B$.  But addition is commutative and associative, so we can sum the terms in a matrix-matrix product in any order and get the same result.  And we can interpret the orders!  A non-exhaustive list is:

- `ij(k)` or `ji(k)`: Compute entry $c_{ij}$ as a product of row $i$ from $A$ and column $j$ from $B$ (the *inner product* formulation)
- `k(ij)`: $C$ is a sum of outer products of column $k$ of $A$ and row $k$ of $B$ for $k$ from $1$ to $n$ (the *outer product* formulation)
- `i(jk)` or `i(kj)`: Each row of $C$ is a row of $A$ multiplied by $B$
- `j(ik)` or `j(ki)`: Each column of $C$ is $A$ multiplied by a column of $B$

At this point, we could write down all possible loop orderings and run a timing experiment, similar to what we did with matrix-vector multiplication.  But the truth is that high-performance matrix-matrix multiplication routines use another access pattern altogether, involving more than three nested loops, and we will describe this now.
"""

# ╔═╡ ffe15400-9b67-40ff-896c-32403a08bc89
md"""
## Memory matters

A detailed discussion of modern memory architectures is beyond the scope of these notes, but there are at least two basic facts that everyone working with matrix computations should know:

1. Memories are optimized for access patterns with *spatial locality*:
   it is faster to access entries of memory that are close to each
   other (ideally in sequential order) than to access memory entries that
   are far apart.  Beyond the memory system, sequential access patterns
   are good for *vectorization*, i.e. for scheduling work to be done
   in parallel on the vector arithmetic units
   that are present on essentially all modern processors.
2. Memories are optimized for access patterns with *temporal locality*;
   that is, it is much faster to access a small amount of data repeatedly
   than to access large amounts of data.

The reason for the improved performance of the column-oriented variants of the matrix-vector products is that they scan through the matrix in order; that is, they have good spatial locality.  Unfortunately, level 1 and 2 BLAS routines like dot products and matrix-vector products involve work is proportional to the amount of memory used, and so it is difficult to organize for temporal locality.

On the other hand, level 3 BLAS routines do $O(n^3)$ work with $O(n^2)$ data, and so it is possible for a clever level 3 BLAS implementation to take advantage of the performance advantages offered by temporal locality.  The main mechanism for optimizing access patterns with temporal locality is a system of *caches*, fast and (relatively) small memories that can be accessed more quickly (i.e. with lower latency) than the main memory. To effectively use the cache, it is helpful if the *working set* (memory that is repeatedly accessed) is smaller than the cache size.
"""

# ╔═╡ 5f3507b9-610a-4e21-a846-3db72fdd4dd4
md"""
### Blocking and performance

The basic matrix multiply organizations described well before don't do well with temporal locality. A better organization would let us move some data into the cache and then do a lot of arithmetic with that data.  The key idea behind this better organization is *blocking*.

When we looked at the inner product and outer product organizations in the previous sections, we really were thinking about partitioning $A$ and $B$ into rows and columns, respectively.  For the inner product algorithm, we wrote $A$ in terms of rows and $B$ in terms of columns

$$\begin{bmatrix} a_{1,:} \\ a_{2,:} \\ \vdots \\ a_{m,:} \end{bmatrix} \begin{bmatrix} b_{:,1} & b_{:,2} & \cdots & b_{:,n} \end{bmatrix},$$

and for the outer product algorithm, we wrote $A$ in terms of colums and $B$ in terms of rows

$$\begin{bmatrix} a_{:,1} & a_{:,2} & \cdots & a_{:,p} \end{bmatrix}
\begin{bmatrix} b_{1,:} \\ b_{2,:} \\ \vdots \\ b_{p,:} \end{bmatrix}.$$

More generally, though, we can think of writing $A$ and $B$ as *block matrices*:

$$\begin{align*}
  A &=
  \begin{bmatrix}
    A_{11} & A_{12} & \ldots & A_{1,p_b} \\
    A_{21} & A_{22} & \ldots & A_{2,p_b} \\
    \vdots & \vdots &       & \vdots \\
    A_{m_b,1} & A_{m_b,2} & \ldots & A_{m_b,p_b}
  \end{bmatrix} \\
  B &=
  \begin{bmatrix}
    B_{11} & B_{12} & \ldots & B_{1,p_b} \\
    B_{21} & B_{22} & \ldots & B_{2,p_b} \\
    \vdots & \vdots &       & \vdots \\
    B_{p_b,1} & B_{p_b,2} & \ldots & B_{p_b,n_b}
  \end{bmatrix}
\end{align*}$$

where the matrices $A_{ij}$ and $B_{jk}$ are compatible for matrix multiplication.  Then we we can write the submatrices of $C$ in terms of the submatrices of $A$ and $B$

$$C_{ij} = \sum_k A_{ij} B_{jk}.$$
"""

# ╔═╡ a61bb975-0fd3-43dd-83e4-386f9485af5c
md"""
### The lazy approach to performance

An algorithm like matrix multiplication seems simple, but there is a lot under the hood of a tuned implementation, much of which has to do with the organization of memory.  We often get the best "bang for our buck" by taking the time to formulate our algorithms in block terms, so that we can spend most of our computation inside someone else's well-tuned matrix multiply routine (or something similar).  There are several implementations of the Basic Linear Algebra Subroutines (BLAS), including some implementations provided by hardware vendors and some automatically generated by tools like ATLAS.  The best BLAS library varies from platform to platform, but by using a good BLAS library and writing routines that spend a lot of time in level 3 BLAS operations (operations that perform $O(n^3)$ computation on $O(n^2)$ data and can thus potentially get good cache re-use), we can hope to build linear algebra codes that get good performance across many platforms.

This is also a good reason to use systems like Julia, MATLAB, or NumPy (built appropriately): they uses pretty good BLAS libraries, and so you can often get surprisingly good performance from it for the types of linear algebraic computations we will pursue.
"""

# ╔═╡ 8ca3c772-f019-4abd-8f93-1e5555dd7739
md"""
## Special matrix structures

Julia supports [special types for several matrix structures](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Special-matrices).  We will encounter all of these (and more) in this course, so it is worth spending some time with them now.
"""

# ╔═╡ 43bc6728-a2b9-4098-854d-5bd91ca79c18
md"""
### Identity and scaling

The symbol `I` in Julia is the identity operator.  This is a special instance of the type `UniformScaling`, which corresponds to multiplication by a scalar constant.  The only thing that needs to be saved for a `UniformScaling` is the scaling factor, so it is quite storage efficient.  A scaling operator is size-agnostic: the symbol `I` can be used to indicate the identity on whatever size vector space you like, with the size determined from context.  A scaling operator can even be concatenated to form a larger matrix, but doing so involves casting the type from a scaling (only needs one number) to an $n$-by-$n$ dense matrix representation.
"""

# ╔═╡ ed5dac5d-3e68-45ad-b15a-df213ebcbefe
I

# ╔═╡ f4f3ae7d-5303-4099-82fa-49c6e5105f7e
2.0*I

# ╔═╡ 9b94f864-bf19-407f-a460-c200e337b721
I*[1.0; 2.0; 3.0]

# ╔═╡ d216131a-be52-4a49-96e9-d631ff5829a5
[I rand(3)]

# ╔═╡ f2837b72-7d12-41c0-8707-e66acbdd2014
md"""
Of course, multiplication by a scalar can also be carried out directly.  The following two expressions give the exact same result.
"""

# ╔═╡ bb9402ab-3ded-430c-a6c6-25ed02f0d72d
2.0*[1.0; 2.0; 3.0]

# ╔═╡ 8a440617-43bf-490d-b1c7-aede074ea3cb
(2*I)*[1.0; 2.0; 3.0]

# ╔═╡ c015ba3f-e114-4a17-aa9f-c5e871a25ff1
md"""
### Diagonal matrices

A diagonal matrix is a matrix that is only nonzero in the diagonal elements.  The action of a diagonal matrix on a vector is a non-uniform scaling.  In Julia, the `Diagonal` type can be used to represent a diagonal matrix.  The `Diagonal` type only stores the diagonal elements explicitly.
"""

# ╔═╡ fedd9488-23c9-4d02-ac11-a6311f416d71
Diagonal([1.0; 2.0; 3.0])

# ╔═╡ 3b30e859-42ae-42c5-85ea-668f7b160815
Diagonal([1.0; 2.0; 3.0]) * [1.0; 1.0; 1.0]

# ╔═╡ b48627d4-b693-4b82-bdc8-ac6ac30bc686
md"""
Pre-multiplying or post-multiplying a matrix by a diagonal scales the rows or the columns of that matrix, respectively.  We illustrate with the matrix of all ones:
"""

# ╔═╡ 2f47ee49-46a8-4c8a-87de-d0ff56bb89d6
Diagonal([1.0; 2.0; 3.0])*ones(3,3)

# ╔═╡ 9becc87b-7960-4258-9dc6-fc2364a9fed9
ones(3,3)*Diagonal([1.0; 2.0; 3.0])

# ╔═╡ 6372dc4b-b0e9-4bee-bc57-01df67c83b6f
md"""
As with uniform scaling, we don't need to explicitly form a diagonal matrix to consisely represent products with the matrix.  We can also use the elementwise product operator (`.*`).  This operator knows how to "broadcast" so that multiplying a vector elementwise with a matrix makes sense.
"""

# ╔═╡ 64a25b24-b7c9-4413-89f3-d6afbc148bac
[1.0; 2.0; 3.0] .* ones(3,3)

# ╔═╡ 76b67b52-520e-4aa3-ae50-6cf5aa14705b
ones(3,3) .* [1.0 2.0 3.0]

# ╔═╡ 92355bbe-d868-479c-857b-50109c9eb09d
md"""
The function `diagm` creates a full dense representation of a diagonal matrix.  It can also be used to fill in other matrices than the main diagonal (positive numbers used for superdiagonals, negative numbers for subdiagonals).  This is useful as a building block for constructing more complicated dense matrices, but we do not recommend it if you just want to work with a diagonal matrix!
"""

# ╔═╡ c67a6934-282c-45b2-a50c-cd7fb0e65e93
diagm([1.0; 2.0; 3.0])

# ╔═╡ 90e1f446-f9e2-4ef3-a041-1066e126d25c
diagm(1 => 1:3, -1 => 4:6)

# ╔═╡ 7d9a5f1c-79ce-4cfe-bce0-48fb4aab5523
md"""
Finally, the function `diag` returns a vector of the diagonal elements of a matrix.
"""

# ╔═╡ c3a14e4c-a882-407c-9734-c8af804a4f8c
let
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 9.0]
	diag(A)
end

# ╔═╡ 93a3bbd8-f27b-408c-b420-575947b87d3d
md"""
An optional second argument to `diag` specifies which diagonal we want, with positive arguments for superdiagonals and negative elements for subdiagonals (and zero for the main diagonal).
"""

# ╔═╡ dd6cf278-fb91-466c-b22f-c3974db3232b
let
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 9.0]
	diag(A,-1)
end

# ╔═╡ bba73dab-15d8-4a81-b616-1db21ca3aa02
md"""
### Tridiagonal and bidiagonal

A *tridiagonal* matrix is a matrix in which the only nonzeros are on the first subdiagonal, the main diagonal, and the first superdiagonal.
"""

# ╔═╡ a9eddef4-70fe-44e2-827d-85d58f10e1fe
Tridiagonal(ones(9), Vector{Float64}(1:10), 3*ones(9))

# ╔═╡ f6ca2d28-6c27-45a9-b573-28a5d41f4283
md"""
We will most frequently deal with symmetric tridiagonals, for which there is a more specialized type.
"""

# ╔═╡ 305d2404-57d1-4465-9397-0880e4ea1e4d
SymTridiagonal(2*ones(10), -1*ones(9))

# ╔═╡ 52e535bc-acae-4ff5-8df3-6bb4e038b161
md"""
We will also sometimes be interested in upper or lower bidiagonal matrices where either the superdiagonal is nonzero (upper bidiagonal) or the subdiagonal is nonzero (lower bidiagonal), but not both.
"""

# ╔═╡ dad9cb8a-6556-4ec6-bc4d-09e16921a7ca
Bidiagonal(Vector{Float64}(1:10), ones(9), :L)

# ╔═╡ e4011132-7be8-4e98-9c0b-e965521d8a58
Bidiagonal(Vector{Float64}(1:10), ones(9), :U)

# ╔═╡ a1729aed-0266-4c84-a10c-6d36d90dc778
md"""
The bidiagonal and tridiagonal types in Julia only require storage for the nonzero parameters that define the matrix.  Dense representations of these matrices can also be constructed using `diagm`; but because these dense representations include explicit zeros, they take more memory.  Also, calling routines like matrix multiplication and linear solves is much faster with the specialized tridiagonal and bidiagonal representations than with general dense representations.

As with uniform scaling and diagonal scaling, we can implement matrix-vector products with bidiagonals and tridiagonals with only a little bit of code -- though more code than we use for the others.
"""

# ╔═╡ 516b0165-17f4-4761-b8eb-40bb126ea20a
let
	x = rand(10)

	# Compute T*x using the SymTridiagonal matrix multiply
	T = SymTridiagonal(2*ones(10), -ones(9))
	y1 = T*x

	# Compute T*x "by hand"
	y2 = 2*x                 # Main diagonal contribution
	y2[1:end-1] -= x[2:end]  # Superdiagonal contribution
	y2[2:end] -= x[1:end-1]  # Subdiagonal contribution

	# Check that they are the same (small relative error)
	norm(y1-y2)/norm(y1)
end

# ╔═╡ 40954f42-e4cc-4ef0-898a-a28e4b5f6131
md"""
There are some cases where it is useful to consider *narrowly-banded* matrices in which there are just a few nonzero super and subdiagonals, but maybe more than one.  Julia does not have explicit type system support for these matrices, but they are supported in linear algebra libraries like LAPACK.
"""

# ╔═╡ d7be1399-0fe8-47ed-a84c-3bdce4a87a8c
md"""
### Triangular and Hessenberg matrix views

An upper triangular matrix has all its nonzeros on or above the main diagonal.  Similarly, a lower triangular matrix has all its nonzeros on or below the main diagonal.  We say a matrix is *unit* lower or upper triangular if it is triangular and the diagonal elements are all one.

The triangular types in Julia are matrix *views*: they do not "own" their own storage, but instead provide an interpretation of an underlying dense matrix.  These views use the usual column-major dense layout, but ignore the value of all the "structural" zeros or ones in the matrix, treating those elements as their proper values.
"""

# ╔═╡ 89d035e2-174e-41a0-8760-8d77aa0f6dba
# Example of an upper triangular view
let
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 9.0]
	U = UpperTriangular(A)
end

# ╔═╡ b086adea-46cc-49d8-9ec2-cdbcb32012c9
# Example of a unit lower triangular view
let
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 9.0]
	L = UnitLowerTriangular(A)
end

# ╔═╡ c80fb416-ad1f-4944-94b7-d9d2eb85a892
# Changing an element in the view changes the original matrix storage
let
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 9.0]
	L = UnitLowerTriangular(A)
	L[3,1] = 0.0
	A
end

# ╔═╡ 7093ab66-ff9c-4abc-ac4c-e417f0577e75
md"""
An *upper Hessenberg* matrix is zero below the first subdiagonal.  That is, it looks like a triangular matrix with one additional nonzero diagonal.
"""

# ╔═╡ d8099871-1586-4c2f-9ce5-0e7f4f2f66df
let
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 9.0]
	H = UpperHessenberg(A)
end

# ╔═╡ fe5e7d42-6c62-42c6-afdf-edb6eec13b72
md"""
Hessenberg matrices appear frequently in algorithms for nonsymmetric eigenvalue problems and iterative solvers that we will see later in the class.  They also play an important role in certain computations in control theory.
"""

# ╔═╡ 770da26b-9e6c-43ff-8b4b-6cce3ab335da
md"""
### Symmetric and Hermitian views

In a symmetric view or Hermitian view, we treat the elements in the upper or lower triangle of the matrix as the reference values, and evaluate elements in the other triangle by symmetry.
"""

# ╔═╡ 9743c4b9-4f1e-4889-b01e-a5a8fdb5a06d
let
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 9.0]
	Symmetric(A, :U)
end

# ╔═╡ c5191dee-3169-4632-a4cb-c99e509016c2
let
	A = [1.0 4.0 7.0;
	     2.0 5.0 8.0;
	     3.0 6.0 9.0]
	Symmetric(A, :L)
end

# ╔═╡ e5350032-3390-4b9b-9f92-14a18fef40b2
md"""
In the Hermitian case, all non-real parts of the diagonal are ignored.
"""

# ╔═╡ b4092668-2fb0-426a-87fa-16c0f14986bb
let
	A = rand(3,3) + 1im * rand(3,3)
	H = Hermitian(A, :U)
end

# ╔═╡ 14135f52-7e97-4826-addb-ae5590ab1dcc
md"""
### Permutations

A permutation matrix is a square matrix with a single one in each row and column and zeros elsewhere.  A permutation matrix acts on a vector by reordering the vector elements.  We will frequently use permutation matrices *symbolically*, but we generally don't form explicit representations of them, and Julia does not have a standard permutation type in the base library.  Instead, we represent permutations by a permuted index vector.
"""

# ╔═╡ 4c5c4b2e-26d0-44e8-ad21-6fbd7bdc50e4
let
	# Matrix representation for a cyclic permutation
	P = [0 1 0;
	     0 0 1;
	     1 0 0]

	# Indexing operation for a cyclic permutation
	idx = [2; 3; 1]

	x = rand(3)
	norm(P*x-x[idx])
end

# ╔═╡ 28feb904-0f10-4ccf-9b28-75e9354a04b1
md"""
Permutations are a type of orthogonal matrix; that is, if $P$ is a permutation, then $P^T P = I$.  The permutation $P^T$ serves to "undo" the permutation $P$.  We can apply the effect of a transposed permutation by an assignment where we re-index the left-hand side of the assignment (as opposed to re-indexing the right-hand side to apply $P$).
"""

# ╔═╡ 93cf7f8a-d03d-404e-9add-6eb310e37fae
let
	# Matrix representation for a cyclic permutation
	P = [0 1 0;
	     0 0 1;
	     1 0 0]

	# Indexing operation for a cyclic permutation
	idx = [2; 3; 1]

	# Compute y = P'*x implicitly.
	x = rand(3)
	y = zeros(3)
	y[idx] = x

	norm(P'*x-y)
end

# ╔═╡ d350da96-c029-4820-9663-63caf44b834e
md"""
### General sparse matrices

All of the matrices that we have discussed so far are defined by a restrictive nonzero structure.  More general *sparse matrix* representations (supported by the [`SparseArrays` package](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)) allow us to just store nonzero elements of a matrix, but with some flexibility about where those nonzeros belong.  There are several possible types of sparse matrix representation, but the one most frequently used in Julia is the *compressed sparse column* representation.
"""

# ╔═╡ 3cf0ee1e-fb77-48e0-8f06-35e9ea33fa3d
sparse([1, 1, 2, 3], [1, 3, 2, 3], [-1, 1, 2, 0])

# ╔═╡ 2760277d-67c2-490f-9ed4-d7b7ee59c964
md"""
Note that a "structurally nonzero" element in a sparse representation may actually have a zero value that is explicitly stored (e.g. the (3,3) entry in the matrix above).

Behind the scenes, a compressed sparse column representation consists of the (structurally) nonzero elements and their row indices listed in column-major order, along with an array of indices pointing to the start of each column (plus one pointer just past the end of the last column).
"""

# ╔═╡ 540da5db-c41e-4474-8cd9-0bd74229bdd6
Acsc_demo = sparse([1, 1, 2, 3], [1, 3, 2, 3], [-1, 1, 2, 0])

# ╔═╡ 154fda0a-2727-4651-b920-58a7dfb74a88
Acsc_demo.nzval

# ╔═╡ 63ded2d8-882c-439e-bb72-70afb57b4dee
Acsc_demo.rowval

# ╔═╡ 76ef93e6-9bad-440e-bd77-dcb4d03969b5
Acsc_demo.colptr

# ╔═╡ b68f7249-055c-44a1-8941-092c6b895c60
md"""
Compressed sparse column representations are relatively compact in memory, and matrix-vector products with CSC representations can be computed relatively quickly -- though there is a performance penalty that comes from the fact that one needs to do "index-chasing" through the data structure.  However, adding a new nonzero to a compressed sparse representation may involve rewriting the entire data structure, which is not particularly efficient.  Therefore, we usually construct compressed sparse matrices from a "coordinate form" where we provide the rows, columns, and values of the elements as three separate arrays (with summation of elements if the same row and column appear more than once).
"""

# ╔═╡ e58c7064-b0fa-4df7-bab6-401e876e9320
md"""
When we deal with sparse matrices, we are frequently as interested in the locations of the nonzero elements as we are in their values.  We can use a *spy plot* to see the locations of nonzeros as dots in Julia.  We give an example below of a 5%-full random sparse matrix plus a diagonal.
"""

# ╔═╡ b29680b3-34b4-423f-be2b-85d7b919f711
let
	A = I + sprand(100, 100, 0.05)
	spy(A)
end

# ╔═╡ 2c0701ac-b050-4b6a-99f1-5b1901ea72be
md"""
## Data-sparse matrices

The special matrices described so far are all defined in terms of their nonzero structure, or *sparsity pattern*.  However, we will also sometimes be concerned with *data-sparse* matrices -- full matrices that are nonetheless described (in the square case) with fewer than $O(n^2)$ parameters, and may admit fast algorithms for matrix-vector multiplication or other linear algebra operations.

We give a few examples here.
"""

# ╔═╡ 9358644a-5e41-4f28-b7a4-a629c124a2e9
md"""
### Low-rank matrices

A low-rank matrix $A \in \mathbb{R}^{m \times n}$ can be written in outer-product form as

$$A = XY^T$$

where $X \in \mathbb{R}^{m \times k}$ and $Y \in \mathbb{R}^{n \times k}$.  If we have such a matrix representation, we are usually better off not forming the matrix explicitly, but instead operating one factor at a time.
"""

# ╔═╡ 2810765c-c1a7-455e-87a2-08fa28556f15
let
	X = rand(1000,3)
	Y = rand(1000,3)
	b = rand(1000)

	# Form A*b explicitly -- time is O(n^2 k)
    @time z1 = (X*Y')*b

	# Do the same computation without forming A -- time is O(nk)
	@time z2 = X*(Y'*b)
end

# ╔═╡ f1005bbc-4475-4c0d-8b01-f3ca83c4f6ba
md"""
### Circulant, Toeplitz, and Hankel matrices

A *Toeplitz* matrix is constant along the diagonals; a *Hankel* matrix is constant along anti-diagonals.  These types of matrices arise frequently in systems theory and signal processing applications.
"""

# ╔═╡ 462ffdda-6fea-40f3-a6b5-d599db8794cf
# Form Toeplitz matrix with specified superdiagonal and subdiagonal elements
function toeplitz(superd, subd)
	n = length(superd)
	T = zeros(n, n)
	for i = 1:n
		T[i,i:n] = superd[1:n+1-i]
		T[i+1:n,i] = subd[1:n-i]
	end
	T
end

# ╔═╡ bd7ca099-a7b1-409f-b3d9-c3449d153520
toeplitz([1, 2, 3, 4, 5], [-1, -2, -3, -4])

# ╔═╡ 8a036196-0b1f-46b6-a9ad-1e6050b195d2
md"""
A Toeplitz matrix can be embedded in a *circulant* matrix, a special case of a Toeplitz matrix with periodic end conditions.
"""

# ╔═╡ 57d8092c-e527-4645-a616-facb39d0fc23
toeplitz([1, 2, 3, 4, 5, -4, -3, -2, -1], [-1, -2, -3, -4, 5, 4, 3, 2])

# ╔═╡ cddc9ad7-10e1-4a9b-9bbb-0dabfe92bdb9
md"""
Circulant matrices can be written in terms of powers of a cyclic permutation matrix

$$P = \begin{bmatrix} 0 & 1 \\ & 0 & 1 \\ & & & \ddots \\ 1 & 0 & 0 & \dots & 0 \end{bmatrix}$$
"""

# ╔═╡ 010daeea-670a-45b2-8412-7bb12b308d0d
md"""
As we will see later in the course, cyclic permutation matrices are diagonalized by the discrete Fourier transform, and this is a gateway to fast matrix-vector multiplication algorithms.

Hankel, Toeplitz, and circulant matrices are not directly supported in the base Julia language, but they are supported in the external [`ToeplitzMatrices` package](https://github.com/JuliaMatrices/ToeplitzMatrices.jl).
"""

# ╔═╡ 7d453f3f-249e-4aab-ad8e-63890953f3e5
md"""
### Kronecker products

A *Kronecker product* of matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$ is the matrix $A \otimes B \in \mathbb{R}^{(mp) \times (nq)}$ expressed in block form as

$$A \otimes B = \begin{bmatrix} a_{11} B & \dots & a_{1n} B \\ \vdots & \ddots & \vdots \\ a_{m1} B & \dots & a_{mn} B \end{bmatrix}$$

Kronecker products can be computed explicitly in Julia with the `kron` function.
"""

# ╔═╡ ce9f013d-d40c-4f4f-a8b1-e9612cc30df4
kron([1 2; 3 4], [1 1; 1 2])

# ╔═╡ 50ab2c87-6d62-4ea5-9977-b4cc4f0a75bb
md"""
Multiplication of a vector by a Kronecker product represents a matrix triple product

$$(A \otimes B) \operatorname{vec}(X) = \operatorname{vec}(B X A^T)$$

where $\operatorname{vec}(X)$ is the vector associated with listing all the elements of $X$ in column-major order.  This can be accomplished in Julia with the expression `X[:]`.
"""

# ╔═╡ 78612833-d9e5-4cbf-9e37-b343fce03f15
let
	X = [5 7; 
		 6 8]
	X[:]
end

# ╔═╡ 4ca3b4de-225d-4ce3-ba10-587f48f882c0
let
	A = [1 2; 3 4]
	B = [1 1; 1 2]
	X = [5 7; 6 8]
	y1 = kron(A,B)*X[:]
	y2 = (B*X*A')[:]
	norm(y1-y2)/norm(y1)
end

# ╔═╡ a3a8f86c-2ddd-4e01-ad66-3cf30dafc305
md"""
Computing a matrix-vector product directly with a Kronecker product of two $n \times n$ matrix takes $O(n^4)$ time (and space), while computing with two matrix-matrix products takes $O(n^3)$ time and $O(n^2)$ space.

Kronecker product structure appears often in control theory applications and in problems that arise from difference or differential equations posed on regular grids — you should expect to see it for regular discretizations of differential equations where separation of variables works well. There is also a small industry of people working on tensor decompositions, which feature sums of Kronecker products.
"""

# ╔═╡ 59623ea5-9878-44b3-af5a-4ae83d8e9586
md"""
### Low-rank block structure

Even matrices that are not low-rank may still have low-rank blocks (or look like a sum of sparse and low-rank components).  For example, the inverse of a tridiagonal matrix is in general dense, but has rank-1 off-diagonal blocks.
"""

# ╔═╡ e6dcbaff-5231-43e7-97c1-11ea58d4a9ac
let
	A = SymTridiagonal(rand(10), rand(9))
	B = inv(A)
	rank(B[5:10,1:4])
end

# ╔═╡ 9b0fa6f9-b173-48a2-ada2-756eeaaa66b5
md"""
The case where a matrix has low-rank (or nearly low-rank) blocks is surprisingly common in practice.  In problems that come from certain areas of mathematical physics, integral equations, and PDE theory, these matrix structures are ubiquitous. The fast multipole method computes a matrix-vector product for one such class of matrices; and again, there is a cottage industry of related methods, including the H matrices studied by Hackbush and colleagues, the sequentially semi-separable (SSS) and heirarchically semi-separable (HSS) matrices, quasi-separable matrices, and a horde of others. 
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Plots = "~1.31.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "30ad1e922e34dfad32f9cf49b9be0afdf313b5d9"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf0a9940f250dc3cb6cc6c6821b4bf8a4286cf9c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2d908286d120c584abbe7621756c341707096ba4"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.2+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f0956f8d42a92816d2bf062f8a6a6a0ad7f9b937"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.2.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "d9ab10da9de748859a7780338e1d6566993d1f25"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "a19652399f43938413340b2068e11e55caa46b65"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.7"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "85bc4b051546db130aeb1e8a696f1da6d4497200"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.5"

[[deps.StaticArraysCore]]
git-tree-sha1 = "5b413a57dd3cea38497d745ce088ac8592fbb5be"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.1.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "4ad90ab2bbfdddcae329cba59dab4a8cdfac3832"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.7"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─e68410ec-5942-4919-88f4-a2ea108d390f
# ╟─25560b2d-fd35-42e3-a4e1-4fc6228631d0
# ╠═e2b81096-20c2-11ed-19f2-619ba8119365
# ╠═1ab5179f-bacd-4ac0-9367-442f76ebb547
# ╠═5c4b5dcf-47df-44c9-845c-e59c3abc9736
# ╟─357169f8-b454-4a78-84ec-296a74a84ca2
# ╟─c230eae7-b74e-4706-84e8-285097d69fcb
# ╠═256d0d02-295d-49a8-b4e7-c3f6969b3ca8
# ╟─3014b737-389d-4db5-91d4-1825301b8a8d
# ╠═5607b546-ba26-45cf-8932-b7fe7b7254d7
# ╠═952b55c3-5813-408d-904d-8874ee51f5b5
# ╟─2ee18e86-3c6c-4041-9cea-6e1f1c4df777
# ╠═5d9c579a-44bb-437f-8d7d-7efcd63744ea
# ╟─4af4c093-4bf6-4bea-8515-c4c76b688ba4
# ╠═6c6d805a-74e0-4a8e-b8a7-41ccacf3ccfc
# ╟─b548746b-8f48-4c25-b59f-5c3d1963bc29
# ╠═771ea5b1-8af4-4ad9-88ea-3abae631669f
# ╠═1bedbcb8-52c1-4752-9379-b383252a1b0a
# ╟─7a95f0e0-1976-4df0-8211-db1f54382414
# ╠═8211824f-b524-4f5f-98e9-a51ca08dbed8
# ╟─ccfa2bf2-7685-4960-a0e3-50614866c77e
# ╟─e9048ab3-db7f-4bbf-ba29-74429fdbce94
# ╠═abac878f-f78f-49bb-8635-9c6a9e291041
# ╟─c3775ef5-a8f1-42ca-81e0-7197554d68e8
# ╠═3d7a671f-5e45-4d61-945b-94eeece85bfb
# ╟─7916fe7e-224c-4850-8cea-66e6db68b85c
# ╟─0495e77e-7e25-4039-a9a1-d8165c9e8e43
# ╠═0883c9d3-1314-487a-ab60-50e78a7d5e8f
# ╟─a1fff200-501e-4bdd-af50-973210a6a071
# ╟─1e83aa95-ccc9-41b4-88c3-b645ef02aef0
# ╠═de7bacb6-ed1f-457d-8d2f-cabd1a1e9ce6
# ╟─86e3332b-81f7-464b-a3c7-ff05a68525e3
# ╠═dada0984-8288-4cd9-8b21-ad9adbb7e4ad
# ╟─604430c0-0dd4-4d36-bb37-5305944cd913
# ╠═c9c8634d-c3c6-494d-bd77-4f6512dc228c
# ╟─75d07949-9106-4cba-a7e3-b748cb27270a
# ╠═a8bc81a8-d8f6-4e0c-91ab-d3422ff6860e
# ╠═f758f229-0615-4043-ae5e-b1d4152fd051
# ╟─eda384ac-cd36-46a3-bdd6-dd9be7dd9695
# ╠═7da00331-f3d0-4f01-a011-a6d033dc885a
# ╟─b084e6db-c06b-469f-8725-bc76c0f4328a
# ╠═1b7a257d-2204-4708-af0b-5212e5d9fc9d
# ╟─7055b7aa-dad0-4c37-93f6-47754a521e6b
# ╟─a31a5d5e-392e-4033-920f-1c52e9fb15dc
# ╠═56a4949f-5b23-45f4-931e-2e1ad369f0e2
# ╟─b32965f1-9694-47e4-b7c9-bc01aa382e60
# ╟─ffe15400-9b67-40ff-896c-32403a08bc89
# ╟─5f3507b9-610a-4e21-a846-3db72fdd4dd4
# ╟─a61bb975-0fd3-43dd-83e4-386f9485af5c
# ╟─8ca3c772-f019-4abd-8f93-1e5555dd7739
# ╟─43bc6728-a2b9-4098-854d-5bd91ca79c18
# ╠═ed5dac5d-3e68-45ad-b15a-df213ebcbefe
# ╠═f4f3ae7d-5303-4099-82fa-49c6e5105f7e
# ╠═9b94f864-bf19-407f-a460-c200e337b721
# ╠═d216131a-be52-4a49-96e9-d631ff5829a5
# ╟─f2837b72-7d12-41c0-8707-e66acbdd2014
# ╠═bb9402ab-3ded-430c-a6c6-25ed02f0d72d
# ╠═8a440617-43bf-490d-b1c7-aede074ea3cb
# ╟─c015ba3f-e114-4a17-aa9f-c5e871a25ff1
# ╠═fedd9488-23c9-4d02-ac11-a6311f416d71
# ╠═3b30e859-42ae-42c5-85ea-668f7b160815
# ╟─b48627d4-b693-4b82-bdc8-ac6ac30bc686
# ╠═2f47ee49-46a8-4c8a-87de-d0ff56bb89d6
# ╠═9becc87b-7960-4258-9dc6-fc2364a9fed9
# ╟─6372dc4b-b0e9-4bee-bc57-01df67c83b6f
# ╠═64a25b24-b7c9-4413-89f3-d6afbc148bac
# ╠═76b67b52-520e-4aa3-ae50-6cf5aa14705b
# ╟─92355bbe-d868-479c-857b-50109c9eb09d
# ╠═c67a6934-282c-45b2-a50c-cd7fb0e65e93
# ╠═90e1f446-f9e2-4ef3-a041-1066e126d25c
# ╟─7d9a5f1c-79ce-4cfe-bce0-48fb4aab5523
# ╠═c3a14e4c-a882-407c-9734-c8af804a4f8c
# ╟─93a3bbd8-f27b-408c-b420-575947b87d3d
# ╠═dd6cf278-fb91-466c-b22f-c3974db3232b
# ╟─bba73dab-15d8-4a81-b616-1db21ca3aa02
# ╠═a9eddef4-70fe-44e2-827d-85d58f10e1fe
# ╟─f6ca2d28-6c27-45a9-b573-28a5d41f4283
# ╟─305d2404-57d1-4465-9397-0880e4ea1e4d
# ╟─52e535bc-acae-4ff5-8df3-6bb4e038b161
# ╠═dad9cb8a-6556-4ec6-bc4d-09e16921a7ca
# ╠═e4011132-7be8-4e98-9c0b-e965521d8a58
# ╟─a1729aed-0266-4c84-a10c-6d36d90dc778
# ╠═516b0165-17f4-4761-b8eb-40bb126ea20a
# ╟─40954f42-e4cc-4ef0-898a-a28e4b5f6131
# ╟─d7be1399-0fe8-47ed-a84c-3bdce4a87a8c
# ╠═89d035e2-174e-41a0-8760-8d77aa0f6dba
# ╠═b086adea-46cc-49d8-9ec2-cdbcb32012c9
# ╠═c80fb416-ad1f-4944-94b7-d9d2eb85a892
# ╟─7093ab66-ff9c-4abc-ac4c-e417f0577e75
# ╠═d8099871-1586-4c2f-9ce5-0e7f4f2f66df
# ╟─fe5e7d42-6c62-42c6-afdf-edb6eec13b72
# ╟─770da26b-9e6c-43ff-8b4b-6cce3ab335da
# ╠═9743c4b9-4f1e-4889-b01e-a5a8fdb5a06d
# ╠═c5191dee-3169-4632-a4cb-c99e509016c2
# ╟─e5350032-3390-4b9b-9f92-14a18fef40b2
# ╠═b4092668-2fb0-426a-87fa-16c0f14986bb
# ╟─14135f52-7e97-4826-addb-ae5590ab1dcc
# ╠═4c5c4b2e-26d0-44e8-ad21-6fbd7bdc50e4
# ╟─28feb904-0f10-4ccf-9b28-75e9354a04b1
# ╠═93cf7f8a-d03d-404e-9add-6eb310e37fae
# ╟─d350da96-c029-4820-9663-63caf44b834e
# ╠═3cf0ee1e-fb77-48e0-8f06-35e9ea33fa3d
# ╟─2760277d-67c2-490f-9ed4-d7b7ee59c964
# ╠═540da5db-c41e-4474-8cd9-0bd74229bdd6
# ╠═154fda0a-2727-4651-b920-58a7dfb74a88
# ╠═63ded2d8-882c-439e-bb72-70afb57b4dee
# ╠═76ef93e6-9bad-440e-bd77-dcb4d03969b5
# ╟─b68f7249-055c-44a1-8941-092c6b895c60
# ╟─e58c7064-b0fa-4df7-bab6-401e876e9320
# ╠═b29680b3-34b4-423f-be2b-85d7b919f711
# ╟─2c0701ac-b050-4b6a-99f1-5b1901ea72be
# ╟─9358644a-5e41-4f28-b7a4-a629c124a2e9
# ╠═2810765c-c1a7-455e-87a2-08fa28556f15
# ╟─f1005bbc-4475-4c0d-8b01-f3ca83c4f6ba
# ╠═462ffdda-6fea-40f3-a6b5-d599db8794cf
# ╠═bd7ca099-a7b1-409f-b3d9-c3449d153520
# ╟─8a036196-0b1f-46b6-a9ad-1e6050b195d2
# ╠═57d8092c-e527-4645-a616-facb39d0fc23
# ╟─cddc9ad7-10e1-4a9b-9bbb-0dabfe92bdb9
# ╟─010daeea-670a-45b2-8412-7bb12b308d0d
# ╟─7d453f3f-249e-4aab-ad8e-63890953f3e5
# ╠═ce9f013d-d40c-4f4f-a8b1-e9612cc30df4
# ╟─50ab2c87-6d62-4ea5-9977-b4cc4f0a75bb
# ╠═78612833-d9e5-4cbf-9e37-b343fce03f15
# ╠═4ca3b4de-225d-4ce3-ba10-587f48f882c0
# ╟─a3a8f86c-2ddd-4e01-ad66-3cf30dafc305
# ╟─59623ea5-9878-44b3-af5a-4ae83d8e9586
# ╠═e6dcbaff-5231-43e7-97c1-11ea58d4a9ac
# ╟─9b0fa6f9-b173-48a2-ada2-756eeaaa66b5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
