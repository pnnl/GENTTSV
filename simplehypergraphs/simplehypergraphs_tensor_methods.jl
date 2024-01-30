using Random
using LinearAlgebra
using SparseArrays
using SciPy
using PyCall
using Arpack
using DSP
using SimpleHypergraphs
using Statistics
using Test

"""
map values of a dictionary dict to integers starting at 0
"""
function get_val(dict, key)
    if !haskey(dict, key)
        n = length(dict) + 1
        dict[key] = n
        return n
    end
    return dict[key]
end
"""
Compute the Banerjee normalization coefficient alpha
"""
COEF(d, r) = sum(((-1) ^ j) * binomial(d, j)*(d - j)^r for j in 0:d)
"""
Create hyperedge to nodes and node to hyperedges dicts
"""
function get_dicts(h::Hypergraph)
    # edge to array of nodes dict
    H = Dict{Int64, Array{Int64}}()
    # node to array of edges dict
    E = Dict{Int64, Array{Int64}}()
    for incidence in findall(h .!== nothing)
        edge = incidence[2]
        node = incidence[1]
        if !haskey(H, edge)
            H[edge] = [node]
        else
            append!(H[edge], node)
        end
    end
    for (edge, nodes) in H
        for node in nodes
            if !haskey(E, node)
                E[node] = [edge]
            else
                append!(E[node], edge)
            end
        end
    end
    return E, H
end

function get_clique(E, H)
    n = length(E)
    W = spzeros(n, n)
    for (_, nodes) in H
        for node1 in nodes
            for node2 in nodes
                if node1 != node2
                    W[node1, node2] += 1
                end
            end
        end
    end
    return W

end       
"""
Get dictionary where keys are node pairs and values are arrays of hyperedge indices they appear in
"""
function pairwise_incidence(H::Dict{Int64,Array{Int64}}, r::Int64)
    EE = Dict{Tuple{Int64,Int64},Array{Int64}}()
    for (e, edge) in H
        l = length(edge)
        for i in 1:l - 1
            for j in i + 1:l
                if !haskey(EE, (edge[i], edge[j]))
                    EE[(edge[i], edge[j])] = [e]
                else
                    append!(EE[(edge[i], edge[j])], e)
                end
            end
        end
        if l < r
            for node in edge
                if !haskey(EE, (node, node))
                    EE[(node, node)] = [e]
                else
                    append!(EE[(node, node)], e)
                end
            end
        end
    end
    return EE
end

using SpecialFunctions

"""
Return the Banerjee alpha coefficient.
Parameters:
    l: length of given hyperedge
    r: maximum hyperedge size
Returns:
    alpha: the Banerjee coefficient
"""
function COEF(l, r)
    return sum((-1)^j * binomial(l, j) * (l - j)^r for j in 0:l)
end

"""
Evaluate the generating function using subset expansion approach
"""
function get_gen_coef_subset_expansion(edge_values::Vector{Float64}, node_value::Float64, r::Int64)
    k = length(edge_values)
    subset_vector = [0.0]
    subset_lengths = [0]
    for i in 1:k
        for t in 1:length(subset_vector)
            push!(subset_vector, subset_vector[t] + edge_values[i])
            push!(subset_lengths, subset_lengths[t] + 1)
        end
    end
    for i in 1:length(subset_lengths)
        subset_lengths[i] = (-1)^(k - subset_lengths[i])
    end
    total = sum((node_value + subset_vector[i])^r * subset_lengths[i] for i in 1:length(subset_lengths))
    return total / factorial(r)
end

"""
Evaluate the generating function using FFT approach
"""
function get_gen_coef_fft_fast_array(edge_without_node::Vector{Int64}, a::Vector{Float64}, node::Int64, l::Int64, r::Int64)
    coefs = [1.0]
    for i in 1:r-1
        push!(coefs, coefs[end] * a[node] / i)
    end
    coefs = collect(coefs)
    for u in edge_without_node
        _coefs = [1.0]
        for i in 1:r-l+1
            push!(_coefs, _coefs[end] * a[u] / i)
        end
        _coefs[1] = 0
        coefs = conv(_coefs, coefs)[1:r]
    end
    gen_fun_coef = coefs[end]
    return gen_fun_coef
end

"""
Compute tensor times same vector in all but one for the hypergraph h using generating functions
Parameters:
    h: hypergraph
    r: maximum hyperedge size
    a: array encoding the vector by which we wish to multiply the tensor
Returns:
    s: the tensor times same vector in all but one
"""
function TTSV1(h::Hypergraph, r::Int64, a::Array{Float64})
    E, H = get_dicts(h)
    s = _TTSV1_kernel(E, H, r, a)
    return s
end

"""
* Fast function for iteration *
Compute tensor times same vector in all but one for a hypergraph encoded by E and H using generating functions
Parameters:
    E: dictionary with nodes as keys and the hyperedges they appear in as values
    H: dictionary with edges as keys and the nodes that appear in them as values
    r: maximum hyperedge size
    a: array encoding the vector by which we wish to multiply the tensor
Returns:
    s: the tensor times same vector in all but one
"""
function _TTSV1(E::Dict{Int64,Array{Int64}}, H::Dict{Int64,Array{Int64}}, r::Int64, a::Array{Float64})
    s = _TTSV1_kernel(E, H, r, a)
    return s
end

"""
Fast kernel for TTSV1
"""
function _TTSV1_kernel(E::Dict{Int64,Array{Int64}}, H::Dict{Int64,Array{Int64}}, r::Int64, a::Array{Float64})
    n = length(E)
    s = zeros(n)
    r_minus_1_factorial = factorial(r-1)
    for (node, edges) in pairs(E)
        c = 0
        for e in edges
            l = length(H[e])
            alpha = COEF(l, r)
            edge_without_node = [v for v in H[e] if v != node]
            if l == r
                gen_fun_coef = prod(a[edge_without_node])
            elseif 2^(l - 1) < r * (l - 1)
                gen_fun_coef = get_gen_coef_subset_expansion(a[edge_without_node], a[node], r - 1)
            else
                gen_fun_coef = get_gen_coef_fft_fast_array(edge_without_node, a, node, l, r)
            end
            c += r_minus_1_factorial * l * gen_fun_coef / alpha
        end
        s[node] = c
    end
    return s
end

"""
Compute tensor times same vector in all but two for a hypergraph h using generating functions
Parameters:
    h: hypergraph
    r: maximum hyperedge size
    a: numpy array encoding the vector by which we wish to multiply the tensor
    n: number of nodes
Returns:
    Y: a sparse array representing the output of TTSV2
"""
function TTSV2(h::Hypergraph, r::Int64, a::Array{Float64})
    _, H = get_dicts(h)
    E = pairwise_incidence(H, r)
    n = nhv(h)
    Y = _TTSV2_kernel(E, H, r, a, n)
    return Y
end

"""
* Fast function for iteration *
Compute tensor times same vector in all but two for a hypergraph encoded by E and H using generating functions
Parameters:
    E: dictionary with nodes pairs as keys and the hyperedges they appear in as values
    H: dictionary with edges as keys and the nodes that appear in them as values
    r: maximum hyperedge size
    a: array encoding the vector by which we wish to multiply the tensor
    n: number of nodes
Returns:
    Y: a sparse array representing the output of TTSV2
"""
function _TTSV2(E::Dict{Tuple{Int64, Int64}, Array{Int64}}, H::Dict{Int64, Array{Int64}}, r::Int64, a::Array{Float64}, n::Int64)
    Y = _TTSV2_kernel(E, H, r, a, n)
    return Y
end

""" 
Fast kernel for TTSV2
"""
function _TTSV2_kernel(E::Dict{Tuple{Int64, Int64}, Array{Int64}}, H::Dict{Int64, Array{Int64}}, r::Int64, a::Array{Float64}, n::Int64)
    s = Dict{Tuple{Int64, Int64}, Float64}()
    r_minus_2_factorial = factorial(r - 2)
    for (nodes, edges) in pairs(E)
        v1 = nodes[1]
        v2 = nodes[2]
        c = 0
        
        for e in edges
            l = length(H[e])
            alpha = COEF(l, r)
            edge_without_node = [v for v in H[e] if v != v1 && v != v2]
            
            if v1 != v2
                if 2^(l - 2) < (r - 2) * (l - 2)
                    gen_fun_coef = get_gen_coef_subset_expansion(a[edge_without_node], a[v1] + a[v2], r - 2)
                else
                    coefs = [1.0]
                    for i in 1:r - 2
                        push!(coefs, coefs[end] * (a[v1] + a[v2]) / i)
                    end
                    coefs = collect(coefs)
                    
                    for u in H[e]
                        if u != v1 && u != v2
                            _coefs = [1]
                            for i in 1:r - l + 1
                                push!(_coefs, _coefs[end] * a[u] / i)
                            end
                            _coefs[1] = 0
                            coefs = conv(_coefs, coefs)[1:r - 1]
                        end
                    end
                    
                    gen_fun_coef = coefs[end]
                end
            else
                if 2^(l - 1) < (r - 2) * (l - 1)
                    gen_fun_coef = get_gen_coef_subset_expansion(a[edge_without_node], a[v1], r - 2)
                else
                    coefs = [1.0]
                    for i in 1:r - 2
                        push!(coefs, coefs[end] * (a[v1]) / i)
                    end
                    coefs = collect(coefs)
                    
                    for u in H[e]
                        if u != v1 && u != v2
                            _coefs = [1]
                            for i in 1:r - l + 1
                                push!(_coefs, _coefs[end] * a[v1] / i)
                            end
                            _coefs[1] = 0
                            coefs = conv(_coefs, coefs)[1:r - 1]
                        end
                    end
                    
                    gen_fun_coef = coefs[end]
                end
            end
            
            c += r_minus_2_factorial * l * gen_fun_coef / alpha
        end
        
        s[nodes] = c
        
        if v1 == v2
            s[nodes] /= 2
        end
    end
    
    first = Int[]
    second = Int[]
    value = Float64[]
    
    for (k, v) in pairs(s)
        push!(first, k[1])
        push!(second, k[2])
        push!(value, v)
    end
    
    Y = SparseArrays.sparse(first, second, value, n, n)
    return Y + transpose(Y)
end


@testset " tensor times same vector " begin
    h = Hypergraph{Float64}(9,4)
    h[1:3,1] .= 1.5
    h[2:4,2] .= 1
    h[5,3] = 1
    h[4,4] = 1
    h[6:9,4] .= 1
    H = Dict(1 => [1, 2, 3], 2 => [2, 3, 4], 3 => [5], 4 => [4, 6, 7, 8, 9])
    E = Dict(1 => [1], 2 => [1,2], 3 => [1,2], 4 => [2, 4], 5 => [3], 6 => [4], 7 => [4], 8 => [4], 9 => [4])
    r = 5
    a = 1.0 * [1, 2, 3, 4, 5, 6, 7, 8, 9]
    s =    [19.2, 105.24, 82.24, 3086.4, 625.0, 2016.0, 1728.0, 1512.0, 1344.0]
    s_new = TTSV1(h, r, a)
    @test norm(s - s_new) < 1e-13
    Y = [2.52   3.78   3.04    0.0     0.0    0.0    0.0   0.0   0.0;
    3.78   9.36  13.9    10.26    0.0    0.0    0.0   0.0   0.0;
    3.04  13.9    6.84    7.72    0.0    0.0    0.0   0.0   0.0;
    0.0   10.26   7.72    4.68    0.0  126.0  108.0  94.5  84.0;
    0.0    0.0    0.0     0.0   125.0    0.0    0.0   0.0   0.0;
    0.0    0.0    0.0   126.0     0.0    0.0   72.0  63.0  56.0;
    0.0    0.0    0.0   108.0     0.0   72.0    0.0  54.0  48.0;
    0.0    0.0    0.0    94.5     0.0   63.0   54.0   0.0  42.0;
    0.0    0.0    0.0    84.0     0.0   56.0   48.0  42.0   0.0]
    Y_new = TTSV2(h, r, a)
    @test norm(Y - Y_new) < 1e-13
end

"""
Compute largest real eigenvector
"""
function LR_evec(A::SparseMatrixCSC)
    evec = eigs(A, nev=1, which=:LR, tol=1e-5, maxiter=200)[2][:,1]
    if evec[1] < 0; evec = -evec; end
    return evec / norm(evec, 1)
end

"""
Fast Z-eigenvecxtor centrality kernel
"""
function _Z_evec_dynsys(E::Dict{Tuple{Int64, Int64}, Array{Int64}}, H::Dict{Int64, Array{Int64}}, r::Int64, n::Int64, niter::Int64=200, tol::Float64=1e-5)
    x_init=ones(n)/n
    f(u::Vector{Float64}) = LR_evec(_TTSV2(E,H,r,u,n)) - u
    x_curr = copy(x_init)
    h = 0.5
    converged = false
    for i = 1:niter
        print("$i of $niter \r")
        flush(stdout)
        x_next = x_curr + h * f(x_curr)
        s = x_next ./ x_curr
        converged = (maximum(s) - minimum(s)) / minimum(s) < tol
        if converged; break; end
        x_curr = x_next
    end
    evec = x_curr
    return (evec, converged)
end
"""
Fast H-tensor eigenvector centrality kernel
"""
function _H_evec_NQI(E::Dict{Int64,Array{Int64}}, H::Dict{Int64,Array{Int64}}, r::Int64, niter::Int64=200, tol::Float64=1e-5)
    m =r
    n=length(E)
    converged = false
    x = ones(Float64, n)/n
    y = _TTSV1(E,H,r,x)
    for i in 1:niter
        print("$i of $niter \r")
        flush(stdout)
        y_scaled = abs.(y) .^ (1.0 / (m - 1))
        x = y_scaled / norm(y_scaled, 1)
        y =_TTSV1(E,H,r,x)
        s = y ./ (x .^ (m - 1))
        converged = (maximum(s) - minimum(s)) / minimum(s) < tol
        # print(x[1:10])
        flush(stdout)

        if converged; break; end
    end
    return (x, converged)
end

"""
Compute the clique eigenvector centrality of hypergraph h

Returns a tuple where the first argument is the centrality vector and the second is a boolean indicating the algorithm's convergence
Note: the algorithm is guaranteed to converge as long as the hypergraph is connected
"""
function CEC(h::Hypergraph)
    @assert length(get_connected_components(h)) == 1
    E, H = get_dicts(h)
    A = get_clique(E, H)
    c = LR_evec(A)
    return (c / norm(c, 1), true)
end

"""
Compute the Z-tensor eigenvector centrality of hypergraph h with maximum hyperedge size r

Returns a tuple where the first argument is the centrality vector and the second is a boolean indicating the algorithm's convergence
"""
function ZEC(h::Hypergraph, r::Int64, niter::Int64=200, tol::Float64=1e-5)
    @assert length(get_connected_components(h)) == 1
    n = nhv(h)
    _, H = get_dicts(h)
    EE = pairwise_incidence(H, r)
    (c, converged) = _Z_evec_dynsys(EE,H,r,n,niter,tol)
    return (c / norm(c, 1), converged)
end

"""
Compute the Z-tensor eigenvector centrality of hypergraph h with maximum hyperedge size r

Returns a tuple where the first argument is the centrality vector and the second is a boolean indicating the algorithm's convergence
"""
function HEC(h::Hypergraph, r::Int64, niter::Int64=200, tol::Float64=1e-5)
    @assert length(get_connected_components(h)) == 1
    E, H = get_dicts(h)
    c, converged = _H_evec_NQI(E, H, r, niter, tol)
    return (c / norm(c, 1), converged)
end
@testset "SimpleHypergraphs hypergraph tensor eigenvector centralities" begin
    h = Hypergraph{Float64}(8,4)
    h[1:3,1] .= 1.5
    h[2:4,2] .= 1
    h[4,3] = 1
    h[4,4] = 1
    h[5:8,4] .= 1
    H = Dict(1 => [1, 2, 3], 2 => [2, 3, 4], 3 => [5], 4 => [4, 6, 7, 8, 9])
    E = Dict(1 => [1], 2 => [1,2], 3 => [1,2], 4 => [2, 4], 5 => [3], 6 => [4], 7 => [4], 8 => [4], 9 => [4])
    r = 5
    cec_old = [0.047127880702636626, 0.10055993491379979, 0.1005599349137999, 0.18089526880786416, 0.14271424516547485, 0.14271424516547493, 0.14271424516547485, 0.14271424516547487]
    zec_old = [9.398837134614217e-6, 7.388052440867785e-5, 7.388052440867382e-5, 0.999817461021228, 6.344773205063297e-6, 6.34477320503e-6, 6.344773204977544e-6, 6.344773204996967e-6]
    hec_old = [0.13145014699863966, 0.16094102703046673, 0.16094102703046673, 0.17414543947771208, 0.09313058986567864, 0.09313058986567864, 0.09313058986567864, 0.09313058986567864]
    cec, cec_conv = CEC(h)
    zec, zec_conv = ZEC(h, 5)
    hec, hec_conv = HEC(h, 5)
    @test norm(cec_old - cec) < 1e-13
    # we cannot write a unit test for zec because we are not guaranteed uniqueness for the centrality vector
    # @test norm(zec_old - zec) < 1e-13
    @test norm(hec_old - hec) < 1e-13
end