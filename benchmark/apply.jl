using Revise
using ITensors
using ITensorMPS
using LinearAlgebra
using BenchmarkTools
using TensorOperations
using Tullio

sites = siteinds(2, 10)

a = sum(random_mpo(sites) for _ in 1:5)
b = random_mps(sites; linkdims=100)

densitymatrix = apply(a, b; alg="densitymatrix")
@benchmark apply(a, b; alg="densitymatrix", maxdim = 100)

@profview src = apply(a, b; alg="src")
@benchmark apply(a, b; alg="src", maxdim = 100, oversample=20)

i = Index(100)
j = Index(10)
k = Index(100)
l = Index(100)
m = Index(100)
n = Index(2)
o = Index(2)
p = Index(10)
q = Index(100)

A = random_itensor(k, o, q)
B = random_itensor(i, j, k)
C = random_itensor(j, n, o, p) 
D = delta(i, l, m)
E = random_itensor(m, n)

@btime C * A * E * B * D #current

ITensors.enable_contraction_sequence_optimization()
ITensors.using_contraction_sequence_optimization()
@btime *(C, A, E, B, D) #TensorOperations optimized
ITensors.optimal_contraction_sequence([A, B, C, D, E])
ITensors.disable_contraction_sequence_optimization()

Aa = array(A)
Ba = array(B)
Ca = array(C)
Da = array(D)
Ea = array(E)

@btime @tullio out[l, p, q] := Aa[k, o, q] * Ba[l, j, k] * Ca[j, n, o, p] * Ea[l, n]

@benchmark @tensor begin
    out[l, p, q] := Aa[k, o, q] * Ba[i, j, k] * Ca[j, n, o, p] * Ea[m, n] * Da[l, i, m]
end