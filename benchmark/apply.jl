using Revise
using ITensors
using ITensorMPS
using LinearAlgebra
using BenchmarkTools

sites = siteinds(2, 10)

a = sum(random_mpo(sites) for _ in 1:5)
b = random_mps(sites; linkdims=100)

densitymatrix = apply(a, b; alg="densitymatrix")
@benchmark apply(a, b; alg="densitymatrix", maxdim = 100)


@profview src = apply(a, b; alg="src")
@benchmark apply(a, b; alg="src", maxdim = 100, oversample=20)