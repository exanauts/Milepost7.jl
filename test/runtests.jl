using ArgParse
using CUDA
# CUDA.device!(1)
# using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using AMDGPU
# using oneAPI
using MPI
using ProxAL
using Test
using Milepost7

case = "../cases/case118.m"
load = "../cases/case118"
T = 2
K = 1
rhopq = 3e3
rhova = 3e4
proxal_iter = 2

MPI.Init()

@testset "Testing Milepost 7" begin
    maxviol_t = 0.0
    maxviol_t_actual = 0.0
    maxviol_c = 0.0
    maxviol_c_actual = 0.0
    maxviol_d = 0.0
    @testset "Testing on CPU" begin
        info = milepost7(case, load, T, K, 2; profile=false, rhopq=rhopq, rhova=rhova, proxal_iter=proxal_iter)

        @test isapprox(info.maxviol_t[end], 2.4908e-01; atol=1e-5)
        @test isapprox(info.maxviol_t_actual[end], 2.4908e-01; atol=1e-5)
        @test isapprox(info.maxviol_c[end], 6.6515e-01; atol=1e-5)
        @test isapprox(info.maxviol_c_actual[end], 6.6515e-01; atol=1e-5)
        @test isapprox(info.maxviol_d[end], 6.5529e-03; atol=1e-5)
        maxviol_t = info.maxviol_t[end]
        maxviol_t_actual = info.maxviol_t_actual[end]
        maxviol_c = info.maxviol_c[end]
        maxviol_c_actual = info.maxviol_c_actual[end]
        maxviol_d = info.maxviol_d[end]
    end
    @testset "Testing on CUDA" begin
        info = milepost7(case, load, T, K, 3; profile=false, rhopq=rhopq, rhova=rhova, proxal_iter=proxal_iter)

        @test isapprox(info.maxviol_t[end], maxviol_t)
        @test isapprox(info.maxviol_t_actual[end], maxviol_t_actual)
        @test isapprox(info.maxviol_c[end], maxviol_c)
        @test isapprox(info.maxviol_c_actual[end], maxviol_c_actual)
        @test isapprox(info.maxviol_d[end], maxviol_d)
    end
    @testset "Testing on KA" begin
        info = milepost7(case, load, T, K, 4; profile=false, rhopq=rhopq, rhova=rhova, proxal_iter=proxal_iter)

        @test isapprox(info.maxviol_t[end], maxviol_t)
        @test isapprox(info.maxviol_t_actual[end], maxviol_t_actual)
        @test isapprox(info.maxviol_c[end], maxviol_c)
        @test isapprox(info.maxviol_c_actual[end], maxviol_c_actual)
        @test isapprox(info.maxviol_d[end], maxviol_d)
    end
end
