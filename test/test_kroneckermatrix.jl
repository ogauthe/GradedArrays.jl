using GradedArrays: SU2
using GradedArrays.KroneckerProducts:
  BlockCartesianRange, CartesianProduct, KroneckerMatrix, cartesianproduct, kroneckermatrix
using Test: @test, @testset

@testset "KroneckerProducts" begin
  @testset "CartesianProduct" begin
    cp = cartesianproduct(SU2(1), 2:3)
    @test cp isa CartesianProduct
    @test length(cp) == 6

    cp1 = cartesianproduct(Block(1), Block(2))
    cp2 = cartesianproduct(1:1, 2:3)
    cp1[cp2] == cartesianproduct(Block(1)[1:1], Block(2)[2:3])
  end

  @testset "BlockCartesianRange" begin
    cp = cartesianproduct(SU2(1), 2:3)
    bcr = Block(1, 1)[cp, cp]
    @test bcr isa BlockCartesianRange
  end

  @testset "KroneckerMatrix" begin
    a = randn(2, 3)
    b = randn(4, 5)
    km = kroneckermatrix(a, b)

    @test km isa KroneckerMatrix
    @test ndims(km) == 2
    @test eltype(km) === Float64
    @test size(km) == (8, 15)

    @test axes(km, 1) == cartesianproductunitrange(
      cartesianproduct(Base.oneto(2), Base.oneto(4)), 1:8, false
    )
    @test axes(km, 1) == cartesianproductunitrange(
      cartesianproduct(Base.oneto(3), Base.oneto(5)), 1:15, false
    )
  end
end
