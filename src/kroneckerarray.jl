using LinearAlgebra: LinearAlgebra, norm, svd

# ==========================================================================================
struct KroneckerArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  identity_dimension::Int  # TBD relax to allow for non-group category?
  # TBD identity_dimension a NTuple{Int} to allow different scaling according to dimension?
  # TBD store tuple of KroneckerRange as axes instead of identity_dimension?
  # TBD only define KroneckerMatrix?
  array::A
  function KroneckerArray{T,N,A}(id_dim, a) where {T,N,A}
    return new{T,N,A}(id_dim, a)
  end
end

const KroneckerVector{T,A} = KroneckerArray{T,1,A}
const KroneckerMatrix{T,A} = KroneckerArray{T,2,A}

# accessors
array(ka::KroneckerArray) = ka.array
identity_dimension(ka::KroneckerArray) = ka.identity_dimension

# constructors
kroneckerarray(id_dim, a) = KroneckerArray{eltype(a),ndims(a),typeof(a)}(id_dim, a)

# Base interface
function Base.show(io::IO, ka::KroneckerArray)
  return print(io, nameof(typeof(ka)), " ", identity_dimension(ka), "×", size(array(ka)))
end

function Base.show(io::IO, ::MIME"text/plain", ka::KroneckerArray)
  println(io, nameof(typeof(ka)), " ", identity_dimension(ka), "×", size(array(ka)))
  return print(io, array(ka))
end

function Base.getindex(ka::KroneckerArray{<:Any,N}, krs::Vararg{KroneckerRange,N}) where {N}
  all(identity_dimension(ka) .== identity_dimension.(krs)) ||
    throw(ArgumentError("identity_dimension must match"))
  return array(ka)[inner_range.(krs)...]
end

Base.iterate(ka::KroneckerArray) = iterate(array(ka))
Base.iterate(ka::KroneckerArray, state) = iterate(array(ka), state)

Base.size(ka::KroneckerArray) = identity_dimension(ka) .* size(array(ka))

for f in [:+, :-, :*]
  @eval function Base.$f(ka1::KroneckerArray, ka2::KroneckerArray)
    identity_dimension(ka1) == identity_dimension(ka2) ||
      throw(ArgumentError("identity_dimension must match"))
    return kroneckerarray(identity_dimension(ka1), $f(array(ka1), array(ka2)))
  end
end

for f in [:-, :adjoint, :copy, :transpose]
  @eval Base.$f(ka::KroneckerArray) = kroneckerarray(identity_dimension(ka), $f(array(ka)))
end

for f in [:maximum, :minimum, :unique, :iszero]
  @eval Base.$f(ka::KroneckerArray) = $f(array(ka))
end

function Base.similar(ka::KroneckerArray, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
  return kroneckerarray(
    identity_dimension(ka), similar(array(ka), T, dims .÷ identity_dimension(ka))
  )
end

# used in DerivableInterfaces.zero!
Base.fill!(ka::KroneckerArray, x) = fill!(array(ka), x)

# needed in
# ~/Documents/itensor/BlockSparseArrays.jl/src/blocksparsearrayinterface/getunstoredblock.jl:16
function Base.similar(::Type{KroneckerArray{T,N,A}}, dims::NTuple{N,Int}) where {T,N,A}
  #return error("identity_dimension not known")
  return kroneckerarray(1, similar(A, dims))  # crash later all(iszero, ::Tuple{::SubArray{T,N,KroneckerArray})
end

function Base.permutedims(ka::KroneckerArray, args...)
  return kroneckerarray(identity_dimension(ka), permutedims(array(ka), args...))
end

# LinearAlgebra
LinearAlgebra.norm(ka::KroneckerArray) = sqrt(identity_dimension(ka)) * norm(array(ka))

function LinearAlgebra.svd(ka::KroneckerArray)  # showcase
  u, s, v = LinearAlgebra.svd(array(ka))
  return kroneckerarray.(identity_dimension(ka), (u, s, v))
end

function LinearAlgebra.mul!(
  C::KroneckerArray, A::KroneckerArray, B::KroneckerArray, α::Number, β::Number
)
  return mul!(array(C), array(A), array(B), α, β)
end

# KroneckerMatrix only
# getindex(::KroneckeMatrix) = isabelian(m) ? getindex(::AbelianStyle) : error()
