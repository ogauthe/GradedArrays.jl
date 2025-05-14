using GradedArrays: arguments

# ====================================  Definitions  =======================================

struct KroneckerMatrix{A<:AbstractMatrix,B<:AbstractMatrix,T} <: AbstractMatrix{T}
  a::A
  b::B
  function KroneckerMatrix{A,B,T}(a, b) where {A,B,T}
    return new{A,B,T}(a, b)
  end
end

# ====================================  Constructors  ======================================

function KroneckerMatrix(a, b)
  return KroneckerMatrix{typeof(a),typeof(b),Base.promote_op(*, eltype(a), eltype(b))}(a, b)
end

kroneckermatrix(a, b) = KroneckerMatrix(a, b)

# =====================================  Accessors  ========================================

GradedArrays.arguments(km::KroneckerMatrix) = (km.a, km.b)

# ==================================  Base interface  ======================================

# TBD isdual?
function Base.axes(km::KroneckerMatrix)
  return (
    cartesianproductunitrange(
      cartesianproduct(first.(axes.(arguments(km)))...), 1:size(km, 1), false
    ),
    cartesianproductunitrange(
      cartesianproduct(last.(axes.(arguments(km)))...), 1:size(km, 2), false
    ),
  )
end

function Base.show(io::IO, km::KroneckerMatrix)
  println(io, typeof(km))
  println(io, first(arguments(km)))
  return println(io, last(arguments(km)))
end

# show(io::IO, ::MIME"text/plain", a::AbstractArray) does not default on show(io, A)
function Base.show(io::IO, ::MIME"text/plain", km::KroneckerMatrix)
  println(io, typeof(km))
  println(io, first(arguments(km)))
  return println(io, last(arguments(km)))
end

function Base.size(km::KroneckerMatrix)
  sizes = size.(arguments(km))
  return (prod(first.(sizes)), prod(last.(sizes)))
end

for f in [:+, :-, :*]
  @eval function Base.$f(km1::KroneckerMatrix, km2::KroneckerMatrix)
    return kroneckermatrix($f.(arguments.((km1, km2))...)...)
  end
end

for f in [:-, :adjoint, :copy, :transpose]
  @eval Base.$f(km::KroneckerMatrix) = kroneckermatrix($f.(arguments(km))...)
end
