using GradedArrays: GradedArrays, arguments

# =====================================  Definition  =======================================

struct CartesianProductUnitRange{T,A,B,Range} <: AbstractUnitRange{T}
  cp::CartesianProduct{A,B}
  full_range::Range
  isdual::Bool

  function CartesianProductUnitRange{T,A,B,Range}(cp, r, b) where {T,A,B,Range}
    return new{T,A,B,Range}(cp, r, b)
  end
end

# ====================================  Constructors  ======================================

function CartesianProductUnitRange(cp, r, b)
  A, B = typeof.(arguments(cp))
  return CartesianProductUnitRange{eltype(r),A,B,typeof(r)}(cp, r, b)
end

# =====================================  Accessors  ========================================

cartesianproduct(cpr::CartesianProductUnitRange) = cpr.cp

GradedArrays.isdual(cpr::CartesianProductUnitRange) = cpr.isdual
