using BlockArrays: Block

# ====================================  Definitions  =======================================
struct CartesianProduct{A,B}
  a::A
  b::B

  CartesianProduct{A,B}(a, b) where {A,B} = new{A,B}(a, b)
end

# =====================================  Accessors  ========================================

arguments(cp::CartesianProduct) = (cp.a, cp.b)

# ====================================  Constructors  ======================================

cartesianproduct(a, b) = CartesianProduct{typeof(a),typeof(b)}(a, b)

# ==================================  Base interface  ======================================

function Base.getindex(cp1::CartesianProduct, cp2::CartesianProduct)
  cpa1, cpb1 = arguments(cp1)
  cpa2, cpb2 = arguments(cp2)
  return cartesianproduct(cpa1[cpa2], cpb1[cpb2])
end

Base.length(cp::CartesianProduct) = prod(length, arguments(cp))
