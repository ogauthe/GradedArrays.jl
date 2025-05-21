
# =====================================  Definition  =======================================

struct Flux{S} <: AbstractSector
  sector::S
  isdual::Bool
end

# ====================================  Constructors  ======================================

flux(s, b) = Flux(s, b)

# =====================================  Accessors  ========================================

isdual(f::Flux) = f.isdual
sector(f::Flux) = f.sector

# =============================  GradedUnitRanges interface  ===============================

dual(f::Flux) = flux(sector(f), !isdual(f))

flip(f::Flux) = flux(dual(sector(f)), !isdual(f))

quantum_dimension(f::Flux) = quantum_dimension(sector(f))

sector_type(::Type{<:Flux{S}}) where {S} = sector_type(S)

# =============================  TensorProducts interface  ===============================

function TensorProducts.tensor_product(f1::Flux, f2::Flux)
  return flux(tensor_product(sector(flip_dual(f1)), sector(flip_dual(f2))), false)
end

# ==================================  Base interface  ======================================

Base.length(f::Flux) = quantum_dimension(f)
