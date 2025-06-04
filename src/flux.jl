
# =====================================  Definition  =======================================

struct Flux{S} <: AbstractSector
  sector::S
  isdual::Bool
end

# ====================================  Constructors  ======================================

flux(s::AbstractSector, b::Bool) = Flux(s, b)

# =====================================  Accessors  ========================================

isdual(f::Flux) = f.isdual
sector(f::Flux) = f.sector

# =============================  GradedUnitRanges interface  ===============================

SymmetryStyle(F::Type{<:Flux}) = SymmetryStyle(sector_type(F))

dual(f::Flux) = flux(sector(f), !isdual(f))

flip(f::Flux) = flux(dual(sector(f)), !isdual(f))

quantum_dimension(f::Flux) = quantum_dimension(sector(f))

sector_type(::Type{<:Flux{S}}) where {S} = sector_type(S)

# ==================================  Base interface  ======================================

Base.length(f::Flux) = quantum_dimension(f)
