using LinearAlgebra
using Ket
using SparseArrays
using Random
using BlackBoxOptim
using Base: log10
using CSV, DataFrames

# WE FIRST NEED TO SET A CUTOFF!
cutoff=5

function ρH(T::Float64)
    """ Ideal heralding state in Eq. (B8)

    # Arguments 
    - T : Beam splitter transmittance

    # Returns
    - 4x4 density matrix 
    """
    return [
        T      0           0           0;
        0   0.5*(1 - T)  0.5*(1 - T)   0;
        0   0.5*(1 - T)  0.5*(1 - T)   0;
        0      0           0           0
    ]
end

function ρH_bar(T::Float64, ηL::Float64)
    """ Ideal heralding state after passing through 
    a lossy channel in Eq. (B9)
    
    # Arguments
    - ηL : loss in the channel

    # Returns
    - 4x4 density matrix 
    """ 
    diag = 0.5 * ηL * (1 - T)
    diag_00 = (1 - ηL + T * ηL)
    return [
        2 * diag_00   0             0             0;
        0             diag         diag           0;
        0             off           diag          0;
        0             0             0             0
    ]
end

function CreationAnhiliation(cutoff::Int, nbrmode::Int)
    """
    Constructs annihilation, creation operators, and 
    projectors onto the non-vacuum subspace

    # Arguments
    - cutoff  : Local Hilbert space cutoff
    - nbrmode : Number of bosonic modes

    # Returns:
        Arrays of shape (cutoff^nbrmode, cutoff^nbrmode, nbrmode)
    """
    # Define single-mode annihilation and creation operators
    x = sqrt.(1:cutoff-1)
    a = spdiagm(1 => x)  # Lower diagonal: annihilation

    # Projector onto vacuum (|0⟩⟨0|)
    sss = zeros(size(a,1),size(a,2))
    sss[1, 1] = 1.0

    # Initialize output arrays
    biga = Array{Matrix{Float64}}(undef, nbrmode)
    bigad = Array{Matrix{Float64}}(undef, nbrmode)
    bigproj = Array{Matrix{Float64}}(undef, nbrmode)

    for s in 0:nbrmode-1
        # Build identity structure with specific operator at position `s`
        MM = repeat(Diagonal(ones(cutoff)), 1, nbrmode)
        LL = repeat(Diagonal(ones(cutoff)), 1, nbrmode)

        MM[1:cutoff, 1 + s*cutoff : (s+1)*cutoff] = a
        LL[1:cutoff, 1 + s*cutoff : (s+1)*cutoff] = sss

        mat = MM[1:cutoff, 1:cutoff]
        mat2 = LL[1:cutoff, 1:cutoff]

        for i = 1:nbrmode-1
            mat = kron(mat, MM[1:cutoff, 1 + i*cutoff : (i+1)*cutoff])
            mat2 = kron(mat2, LL[1:cutoff, 1 + i*cutoff : (i+1)*cutoff])
        end

        biga[s+1] = mat
        bigad[s+1] = mat'
        bigproj[s+1] =I(cutoff^nbrmode)- mat2
    end

    return biga,bigad,bigproj
end

function CanonicalBasis(v::Vector{Int}, n::Int)
    """
     Generate the canonical (computational) basis for
     an n-mode quantum system

    # Arguments
    - v : Vector representing local basis values (e.g., 1:cutoff)
    - n : Number of modes

    # Returns
    - A matrix of size (length(v)^n) × n
    """
    b=length(v)
    totalsize = b^n
    X =zeros(totalsize, n) 

    for i in 1:n
        rep1 = b^(n - i)
        rep2 = b^(i - 1)
        X[:, i] = repeat(v, inner=rep1, outer=rep2)
    end

    return X .- 1  # Convert from 1-based to 0-based indexing
end

function BuildState(v::Vector{Int}, cutoff::Int, nbrmode::Int)
    """
    Builds a quantum state vector corresponding to a basis element

    # Arguments
    - v      : Basis vector in terms of occupation numbers
    - cutoff : Local Hilbert space cutoff
    - nbrmode: Number of modes

    # Returns
    - A state vector (ket) in canonical basis
    """
    # Get the canonical basis
    canbasis = CanonicalBasis(collect(1:cutoff), nbrmode)
    # Initialize state vector with zeros
    state = zeros(Float64, size(canbasis, 1))
    # Find the position of v in the canonical basis and set that
    # entry to 1
    k = 1
    while v != canbasis[k, :]
        k += 1
    end
    state[k] = 1.0

    return state
end

# We predefine the operators for a given number
# of modes to make the code faster when increasing
# the cutoff
a_4, ad_4 = CreationAnhiliation(cutoff, 4);
a_2, ad_2 = CreationAnhiliation(cutoff, 2);
a_1, ad_1 = CreationAnhiliation(cutoff, 1);

function CharlieMeasurement(ηD_C::Float64, cutoff::Int)
    """
    Generates the POVM element corresponding to a detection
    event at Charlie's station

    # Arguments
    - ηD_C   : Detector efficiency
    - cutoff : Local Hilbert space cutoff

    # Returns
    - POVM element matrix
    """

    # Detection left detector with imperfect efficiency
    oper = kron((1 -ηD_C)^(ad_1[1] * a_1[1]), I(cutoff) - (1 -ηD_C)^(ad_1[1] * a_1[1])) 
    
    φ = 0.0; θ = π/2.0
    # Apply phase shift and beamsplitter 
    S = exp(-1im*φ*ad_2[1]*a_2[1])*exp(0.5*θ*(ad_2[1]*a_2[2]-ad_2[2] * a_2[1])) 

    return S*oper*S'
end
##

function SinglePhotonSPDC(ξ::Float64, ϕ::Float64, η::Float64, 
                          cutoff::Int)
    """
    Generates the density matrix corresponding to a heralded 
    state produced by an SPDC source, where one mode is heralded 
    using a single-photon detector

    # Arguments
    - ξ      : Squeezing of the nonlinear crystal
    - ϕ      : Phase of the squeezing operator
    - η      : Detector efficiency of the heralding detector
    - cutoff : Local Hilbert space cutoff 

    # Returns
    - Reduced density matrix
    """

    # Apply a two-mode squeezing operation to the vacuum
    S = exp((exp(-1im*ϕ)*a_2[1]*a_2[2] - exp(1im *ϕ) *ad_2[1]*ad_2[2])*ξ/2)

    # Build initial vacuum state |0,0⟩
    state = S * BuildState([0, 0], cutoff, 2)

    # Construct the corresponding density matrix
    ρ = state * state'

    # Define POVM corresponding to a "click" of a single-photon detector
    # considering detector efficiency
    opelose = I(cutoff^2) - (1-η)^(ad_2[2] * a_2[2])

    # Apply the POVM
    projii = opelose* ρ *opelose'
    projii = projii/tr(projii)

    # Trace out the measured mode to obtain the heralded state
    final_state = partial_trace(projii,[2],[cutoff, cutoff])

    return final_state
end

function SinglePhotonQuantumDot(cutoff::Int,g::Float64)
    """
    Generates a single-photon state produced by a quantum dot source

    # Arguments
    - cutoff : Local Hilbert space cutoff
    - g      : Purity-related parameter for quantum dot source.

    # Returns
    - Density matrix
    """
    # Isolating P1 from g = (2 (1 - P1))/(P1 + 2 (1 - P1))^2
    P1 = (2 * g - 1)/g + sqrt(1 - 2 * g)/g;
    ketbra1 = BuildState([1], cutoff, 1)*BuildState([1], cutoff, 1)'
    ketbra2 = BuildState([2], cutoff, 1)*BuildState([2], cutoff, 1)'

    state = P1*ketbra1 + (1-P1)*ketbra2

    return state
end

function HeraldedState(cutoff::Int, ξ::Float64, ηSPDC::Float64,
                       g::Float64 , V::Float64, T::Float64, 
                       ηD_C::Float64,L::Float64,QD::Bool)
    """ 
    Computes the heralded quantum state generated using two 
    single-photon sources modeled as heralded SPDC sources
    or a quantum dot 

    # Arguments
    - cutoff : Local Hilbert space cutoff 
    - ξ      : Squeezing parameter for the SPDC sources
    - ηSPDC  : Detector efficiency SPDC source
    - ηSPDC  : Detector efficiency SPDC source
    - g      : Quantum dot emission parameter 
    - V      : HOM Visibility
    - ηD_C   : Charlie's efficiency
    - L      : Distance (used to model attenuation)
    - QD     : Boolean to toggle between SPDC and quantum dot source

    # Returns
    - The heralded state and heralding rate
    """
    # Compute beam splitter angle from transmittance
    θ = 2*acos(sqrt(1 - T))
    # Two beam splitter transformations: one for Alice, one for Bob
    BS_A = exp(0.5 *θ*(ad_4[1] * a_4[2] - ad_4[2]*a_4[1]))
    BS_B = exp(0.5 *θ*(ad_4[4] * a_4[3] - ad_4[3]*a_4[4]))
    
    if QD==true
        # Single-photon source modeled as a quantum dot
        ρ0 = SinglePhotonQuantumDot(cutoff, g)
    else
        # Single-photon source modeled as a heralded SPDC source
        ρ0 = SinglePhotonSPDC(ξ, 0.0, ηSPDC, cutoff)
    end

    # Vacuum state
    vac = BuildState([0], cutoff, 1)*BuildState([0], cutoff, 1)' 

    # Apply beam splitter to (ρ0 ⊗ vac) ⊗ (vac ⊗ ρ0)
    state = BS_A*BS_B*kron(kron(ρ0, vac),kron(vac,ρ0))*BS_B'*BS_A'

    # Heralding efficiency central station
    ηH = ηD_C * 10^(-0.2*L/20) 
    
    # Apply Charlie's measurement
    heralproj = kron(kron(I(cutoff), CharlieMeasurement(ηH, cutoff) ), I(cutoff))
    projii = heralproj * state * heralproj'
 
    # Trace out Charlie's modes
    Heralded_State = partial_trace(projii, [2, 3],[cutoff,cutoff,cutoff,cutoff])
    Heralded_State = Heralded_State/real(tr(Heralded_State))

    # Apply visibility factor
    Heralded_State[2, cutoff+1] *= sqrt(V)
    Heralded_State[cutoff+1, 2] *= sqrt(V)

    # We define the heralding rate
    if QD
        R_QD = 75e6
        heralding_ps = R_QD * 4T * (1-T)* ηH
    else
        R_SPDC = 1e9
        heralding_ps = R_SPDC * ξ^4 * ηSPDC^2 * 4T* (1-T) * ηH 
    end

    return Heralded_State, heralding_ps
end


function NoisyHeraldedState(cutoff::Int,ξ::Float64, ηSPDC::Float64,
                            g::Float64,V::Float64,T::Float64,
                            ηD_C::Float64, ηL::Float64,darkcount::Int,
                            L::Float64, QD::Bool)
        """
    Adds loss and dark counts to the heralded state.

    # Arguments
    - we keep the same argument as `HeraldedState`, with additional:
    - ηL        : Local transmission efficiency
    - darkcount : Dark count rate

    # Returns
    - Normalized two-mode density matrix including imperfections
    """

    ket11 = BuildState([1, 1], cutoff, 2)*BuildState([1, 1], cutoff, 2)'
    ket00 = BuildState([0, 0], cutoff, 2)*BuildState([0, 0], cutoff, 2)'
    ket10 = BuildState([1, 0], cutoff, 2)*BuildState([1, 0], cutoff, 2)'
    ket01 = BuildState([0, 1], cutoff, 2)*BuildState([0, 1], cutoff, 2)'
    ρH, HeraldingRate = HeraldedState(cutoff, ξ, ηSPDC,g,V, T, ηD_C,L, QD)

    # Compute beam splitter angle from efficiency ηL
    θ = 2*acos(sqrt(ηL))

    # Two beam splitter transformations: one for Alice, one for Bob
    BS_A = exp(0.5 *θ*(ad_4[1] * a_4[3] - ad_4[3]*a_4[1]))
    BS_B = exp(0.5 *θ*(ad_4[2] * a_4[4] - ad_4[4]*a_4[2]))

    # We apply the loss model
    ρH_vac = kron(ρH, ket00)
    ρH_BS = BS_A*BS_B*ρH_vac *BS_B'* BS_A'
    ρH_noisy = partial_trace(ρH_BS, [3, 4],[cutoff,cutoff,cutoff,cutoff])

    # Normalize the state
    ρ_NoNormalized =  HeraldingRate*ρH_noisy + darkcount*(ηL^2*kb11 + ηL*(1-ηL)*(kb10+ kb01) + (1-ηL)^2*kb00)
    ρ = ρ_NoNormalized/real(tr(ρ_NoNormalized))

    return ρ
end

function POVM_NCNC(ηD_AB::Float64,s::Int, X::Vector{Float64})
    """
    Constructs the local POVM element Π_-1 in Eq. (A11)

    # Arguments
    - ηD_AB : Detection efficiency of Alice and Bob
    - s     : Mode index (1 for Alice, 2 for Bob).
    - X     : POVM parameters [ϕ, ξ, α, θ].

    # Returns
    - POVM operator acting in the whole Hilbert space
    """

    ϕ=X[1] ;ξ=X[2]; α=X[3]; θ=X[4]
    Displacement = exp(exp(1im*θ)*α*ad_2[s] -exp(-1im*θ)conj(α)*a_2[s])  
    Squeezing= exp(0.5*ξ*(exp(-1im * ϕ)*a_2[s]^2 - exp(1im * ϕ)*ad_2[s]^2))

    Π = Displacement*Squeezing*((1-ηD_AB)^(ad_2[s]*a_2[s]))*Squeezing'*Displacement'

    return Π
end

function POVM_NCNC_Disp(ηD_AB::Float64,s::Int, X::Vector{Float64})
    """
    Constructs a displacement-only local POVM element 

    # Arguments
    - ηD_AB : Detection efficiency of Alice and Bob
    - s     : Mode index (1 for Alice, 2 for Bob)
    - X     : POVM parameters [α, θ]

    # Returns
    - POVM element acting in the whole Hilbert space
    """

    α=X[1];θ=X[2];
    Displacement = exp( exp(1im * θ)*α*ad_2[s] - exp(-1im * θ)*α*a_2[s] )  

    Π =Displacement*((1-ηD_AB)^(ad_2[s]*a_2[s]))*Displacement'

    return Π
end

function CHSH(ρ, ηD_AB::Float64, X::Vector{Float64}, squeezing::Bool=true)
    """
    Evaluates the CHSH expression given a state and a set of POVMs

    # Arguments
    - ρ         : Two-mode quantum state
    - ηD_AB     : Detection efficiency of Alice and Bob
    - X         : Vector with measurement parameters 
    - squeezing : If true, includes squeezing in the measurement model

    # Returns
    - CHSH Bell value (absolute value)
    """

    # Decide measurement model
    if squeezing == true
        A1 = X[1:4];A2 = X[5:8];B1 =X[9:12] ;B2 = X[13:16]
        ΠA = [POVM_NCNC(ηD_AB,1, A1),POVM_NCNC(ηD_AB,1, A2)]
        ΠB = [POVM_NCNC(ηD_AB,2, B1),POVM_NCNC(ηD_AB,2, B2)]
    else
        A1 = X[1:2]; A2 = X[3:4]; B1 =X[5:6] ; B2 =X[7:8] 
        ΠA = [POVM_NCNC_Disp(ηD_AB,1, A1),POVM_NCNC_Disp(ηD_AB,1, A2)]
        ΠB = [POVM_NCNC_Disp(ηD_AB,2, B1),POVM_NCNC_Disp(ηD_AB,2, B2)]
    end

    # Calculate CHSH
    E = zeros(2,2)
    for x=[1,2]
        for y=[1,2]
            E[x,y] = 1+4*real(tr(ρ*(ΠA[x]*ΠB[y])) )-2*real(tr(ρ*ΠA[x]))-2*real(tr(ρ*ΠB[y]))
        end
    end
    cs = E[1,1]+ E[1,2]+E[2,1]-E[2,2]
    return cs
end
