"""
This Julia script performs optimization of the CHSH value and the asymptotic
quantum key rate for heralded single-photon quantum states, based on either
SPDC or quantum dot sources.

It supports:
- CHSH violation optimization using squeezing-based or displacement-only POVMs.
- Asymptotic key rate computation using the Devetak-Winter formula.
- Key rate optimization over distance (L) or local transmissivity (ηL).

Required support: `State_Measurement_functions.jl`

"""

include("State_Measurement_functions.jl")

function CHSH_bbo(cutoff::Int, ξ::Float64, ηSPDC::Float64, g::Float64, V::Float64, T::Float64, ηD_C::Float64, dark_ps::Int, ηD_AB::Float64, sq::Bool)
    """
    Optimizes CHSH violation for a heralded quantum state over ηL values.

    # Arguments
    - Full state and setup parameters.
    - `sq`: Use squeezed or displacement-only measurements.

    # Returns
    - Tuple: `(list_chsh, list_eta, list_param)` for each ηL.
    """

    if sq 
        lower = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
        upper = [2π,1,1,2π, 2π,1,1,2π, 2π,1,1,2π, 2π,1,1,2π]

        bounds = [(lower[i], upper[i]) for i in 1:length(lower)]

        initial_param = [4.22207288, 0.133828628, 0.358334741, 5.25260075, 
        1.729494649, 0.286134609, 0.541409744, 0.864787961, 
        0.226477811, 0.129038206, 0.348471305, 3.254900426, 
        2.771157308, 0.292551666, 0.5491544, 1.385547778]

    else
        lower = [0,0, 0,0, 0,0, 0,0]
        upper = [1,2π, 1,2π, 1,2π, 1,2π]
    
        bounds = [(lower[i], upper[i]) for i in 1:length(lower)]

        initial_param = [0.358334741, 5.25260075, 
        0.541409744, 0.864787961, 
        0.348471305, 3.254900426, 
        0.5491544, 1.385547778]
    end
    
    
    list_chsh = []; list_param = []; list_eta = []
    L=0.0
    
    for ηL in 1.0:-0.001:0.8
        println("Optimizing for ηL = $ηL...")
        ρ = NoisyHeraldedState(cutoff, ξ, ηSPDC, g,V,T, ηD_C, ηL, dark_ps, L, true)
        
        function objective(x)
            # Enforce bounds explicitly
            x = clamp.(x, first.(bounds), last.(bounds))
            try
                -CHSH(ρ, ηD_AB, x, sq)
            catch e
                return Inf
            end
        end
        
        result = bboptimize(objective, initial_param;
            SearchRange = bounds,
            NumDimensions = length(lower),
            MaxSteps = 50000,
            Method = :adaptive_de_rand_1_bin_radiuslimited,
            TraceMode = :compact,
            PopulationSize = 50,
        )

        best_x = best_candidate(result)
        best_val = max(1e-12, -best_fitness(result))
        initial_param = best_x

        append!(list_chsh, best_val)
        append!(list_eta, ηL)
        append!(list_param, best_x)

        println("  → Max chsh = $(round(best_val, digits=6)) at ηL = $ηL")
    end

    return list_chsh, list_eta, list_param
end

function h(x::Float64)
    """
    Binary entropy function.

    # Arguments
    - `x`: A probability value between 0 and 1.

    # Returns
    - `h(x) = -x log₂ x - (1 - x) log₂ (1 - x)`
    """

    if x == 0.0 || x == 1.0
        return 0.0
    else
        return -x * log2(x) - (1 - x) * log2(1 - x)
    end
end

function probs_x1_y3(ρ, ηD_AB::Float64, X::Vector{Float64}; squeezing::Bool=true)
    """
    Computes probabilities p(a, b | x=1, y=3) using POVM measurements.

    # Arguments
    - `ρ`: Quantum state.
    - `ηD_AB`: Detection efficiency.
    - `X`: POVM parameters.
    - `squeezing`: Whether to include squeezing in the measurement.

    # Returns
    - Dictionary of joint probabilities `Dict{(a,b) => p}`.
    """

    # Prepare POVMs 
    if squeezing
        A1 = X[1:4]; B3 = X[17:20]
        ΠA = POVM_NCNC(ηD_AB, 1, A1)  
        ΠB = POVM_NCNC(ηD_AB, 2, B3)
    else
        A1 = X[1:2]; B3 = X[9:10]
        ΠA = POVM_NCNC_Disp(ηD_AB, 1, A1)
        ΠB = POVM_NCNC_Disp(ηD_AB, 2, B3)
    end

    # Compute p(a,b | x=1, y=3)
    p = Dict{Tuple{Int,Int}, Float64}()
    for a in 0:1, b in 0:1
        PA = a == 0 ? ΠA : I - ΠA
        PB = b == 0 ? ΠB : I - ΠB
        M = PA * PB  
        p[(a, b)] = real(tr(ρ * M))
    end
    return p
end


function all_probs(ρ, ηD_AB::Float64, X::Vector{Float64}; squeezing::Bool=true)
    """
    Computes joint probabilities p(a, b | x, y) for all x ∈ {0,1}, y ∈ {0,1,2,3}.

    # Arguments
    - `ρ`: Quantum state.
    - `ηD_AB`: Detection efficiency.
    - `X`: POVM parameters.
    - `squeezing`: Whether to use squeezing-based POVMs.

    # Returns
    - A dictionary of the form Dict{Tuple{Int,Int,Int,Int}, Float64} mapping (a,b,x,y) → p.
    """

    p_all = Dict{NTuple{4, Int}, Float64}()

    # Define index ranges for POVMs depending on squeezing
    if squeezing
        A_idxs = [(1:4), (5:8)]
        B_idxs = [(9:12), (13:16), (17:20)]
    else
        A_idxs = [(1:2), (3:4)]
        B_idxs = [(5:6), (7:8), (9:10)]
    end

    for (x_idx, A_range) in enumerate(A_idxs)
        x = x_idx - 1
        for (y_idx, B_range) in enumerate(B_idxs)
            y = y_idx - 1

            Aparams = X[A_range]
            Bparams = X[B_range]
            ΠA = squeezing ? POVM_NCNC(ηD_AB, 1, Aparams) : POVM_NCNC_Disp(ηD_AB, 1, Aparams)
            ΠB = squeezing ? POVM_NCNC(ηD_AB, 2, Bparams) : POVM_NCNC_Disp(ηD_AB, 2, Bparams)

            for a in 0:1, b in 0:1
                PA = a == 0 ? ΠA : I - ΠA
                PB = b == 0 ? ΠB : I - ΠB
                M = PA * PB
                p = real(tr(ρ * M))
                p_all[(a, b, x, y)] = p
            end
        end
    end

    return p_all
end


function cond_ent(p::Dict{Tuple{Int,Int}, Float64}; q::Float64 = 0.0)
    """
    Computes the conditional entropy H(A|B) given joint probabilities.

    # Arguments
    - `p`: Dictionary of joint probabilities.
    - `q`: Noisy preprocessing probability.

    # Returns
    - Conditional entropy H(A|B) in bits.
    """

    marg = Dict{Int, Float64}()
    for b in 0:1
        marg[b] = p[(0,b)] + p[(1,b)]
    end

    joint_q = Dict{Tuple{Int,Int}, Float64}()
    joint_q[(0,0)] = (1 - q)*p[(0,0)] + q*p[(1,0)]
    joint_q[(0,1)] = (1 - q)*p[(0,1)] + q*p[(1,1)]
    joint_q[(1,0)] = marg[0] - joint_q[(0,0)]
    joint_q[(1,1)] = marg[1] - joint_q[(0,1)]

    hab = 0.0
    eps = 1e-12
    
    hab=0.0
    for (key, val) in joint_q
        hab -= val * log2(val)
    end

    hb=0.0
    for (key, val) in marg
        hb -= val * log2(val)
    end


    return hab - hb  # H(A|B)
end


function keyrate(ρ, ηD_AB::Float64, X::Vector{Float64}, q::Float64, L, ηL_fix::Bool; squeezing::Bool = true)
    """
    Computes the asymptotic key rate using Devetak-Winter formula.

    # Arguments
    - `ρ`: Quantum state.
    - `ηD_AB`: Detection efficiency.
    - `X`: POVM parameters.
    - `q`: Noisy preprocessing probability.
    - `L`: Distance (affecting heralding).
    - `squeezing`: Use squeezed measurement model.
    - `ηL_fix`: If true, keyrate includes heralding probability.

    # Returns
    - Secret key rate (asymptotic).
    """

    # Compute probabilities
    p = probs_x1_y3(ρ, ηD_AB, X, squeezing=true)

    chshval = CHSH(ρ, ηD_AB, X, squeezing)
    # Compute conditional entropy
    H_A_given_B = cond_ent(p; q=q)

    # Intermediate expressions for domain checks
    arg1 = (chshval / 2)^2 - 1
    arg2 = 1 - q * (1 - q) * (8 - chshval^2)

    # Devetak-Winter terms
    term1 = 1 - h((1 + sqrt(arg1)) / 2)
    term2 = h((1 + sqrt(arg2)) / 2)

    key_L0 = term1 + term2 - H_A_given_B

    if ηL_fix
        if QD
            heralding_rate = 4 * R * T * (1-T) * ηD_C * 10^(-0.2*L/20) 
        else
            heralding_rate = 4 * ξ^4 * ηSPDC^2 * T * (1-T) * ηD_C * 10^(-0.2*L/20) * R
        end
    
    else
        # When computing the asymptotic key rate without distance use
        heralding_rate = 1
    end
    return heralding_rate * key_L0
end



function keyrate_bbo(cutoff::Int, ξ::Float64, ηSPDC::Float64, g::Float64, V::Float64, T::Float64, ηD_C::Float64, dark_ps::Int, ηD_AB::Float64, QD::Bool, ηL_fix::Bool; etaL::Float64 = 1.0)
    """
    Performs numerical optimization of the key rate over measurement parameters 
    using BlackBoxOptim, either as a function of distance (L) or of local loss (ηL).

    # Arguments
    - `cutoff`: Fock space cutoff.
    - `ξ`: SPDC squeezing parameter.
    - `ηSPDC`: Heralding efficiency at the source.
    - `g`: Quantum dot purity parameter (used if `QD == true`).
    - `V`: HOM visibility.
    - `T`: Beamsplitter transmittance.
    - `ηD_C`: Detector efficiency at Charlie.
    - `dark_ps`: Dark count probability.
    - `ηD_AB`: Detector efficiency at Alice/Bob.
    - `QD`: If `true`, simulate quantum dot sources; else SPDC.
    - `ηL_fix`: If `true`, sweep over distance (L); if `false`, sweep over ηL.
    - `etaL`: (Optional) Fixed value of ηL to use when sweeping over L.

    # Returns
    - Tuple `(list_key, list_x, list_param)`:
        - `list_key`: Key rate values
        - `list_x`: Corresponding values of L or ηL
        - `list_param`: Optimized measurement parameters for each point
    """

    lower = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    upper = [2π, 1, 1, 2π, 2π, 1, 1, 2π, 2π, 1, 1, 2π, 2π, 1, 1, 2π, 2π, 1, 1, 2π, 0.35]
    bounds = [(lower[i], upper[i]) for i in 1:length(lower)]

    # Params for etaL = 1
    if QD 
        #QD
        initial_param = [4.22207288, 0.133828628, 0.358334741, 5.25260075, 
        1.729494649, 0.286134609, 0.541409744, 0.864787961, 
        0.226477811, 0.129038206, 0.348471305, 3.254900426, 
        2.771157308, 0.292551666, 0.5491544, 1.385547778, 
        4.222087437, 0.134865863, 0.361269319, 2.111025306, 
        0.0063853]
    else 
        # SPDC
        initial_param = [4.420544498, 0.220729993, 0.446485874, 5.35186487, 
        1.562530193, 0.363496538, 0.551815552, 0.781265101, 
        9.17e-08, 0.217256943, 0.44157221, 3.141592561, 
        2.878549044, 0.370376193, 0.555959852, 1.439274414, 
        4.420544638, 0.221666575, 0.448261893, 2.210272224, 
        0.001059851]
    end
    
    list_key = []; list_x = []; list_param = []; 
    list_probs = Vector{Dict{NTuple{4, Int}, Float64}}()  # empty vector to store all dicts

    if ηL_fix
        ηL=etaL

        L_min = 1
        L_max = 500
        n_points = 40        # Number of points
        L_values = 10 .^ range(log10(L_min), log10(L_max), length=n_points)
        for L in L_values
            
            println("Optimizing for L = $L...")
            ρ = NoisyHeraldedState(cutoff, ξ, ηSPDC, g,V,T, ηD_C, ηL, dark_ps, L, QD)

            # Define objective function        
            function objective(x)
                # Enforce bounds explicitly
                x = clamp.(x, first.(bounds), last.(bounds))
                try
                    -keyrate(ρ, ηD_AB, x[1:end-1], x[end], L, ηL_fix, squeezing=true)
                catch e
                    return Inf                                             # Penalize invalid parameter sets during optimization
                end
            end
        
            result = bboptimize(objective, initial_param;
                SearchRange = bounds,
                NumDimensions = length(lower),
                MaxSteps = 50000,
                Method = :adaptive_de_rand_1_bin_radiuslimited,
                TraceMode = :compact,
                PopulationSize = 50,
            )

            best_x = best_candidate(result)
            best_val = max(1e-12, -best_fitness(result))                    # Ensure positive key rate
            initial_param = best_x

            p_abxy = all_probs(ρ, ηD_AB, Float64.(best_x[1:end-1]))

            push!(list_probs, p_abxy)  # add the dict as one element

            append!(list_key, best_val)
            append!(list_x, L)
            append!(list_param, best_x)
            println("  → Max key rate = $(round(best_val, digits=6)) at L = $L")
        end
    else

        L = 0.0
        for ηL in 1.0:-0.001:0.94

            println("Optimizing for ηL = $ηL...")
            
            ρ = NoisyHeraldedState(cutoff, ξ, ηSPDC, g,V,T, ηD_C, ηL, dark_ps, L, QD)

            # Define objective function
            function objective(x)
                # Enforce bounds explicitly
                x = clamp.(x, first.(bounds), last.(bounds))
                try
                    -keyrate(ρ, ηD_AB, x[1:end-1], x[end], L, ηL_fix, squeezing=true)
                catch e
                    return Inf
                end
            end
            
            result = bboptimize(objective, initial_param;
                SearchRange = bounds,
                NumDimensions = length(lower),
                MaxSteps = 50000,
                Method = :adaptive_de_rand_1_bin_radiuslimited,
                TraceMode = :compact,
                PopulationSize = 50,
            )

            best_x = best_candidate(result)
            best_val = max(1e-12, -best_fitness(result))
            initial_param = best_x

            p_abxy = all_probs(ρ, ηD_AB, Float64.(best_x[1:end-1]))

            push!(list_probs, p_abxy)  # add the dict as one element

            append!(list_key, best_val)
            append!(list_x, ηL)
            append!(list_param, best_x)
            println("  → Max key rate = $(round(best_val, digits=6)) at ηL = $ηL")
        end
    end

    return list_key, list_x, list_param, list_probs
end



function save_CHSH_results(list_param, list_S, list_eta; sq::Bool, folder::String, file_name::String)
    """
    Saves CHSH optimization results to a CSV file.

    # Arguments
    - `list_param`: Flattened list of optimal POVM parameters.
    - `list_S`: CHSH values.
    - `list_eta`: Detection efficiencies.
    - `sq`: Whether squeezing was used.
    - `folder`: Target folder path.
    - `file_name`: Output CSV file name.
    """

    n_params = sq ? 16 : 8
    param_matrix = reshape(list_param, n_params, length(list_S))'       # Reshape flat list into matrix of POVM parameters (one row per data point)

    df = DataFrame(param_matrix, :auto)
    df.S = list_S
    df.eta = list_eta
    select!(df, Cols(:eta, :S, All()))                                  # Reorder columns: put eta and S first for readability 

    CSV.write(joinpath(folder, file_name), df)
end

using CSV, DataFrames

function save_rate_results(list_param, list_key, list_x;
                           folder::String, file_name::String)

    """
    Saves key rate optimization results and corresponding probabilities to CSV files.

    # Arguments
    - `list_param`: Flattened list of optimal POVM parameters.
    - `list_key`: List of key rate values.
    - `list_x`: List of corresponding L or ηL values.
    - `folder`: Target folder path.
    - `file_name`: Output CSV file name.
    """

    # ---- Save POVM parameters and key rates ----
    param_matrix = reshape(list_param, :, length(list_key))'        # Reshape flat list into matrix of POVM parameters (one row per data point)
    df = DataFrame(param_matrix, :auto)
    df.keyrate = list_key
    df.x_label = list_x
    select!(df, Cols(:x_label, :keyrate, All()))                    # Reorder columns: put x_label and keyrate first for readability
    CSV.write(joinpath(folder, file_name), df)

end

function save_probs(list_L::Vector, list_probs::Vector{Dict{NTuple{4, Int}, Float64}}; folder::String, file_name::String)
    rows = Vector{NamedTuple{(:L, :a, :b, :x, :y, :p_abxy), Tuple{typeof(list_L[1]), Int, Int, Int, Int, Float64}}}()
"""
    Saves probability distributions for each value in list_L to a CSV file.

    # Arguments
    - `list_L`: List of L or ηL values corresponding to each probability distribution.
    - `list_probs`: List of dictionaries mapping (a,b,x,y) tuples to probabilities, one dictionary per L.
    - `folder`: Target folder path.
    - `file_name`: Output CSV file name.

    # Output
    - A CSV file where each row corresponds to a single (L, a, b, x, y, p_abxy) entry.
    """

    # Flatten all probabilities into a table with columns (L, a, b, x, y, p_abxy)
    for (i, dict) in enumerate(list_probs)
        L = list_L[i]
        for ((a,b,x,y), p) in dict
            push!(rows, (L=L, a=a, b=b, x=x, y=y, p_abxy=p))
        end
    end

    df = DataFrame(rows)
    CSV.write(joinpath(folder, file_name), df)
end



# ------------------------------------------------------------------
# MAIN EXECUTION: Run CHSH optimization and key rate computations
# ------------------------------------------------------------------

# Set parameters for the heralded quantum state and detectors

# #______________IDEAL PARAMETERS____________________
# ξ=10^-10; ϕSPDC=0.0; ηSPDC=0.99999; cutoff=5
# T=10^-10; ηD_C = 0.99999;ηL=0.99999; dark_ps=0
# g = 10^-10
# V_QD = 1.0
# V_SPDC = 1.0
#  ηD_AB=0.95                                             # Efficiency of Alice and Bob's detectors             

# ______________REALISTIC PARAMETERS____________________
cutoff=6                                                  # Fock space cutoff
ξ=0.1; ϕSPDC=0; ηSPDC=0.95; R_SPDC = 1e9; V_SPDC = 0.99   # SPDC source params
T=0.01;                                                   # Beamsplitter transmittance
ηD_AB=0.95; ηD_C = 0.95; dark_ps=1                        # Detector efficiencies
R_QD = 75e6; g = 10^-2; V_QD = 0.975                      # QD source params

# Toggle quantum dot or SPDC source
QD = true

# Select visibility and repetition rate depending on source type
if QD
    V = V_QD
    R = R_QD
else
    V = V_SPDC
    R = R_SPDC
end

# Prepare annihilation and creation operators (needed by state generator)
a_4, ad_4 = CreationAnhiliation(cutoff, 4);
a_1, ad_1 = CreationAnhiliation(cutoff, 1);
a_2, ad_2 = CreationAnhiliation(cutoff, 2);


########################################################
###### CHSH optimization (using squeezing or not) ######
########################################################
sq = true
list_S, list_eta, list_param = CHSH_bbo(cutoff, ξ, ηSPDC,g, V, T, ηD_C, dark_ps, ηD_AB, sq)

# Save CHSH results to CSV
folder = "results/"  # Change to your target folder
file_name = sq ? "CHSH_squeezing.csv" : "CHSH_dispacement.csv"
save_CHSH_results(list_param, list_S, list_eta; sq, folder, file_name)



##################################
###### keyrate optimization ######
##################################
# Compute key rate as function of local transmission ηL
eta_fix = false
list_key, list_eta, list_param, list_probs = keyrate_bbo(cutoff, ξ, ηSPDC,g,V, T, ηD_C, dark_ps, ηD_AB, QD, eta_fix)  

T_str = string(round(T, digits=3))
V_str = string(V)
g_str = string(round(g, digits=3))
xi_str = string(round(ξ, digits=3))
folder = "results/"
file_name = "asymptotic_etaL_QD_g$(g_str)_V$(V_str)_T$(T_str).csv"
save_rate_results(list_param, list_key, list_eta; folder, file_name)


# Compute key rate as function of distance L (ηL fixed)
eta_fix = true
eta_val = 0.954
list_key, list_L, list_param, list_probs = keyrate_bbo(cutoff, ξ, ηSPDC,g,V, T, ηD_C, dark_ps, ηD_AB, QD, eta_fix, etaL=eta_val)  

# Save results
T_str = string(round(T, digits=3))
V_str = string(V)
g_str = string(round(g, digits=3))
etaL_str = string(round(eta_val, digits=3))
folder = "results/"
file_name = "asymptotic_L_QD_etaL$(etaL_str)_g$(g_str)_V$(V_str)_T$(T_str).csv"
save_rate_results(list_param, list_key, list_L; folder, file_name)

# Save probabilities for each L value
file_name = "probabilities_L_QD_etaL$(etaL_str)_g$(g_str)_V$(V_str)_T$(T_str).csv"
save_probs(list_L, list_probs; folder, file_name)
