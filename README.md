# DIQKD-with-single-photons
Examples of scripts used to obtain the data presented in arXiv:2409.17075.

- State_Measurement_functions.jl include functions to generate the heralded state (both for SPDC sources and QD) and measurement (both with and without squeezing) considered in our paper. 
- Asymptotic_key_rate.jl imports State_Measurement_functions.jl and optimizes the parameters to compute CHSH and asymptotic key rate both as a function of noise (for fixed distance) and as a function of distance (for fixed noise). It is used to create the data for Fig.2 in the main text and all the Fig. in Supplemental Material Sec. C.
- Finite_size_analysis.py can be used to obtain the finite-size key rate as a function of the distance for different number of protocol rounds N. This data is used for Fig.3 in the main text.
