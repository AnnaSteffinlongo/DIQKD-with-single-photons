"""
Script to compute the parameters that maximize the analytical expression for the key rate 
with noisy preprocessing.
This script in particular uses states and measurements of the form (Enky's)
"""


######################### FUNCTIONS TO LOAD THE PARAMETERS #########################
def fix_OBS(eta_T, t_BS):
    """
    Load the observables for a given:
        eta_T    --    total efficiency
        t_BS     --    beamsplitter transmissivity
        
    """
    if eta_T==0.90:
        if t_BS == 0:    
            A1 = np.array([[0.582588560928011+3.469446951953614e-18j, 0.6938328891619736-0.36767759589619403j], 
                           [0.6938328891619738+0.36767759589619387j, -0.5346349753298426-1.3877787807814457e-17j]])
            A2 = np.array([[0.2714745970586325+0j,0.3036432204360167+0.8537612160621038j], 
                           [0.3036432204360167-0.8537612160621033j, -0.29905948104750557+1.1102230246251565e-16j]])
            B1 = np.array([[0.5835899121678925+3.469446951953614e-18j, -0.705206478808073-0.3437397792645212j], 
                           [-0.7052064788080729+0.3437397792645211j, -0.5356416957343718+2.7755575615628914e-17j]])
            B2 = np.array([[0.27049353059001935+0j, -0.2757939199856589+0.8633252763099399j], 
                           [-0.27579391998565894-0.8633252763099398j, -0.2984020226379631-6.938893903907228e-17j]])
            B3 = np.array([[0.5807080637503155-1.734723475976807e-18j, -0.6949137477371763+0.36827616324878554j], 
                           [-0.6949137477371764-0.36827616324878554j, -0.533014500702605-4.163336342344337e-17j]])

        if t_BS == 0.005:
                A1 = np.array([[ 0.58232731-1.73472348e-18j,  0.67841927+3.95733524e-01j],
                               [ 0.67841927-3.95733524e-01j, -0.53441818-2.77555756e-17j]])
                A2 = np.array([[ 0.2713401 +0.00000000e+00j,  0.33832266-8.40645252e-01j],
                               [ 0.33832266+8.40645252e-01j, -0.2989694 +2.77555756e-17j]])
                B1 = np.array([[ 0.58326678-1.73472348e-18j, -0.71886631+3.14698056e-01j],
                               [-0.71886631-3.14698056e-01j, -0.53537424+0.00000000e+00j]])
                B2 = np.array([[ 0.27038888-6.93889390e-18j, -0.24040877-8.73857310e-01j],
                               [-0.24040877+8.73857310e-01j, -0.29833265-1.38777878e-17j]])
                B3 = np.array([[ 0.58046355-2.60208521e-18j, -0.67942729-3.96433885e-01j],
                               [-0.67942729+3.96433885e-01j, -0.53281124-2.77555756e-17j]])
    else:
        print('measurements not found')
    return A1, A2, B1, B2, B3



######################### FUNCTIONS TO COMPUTE THE SYMBOLIC ANALYTICAL KEY #########################


def state_one_photon(eta_T, t_BS):
    """
    Computes the state for a given 
        eta_T    --    total efficiency
        t_BS     --    beamsplitter transmissivity
    
    """
    eta_D=0.95              # detector efficiency
    eta_L = eta_T/eta_D     # local channel efficiency
    
    
    alpha = 2* (1-eta_L+t_BS*eta_L)
    beta = eta_L*(1-t_BS)
    

    state = np.array([[alpha, 0, 0, 0],
                            [0, beta, beta, 0],
                            [0, beta, beta, 0],
                            [0, 0, 0, 0]])
    state = state/2
                            
    
    return state




def CHSH_one_photon(eta_T, t_BS):
    """
    Computes the symbolic expression for the CHSH
        eta_T    --    total efficiency
        t_BS     --    beamsplitter transmissivity
        
    """
    
    S = np.trace(state_one_photon(eta_T, t_BS) @ (np.kron( (fix_OBS(eta_T, t_BS)[0]+fix_OBS(eta_T, t_BS)[1]), fix_OBS(eta_T, t_BS)[2] )+np.kron( (fix_OBS(eta_T, t_BS)[0]-fix_OBS(eta_T, t_BS)[1]), (fix_OBS(eta_T, t_BS)[3]) )))
    
    return sp.Abs(S)
    

def h_func(X):
    """
    Computes the binary entropy of X, where 0 log 0 = 0 and 1 log 1 = 0
        
    """
    f = - X * sp.log(X, 2) - (1-X) * sp.log(1-X, 2)
    piecewise_expr = sp.Piecewise((0, sp.Or(sp.Eq(X, 0), sp.Eq(X, 1))), (f, True))

    return piecewise_expr
    

def analytical(eta_T, t_BS):
    """
    Computes the symbolic expression for the analytical key rate
        eta_T    --    total efficiency
        t_BS     --    beamsplitter transmissivity
                
    """
    q = sp.symbols('q')
    r = 1 - h_func((1+sp.sqrt((CHSH_one_photon(eta_T, t_BS)/2)**2-1))/2) + h_func((1+sp.sqrt(1-q*(1-q)*(8-CHSH_one_photon(eta_T, t_BS)**2)))/2) - cond_ent_symbolic(eta_T, t_BS) 
    
    return r

def opt_analytical(eta_T, t_BS):
    """
    Computes optimal value of the analytical key rate
        eta_T    --    total efficiency
        t_BS     --    beamsplitter transmissivity
        
    """
    # Definition of the symbolic variables over which to optimizze
    q = sp.symbols('q')

    # Set initial values
    
    initial_values = q_guess
    
    # Define objective function of the optimization problem
    objective_function = -analytical(eta_T, t_BS)
    
    # Define a function that returns the value of the objective function
    def objective_function_wrapper(values):
        q_val = values
        
        # Substitute values into the symbolic expression
        objective_function_numeric = sp.lambdify((q, ), objective_function, 'numpy')
        
        # Evaluate the objective function
        result = objective_function_numeric(q_val)
         
        print("Objective function value:", result)
    
        return float(result)

    ########## Perform the optimization ###############
    
    # Specify bounds for specific variables
    bounds = [(None, None)] * len(initial_values)  # Default bounds, no constraints
    bounds[0] = (0,0.35)
    
    # Run the optimization
    result = minimize(objective_function_wrapper, initial_values, method='SLSQP', bounds=bounds, options={'ftol': 1e-6,'disp': True, 'maxiter': 1000})
    
    # Save the value of the optimized objective function and the corresponding parameters
    opt_variable = result.x 
    opt_val = -result.fun
    
    return opt_variable, opt_val



def p_symbolic(a, b, x, y, eta_T, t_BS):
    """
    Computes the value of the symbolic probability element for the inputs x=1, y=3, leaving Ba and Bz
    as free variables
        a     --      Alice's output
        b     --      Bob's output
        x     --      Alice's input
        y     --      Bob's input
        eta_T    --    total efficiency
        t_BS     --    beamsplitter transmissivity
    
    """
    
    # Define symbolic variables
    
    A={}
    B={}
    A['1'], A['2'], B['1'], B['2'], B['3'] = fix_OBS(eta_T, t_BS)

    
    # Create symbolic expressions
    id = [[1,0],[0,1]]
    POVM = (id + ((-1)**b) * B[str(y)]  )/2      # Unnormalized POVM
    kron_expr = np.kron((id + ((-1)**a) * A[str(x)])/2, POVM)
    p_expr = np.trace(state_one_photon(eta_T, t_BS) @ kron_expr)

    return p_expr.real


def cond_ent_symbolic(eta_T, t_BS):
    """
    Computes the symbolic expression for H(A|B) = H(AB) - H(B) 
        eta_T    --    total efficiency
        t_BS     --    beamsplitter transmissivity
    
    """
    
    ##### Computing the probabilities to define the conditional entropy expression

    # Compute marginal, they are the same with and without preproc since they are the sum over a
    marg = []
    for b in range(2):
        marg_tmp = 0.0
        for a in range(2):
            marg_tmp += p_symbolic(a, b, 1, 3, eta_T, t_BS)
        marg.append(marg_tmp)
    
    
    q = sp.symbols('q')             # Bit-flip probability

    # Compute noisy joint
    joint_q = {}
    joint_q['00'] = (1 - q) * p_symbolic(0, 0, 1, 3, eta_T, t_BS) + q * p_symbolic(1, 0, 1, 3, eta_T, t_BS)
    joint_q['01'] = (1 - q) * p_symbolic(0, 1, 1, 3, eta_T, t_BS) + q * p_symbolic(1, 1, 1, 3, eta_T, t_BS)
 
    joint_q['10'] = marg[0] - joint_q['00']
    joint_q['11'] = marg[1] - joint_q['01']
    
    hab, hb = 0.0, 0.0
    
    for elem in joint_q:
        hab += -joint_q[elem] * sp.log(joint_q[elem], 2)
    
    # for prob in marg:
    for prob in marg:
        hb += -prob * sp.log(prob, 2)
    
    # Conditional entropy expression
    err_term = hab - hb
    
    return err_term




import csv
import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize

t_BS=0.005
eta_T=0.90

print(sp.N(CHSH_one_photon(eta_T, t_BS)))
print(sp.N(cond_ent_symbolic(eta_T, t_BS)))
print(sp.N(analytical(eta_T, t_BS)))


q_guess = [0.3]
   
keys = []

# Run the optimization
variabs, val = opt_analytical(eta_T, t_BS)
print('eta_T = ', eta_T)
print('key = ', val)
print('q = ', variabs[0])

# Store the values obtained
key = val

# Add the optimized values to the input params in order to start the next optimization from the previously optimized point
q_final = variabs[0]
           # Index to be updated



file_path = './t' + str(t_BS) + '_eta_T' + str(eta_T)+'.csv'

headers = np.array(["eta", "keys", "q"])
save_data = np.array([[eta_T], [key], [q_final]])

# Combine headers and data
output_data = np.hstack((headers[:, None], save_data))

# Save data with headers vertically
np.savetxt(file_path, output_data, fmt='%s', delimiter=',')

