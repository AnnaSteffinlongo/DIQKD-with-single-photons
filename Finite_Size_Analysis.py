"""
This script implements the finite size analysis for DIQKD protocols.
It computes the key rate for a given set of probabilities for each distance L, for different number of rounds.
Requires to import a set of probabilities for each considered distance L. 
These probabilities can be generated using Asymptotic_key_rate.jl

Author: Anna Steffinlongo
"""


######################### FUNCTIONS TO LOAD THE PARAMETERS #########################

def CHSH(probs):
    """
    Computes the CHSH for given probabilities.
    
    # Arguments
    probs: numpy array of shape (2, 2, 2, 3)
        - probs[a, b, x, y] gives the probability for outcomes a, b given inputs x (Alice) and y (Bob)
        - The last dimension (y) corresponds to Bob's three possible inputs; only y = 0 and y = 1 are used for CHSH computation        
    
    # Returns
    S: CHSH value.

    """
    S = np.sum([
        (-1)**(a + b + x*y) * probs[a, b, x, y]
        for a in range(2)
        for b in range(2)
        for x in range(2)
        for y in range(2)
    ])
    return abs(S)
    
def h_func(X):
    """
    Computes the binary entropy of X, where 0 log 0 = 0 and 1 log 1 = 0.

    # Arguments
    X: symbolic variable or expression representing the probability.

    # Returns
    piecewise_expr: symbolic expression for the binary entropy function.
        
    """
    f = - X * sp.log(X, 2) - (1-X) * sp.log(1-X, 2)
    piecewise_expr = sp.Piecewise((0, sp.Or(sp.Eq(X, 0), sp.Eq(X, 1))), (f, True))

    return piecewise_expr


######################### FUNCTIONS FOR FINITE SIZE ANALYSIS #########################
# More details in Supplemental Material Sec. E

def eta_func_symb():
    # Define the symbolic variable
    t = sp.symbols('t')
    q = sp.symbols('q')
    
    # Define the piecewise function
    func = sp.Piecewise(
        (0.0, (0.25 <= t) & (t <= 0.75)),
        (1 - h_func((1 + sp.sqrt(16*t*(t-1)+3))/2) + h_func((1+sp.sqrt(1 - (1 - q) * q * (8 - (4*(2*t-1))**2)))/2), ((1-1/sqrt(2))/2 <= t) & (t <= (1+1/sqrt(2))/2)),
        )
    return func
 
def g_func(w):
    t = sp.symbols('t')

    gS = eta_func_symb() + (w-t)*sp.diff(eta_func_symb(), t)
    return gS

def f_min(u):
    gamma = sp.symbols('gamma')

    if u == 0:
        fS = 1/gamma *g_func(0) + (1-1/gamma) * g_func(1)
    elif u == 1:
        fS = g_func(1)
    elif u == 2:
        fS = g_func(1)
    return fS

def f_min_prob():
    gamma = sp.symbols('gamma')
    w = sp.symbols('w')
    
    p0 = gamma*(1-w)
    p1 = gamma*w
    p2 = 1 - gamma
    return p0 * f_min(0) + p1 * f_min(1) + p2 * f_min(2)

def Var_f():
    w = sp.symbols('w')
    gamma = sp.symbols('gamma')
    
    p0 = gamma*(1-w)
    p1 = gamma*w
    p2 = 1 - gamma
    var = p0 * f_min(0)**2 + p1 * f_min(1)**2 + p2 * f_min(2)**2 - f_min_prob()**2
    return var

def MVar():
    gamma = sp.symbols('gamma')

    Delta0 = g_func(1)-g_func(0)
    value =(2+sqrt(2))/(4*gamma)*Delta0**2
    return value

def Volume():
    return sp.ln(2)/2 * (log(33,2)+sp.sqrt(2+MVar()))**2

def theta(eps):
    return sp.log(2/eps**2,2)

def K(a1):
    a1 = sp.symbols('a1')

    wmin = (1-1/sqrt(2))/2
    func = 1/(6*(2-a1)**3*sp.ln(2))*2**((a1-1)*(2+g_func(1)-g_func(wmin)))*(sp.ln(2**(2+g_func(1)-g_func(wmin))+np.e**2))**3
    return func

def Yb(x):
    from scipy.special import lambertw
    b = 4/log(2)
    return b*sp.LambertW(sp.exp(x/b)/b)


def EC(probs):
    gamma = sp.symbols('gamma')
    n = sp.symbols('n')

    S_EC = CHSH(probs)
    m = n*((1-gamma)*cond_ent_symbolic(probs) + gamma*h_func((4-S_EC)/8)) + 50*sp.sqrt(n)
    return m

def wth(probs):
    gamma = sp.symbols('gamma')
    n = sp.symbols('n')
    
    S = CHSH(probs)
    w1 = (4+S)/8
    k = 3
    p1 = gamma*(1-w1)
    
    pth = p1 + k*sp.sqrt(p1*(1-p1)/n)
    eq25 = 1 - pth/gamma
    return eq25    



######################### FINITE SIZE ANALYSIS #########################

def argument(probs, N):
    """ Computes the argument of the optimization function.
    
    # Arguments 
    probs: numpy array of shape (2, 2, 2, 3).
    N: number of rounds.

    # Returns
    l: symbolic expression for the argument of the optimization function.
    """
    
    a1 = sp.symbols('a1')
    a2 = sp.symbols('a2')
    eps = sp.symbols('eps')
    eps1 = sp.symbols('eps1')
    eps2 = sp.symbols('eps2')
    eps_EA = sp.symbols('eps_EA')
    eps_PA = sp.symbols('eps_PA')
    gamma = sp.symbols('gamma')
    n = sp.symbols('n')
    
    # Fixing the security parameters 
    sub={eps:1e-10, eps1:3e-11, eps2:3e-11, eps_EA:1e-10, eps_PA:1e-10, n:N}
    
    l = N * g_func(wth(probs)) - (a1-1)*Volume()-\
        n*(a1-1)**2*K(a1) - n*gamma -n*(a2-1)*(log(5,2))**2-\
        1/(a1-1)*(theta(eps1)+a1*sp.log(1/eps_EA,2)) - \
        1/(a2-1)*(theta(eps2)+a2*sp.log(1/eps_EA,2)) - \
        3*theta(eps-eps1-2*eps2) - 5*sp.log(1/eps_PA,2) - EC(probs) -264
        
    l = l.subs(sub)
      
    return l


def optimize_key(probs, N):
    """
    Optimizes the key rate for given probabilities and number of rounds N.

    # Arguments
    probs: numpy array of shape (2, 2, 2, 3).
    N: number of rounds.

    # Returns
    key_rate: optimized key rate.
    opt_variables: optimized parameters.
    result: optimization result object.
    """ 

    a1 = sp.symbols('a1')
    a2 = sp.symbols('a2')
    gamma = sp.symbols('gamma')
    t = sp.symbols('t')
    q = sp.symbols('q')

    # Define the objective function for the optimization variables
    obj= -(argument(probs, N)/N)
    
    def objective_function_wrapper(values):
        a1_val, a2_val, t_val, gamma_val, q_val = values
        # Substitute values into the symbolic expression
        objective_function_numeric = sp.lambdify((a1, ) + (a2,) + (t,) + (gamma,) + (q,), obj, 'numpy')
        
        # Evaluate the objective function
        with np.errstate(divide='ignore', invalid='ignore'):
            result = objective_function_numeric(a1_val, a2_val, t_val, gamma_val, q_val)
        
        
        print("Objective function value:", result)
        return float(result)
  
    # Initial guess and bounds for the optimization variables
    initial_values = guess[j]
    bounds = [(1.00000001, 2), (1.000000001, 1 + 1/log(5,2)), (0.7500001,(1+1/sqrt(2))/2), (1e-6,1), (0,0.5)]
    
    # Run the optimization    
    result = minimize(objective_function_wrapper, initial_values, method='SLSQP', bounds=bounds, options={'disp': True, 'maxiter': 1000})
    
    # Save the value of the optimized objective function and the corresponding parameters
    if result.success:    
        opt_variables = result.x 
        opt_val = -result.fun
        key_rate = Yb(opt_val*N)/N
        
        print('key = ', key_rate)
        
    else:
        print('Error in the optimization')
        opt_variables = result.x 
        opt_val = -result.fun
        key_rate = Yb(opt_val*N)/N
        
        print(result)

    return key_rate, opt_variables, result

def cond_ent_symbolic(probs):
    """
    Computes the symbolic expression for the conditional entropy expression H(A|B) = H(AB) - H(B) 

    # Arguments
    probs: numpy array of shape (2, 2, 2, 3).

    # Returns
    err_term: symbolic expression for the conditional entropy H(A|B).
    """

    # Compute marginal, they are the same with and without noisy preprocessing
    marg = []
    for b in range(2):
        marg_tmp = 0.0
        for a in range(2):
            marg_tmp += probs[a, b, 0, 2]
        marg.append(marg_tmp)
    
    q = sp.symbols('q')             # Bit-flip probability

    # Compute joint probabilities for noisy preprocessing
    joint_q = {}
    joint_q['00'] = (1 - q) * probs[0, 0, 0, 2] + q * probs[1, 0, 0, 2]
    joint_q['01'] = (1 - q) * probs[0, 1, 0, 2] + q * probs[1, 1, 0, 2]
 
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



######################### MAIN CODE #########################

import numpy as np
from math import sqrt, log2, log, pi, cos, sin
from sympy.physics.quantum.dagger import Dagger
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize
import pandas as pd

# Constants (example placeholders, adjust to your values)
R = 75e6
ηD_C = 0.95
t_BS = 0.01
   
# Number of rounds for the finite size analysis
N_list=[1e10,1e9,1e8]

# Include guess for optimization of the noisy preoprocessing probability q for each L
q_guess = np.loadtxt("q_guess.csv")

# Load the probability data
df = pd.read_csv("data.csv")

# Group by L to get one distribution per L
grouped = df.groupby("L")
   


# === Initialize Results Storage ===
results = []

# === Loop Over Each Group of Probabilities (Each L) ===
for i, (L_val, group) in enumerate(grouped):
    # Build Probability Array for Current L 
    probs_dict = {
        (a, b, x, y): p
        for a, b, x, y, p in zip(group['a'], group['b'], group['x'], group['y'], group['p_abxy'])
    }

    p_array = np.zeros((2, 2, 2, 3))
    for (a, b, x, y), p in probs_dict.items():
        p_array[a, b, x, y] = p

    # Set Initial Guess for Optimization Variables 
    guess = [[1.00001,  1.00001,  0.78,  2e-6, q_guess[i]]]

    # Calculate Heralding Rate for Current L
    heralding_rate = R * t_BS * (1 - t_BS) * ηD_C * 10 ** (-0.2 * L_val / 20)

    j = 0
    # === Loop Over Each N (Number of Rounds) ===
    for N in N_list:
        print(f"L = {L_val:.2f}, N = {N:.1e}")
        # Run key rate optimization
        key, vars_opt, stats = optimize_key(p_array, N) 

        guess.append(vars_opt)
        j = j+1

        # Store results for current L and N
        results.append({
            "L": L_val,
            "N": int(N),
            "key": heralding_rate*key,
            "a1": vars_opt[0],
            "a2": vars_opt[1],
            "t": vars_opt[2],
            "gamma": vars_opt[3],
            "q": vars_opt[4],
        })

# === Save Results to CSV ===
df_results = pd.DataFrame(results)

df_results.to_csv(f"./finite_size_data.csv", index=False)
