"""
Script to compute the parameters that maximize the analytical expression for the 
finite size key rate with noisy preprocessing.
This script in particular uses states and measurements from appendix B 
and follows the procedure in appendix C.
"""



def load_params(namefile,col):  
    """
    Opens a cvs file with the noisy preprocessing parameters and loads it. Outputs the vectors with the data
    
        namefile    --    name of the file from which to load the data
        col         --    the number of the column of the file from which to load
                          the data. It is related to the eta considered.
    """
    
    csv_data = []
    with open(namefile, 'r') as file:

        csv_reader = csv.reader(file)

        # Iterate over the rows and append them to the list
        for line in csv_reader:
            csv_data += [line[1:]]
                       
        
    eta_list = float(csv_data[0][col]) 
    qs = float(csv_data[2][col])
    abs_key = float(csv_data[1][col])  
    
    A = {}
    B = {}
    
    for x in range(1,3):
        A[str(x) + 'a'] = [float(csv_data[3+2*(x-1)][col])]
        A[str(x) + 'z'] = [float(csv_data[4+2*(x-1)][col])]
        A[str(x) + 'ap'] = [float(csv_data[5+2*(x-1)][col])]
        A[str(x) + 'zp'] = [float(csv_data[6+2*(x-1)][col])]
    for y in range(1,4):
        B[str(y) + 'a'] = [float(csv_data[11+2*(y-1)][col])]
        B[str(y) + 'z'] = [float(csv_data[12+2*(y-1)][col])]
        B[str(y) + 'ap'] = [float(csv_data[13+2*(y-1)][col])]
        B[str(y) + 'zp'] = [float(csv_data[14+2*(y-1)][col])]
            
        
    return  eta_list, abs_key,qs, A, B


def state(eta, T):
    """
    Computes the state for a given eta 
    
        eta   --    local efficiecny
        T     --    beamsplitter transmissivity
    
    """
    
    alpha = 2* (1-eta+T*eta)
    beta = eta*(1-T)
    

    state =sp.Matrix([[alpha, 0, 0, 0],
                           [0, beta, beta, 0],
                           [0, beta, beta, 0],
                           [0, 0, 0, 0]])
    
    state = np.array(state)/2
    
    return state


def OBS(params):
    """
    Computes the observable
        
        params    --    vector (displacement parameter, squezing parameter)
        
    """
    
    from sympy.parsing import mathematica as M
    
    a = params[0]
    z = params[1]
    phi = params[3]
    theta = params[2]
    
    sech = 1 / sp.cosh(z)
    
    P = sp.Matrix([[2*sp.exp(a**2*(sp.cos(phi - 2*theta)*sp.tanh(z) - 1))*sech - 1, 2*a*sp.exp(a**2*sp.cos(phi - 2*theta)*sp.tanh(z) - a**2 - 1j*theta)*sech**2], [2*a*sp.exp(a**2*sp.cos(phi - 2*theta)*sp.tanh(z) - a**2 + 1j*theta)*sech**2, 2*a**2*sp.exp(a**2*(sp.cos(phi - 2*theta)*sp.tanh(z) - 1))*sech**3 - 1]]
        )
    return P



def CHSH(eta,T):
    """
    Computes the symbolic expression for the CHSH
    
        eta   --    local efficiecny
        T     --    beamsplitter transmissivity
        
    """
    A_sym = {}
    A_sym[1] = sp.symbols('A1a A1z A1ap A1zp')
    A_sym[2] = sp.symbols('A2a A2z A2ap A2zp')
    B_sym = {}
    B_sym[1] = sp.symbols('B1a B1z B1ap B1zp')
    B_sym[2] = sp.symbols('B2a B2z B2ap B2zp')
    B_sym[3] = sp.symbols('B3a B3z B3ap B3zp')
    
    S = sp.trace(state(eta,T) @ (sp.kronecker_product( (OBS(A_sym[1])+OBS(A_sym[2])), OBS(B_sym[1]) )+np.kron( (OBS(A_sym[1])-OBS(A_sym[2])), (OBS(B_sym[2])) )))

    return sp.Abs(S)
    

def h_func(X):
    """
    Computes the binary entropy of X, where 0 log 0 = 0 and 1 log 1 = 0
        
    """
    f = - X * sp.log(X, 2) - (1-X) * sp.log(1-X, 2)
    piecewise_expr = sp.Piecewise((0, sp.Or(sp.Eq(X, 0), sp.Eq(X, 1))), (f, True))

    return piecewise_expr
    

def eta_func_symb():
    # Define the symbolic variable
    t = sp.symbols('t')
    q = sp.symbols('q')
    tolerance = 1e-15
        
    # Define the piecewise function
    func = sp.Piecewise(
        (0.0, (0.25 <= t) & (t <= 0.75)),
        (1 - h_func((1 + sp.sqrt(16*t*(t-1)+3))/2) + h_func((1+sp.sqrt(1 - 8*(-1 + q)*q*(1 - 8*t + 8*t**2)))/2), ((1-1/sqrt(2))/2 <= t) & (t <= (1+1/sqrt(2))/2)),
        (1 - h_func((1 + sp.sqrt(16*((1+1/sqrt(2))/2)*(((1+1/sqrt(2))/2)-1)+3))/2) + h_func((1+sp.sqrt(1 - 8*(-1 + q)*q*(1 - 8*((1+1/sqrt(2))/2) + 8*((1+1/sqrt(2))/2)**2)))/2), (((1+1/sqrt(2))/2) <= t) & (t <= (1+1/sqrt(2))/2+tolerance))
    )
    return func
 

"""
    The following functions are defined according to appendix C
"""

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

def MVar():
    gamma = sp.symbols('gamma')
    Delta0 = g_func(1)-g_func(0)
    Delta1 = 0
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

def EC(eta, T):
    gamma = sp.symbols('gamma')

    S_EC = CHSH(eta, T)
    
    n = sp.symbols('n')
    
    m = n*((1-gamma)*cond_ent_symbolic(eta, T) + gamma*h_func((4-S_EC)/8)) + 50*sp.sqrt(n)
    return m


def wth(eta, T):
    gamma = sp.symbols('gamma')
    n = sp.symbols('n')

    S = CHSH(eta, T)
    
    w1 = (4+S)/8
    k = 3
    p1 = gamma*(1-w1)
    
    pth = p1 + k*sp.sqrt(p1*(1-p1)/n)
    eq25 = 1 - pth/gamma
    return eq25    


def argument_N(eta, T, N):
    a1 = sp.symbols('a1')
    a2 = sp.symbols('a2')
    eps = sp.symbols('eps')
    eps1 = sp.symbols('eps1')
    eps2 = sp.symbols('eps2')
    eps_EA = sp.symbols('eps_EA')
    eps_PA = sp.symbols('eps_PA')
    gamma = sp.symbols('gamma')
    n = sp.symbols('n')
    S = sp.symbols('S')
    t = sp.symbols('t')
    
    # Compute the key according to Eq. C16
    l = N * g_func(wth(eta, T)) - (a1-1)*Volume()-\
        n*(a1-1)**2*K(a1) - n*gamma -n*(a2-1)*(log(5,2))**2-\
        1/(a1-1)*(theta(eps1)+a1*sp.log(1/eps_EA,2)) - \
        1/(a2-1)*(theta(eps2)+a2*sp.log(1/eps_EA,2)) - \
        3*theta(eps-eps1-2*eps2) - 5*sp.log(1/eps_PA,2) - EC(eta, T) -264
        
    # Fix some parameters 
    sub={eps:1e-6, eps1:3e-7, eps2:3e-7, eps_EA:1e-6, eps_PA:1e-6, n:N}
    l = l.subs(sub)
          
    return l


def optimize_key(eta, T, N, i):
    """
    Computes optimal value of the analytical key rate
    
        eta   --    local efficiecny
        T     --    beamsplitter transmissivity
        N     --    number of rounds 
        i     --    index corresponding to eta
        
    """
    
    a1 = sp.symbols('a1')
    a2 = sp.symbols('a2')
    gamma = sp.symbols('gamma')
    t = sp.symbols('t')
    q = sp.symbols('q')
    
    A_sym = {}
    A_sym[1] = sp.symbols('A1a A1z A1ap A1zp')
    A_sym[2] = sp.symbols('A2a A2z A2ap A2zp')
    B_sym = {}
    B_sym[1] = sp.symbols('B1a B1z B1ap B1zp')
    B_sym[2] = sp.symbols('B2a B2z B2ap B2zp')
    B_sym[3] = sp.symbols('B3a B3z B3ap B3zp')
       
 
    # Define the function to be optimized
    obj= -(argument_N(eta, T, N)/N)

    A_sym_flat = [sym for sublist in A_sym.values() for sym in sublist]
    B_sym_flat = [sym for sublist in B_sym.values() for sym in sublist]
    
    
    
    def objective_function_wrapper(values):
        a1_val, a2_val, t_val, gamma_val, q_val, *rest_values = values
        A_values = rest_values[:len(A_sym_flat)]
        B_values = rest_values[len(A_sym_flat):]
       
        # Substitute values into the symbolic expression
        objective_function_numeric = sp.lambdify((a1, ) + (a2,) + (t,) + (gamma,) + (q,) + tuple(A_sym_flat) + tuple(B_sym_flat), obj, 'numpy')
        
        # Evaluate the objective function
        result = objective_function_numeric(a1_val, a2_val, t_val, gamma_val, q_val, *A_values, *B_values)
        
        print("Objective function value:", result)
        return float(result)
  
    # Fix the initial values
    initial_values = guess[i]
    
    # Bounds for the variables
    bounds = [(1, 2), (1, 1 + 1/log(5,2)), (3/4,(1+1/sqrt(2))/2), (0,1), (0,0.35)]
        
    for ii in range(len(bounds), len(initial_values),4):
        bounds += [(-1,1)]
        bounds += [(-1,1)]
        bounds += [(-np.pi, np.pi)]
        bounds += [(-np.pi, np.pi)]
    
    result = minimize(objective_function_wrapper, initial_values, method='SLSQP', bounds=bounds, options={'disp': True, 'maxiter': 1000})
    
    # Save the value of the optimized objective function and the corresponding parameters
    if result.success:    
        opt_variables = result.x 
        opt_val = -result.fun
        status = 0
        key_rate = Yb(opt_val*N)/N
        
        print('key = ', key_rate)
        
    else:
        print('Error in the optimization')
        status = -1
        opt_variables = result.x 
        opt_val = -result.fun
        key_rate = Yb(opt_val*N)/N
        print(result)
        
    return status, key_rate, opt_variables, result


    
def p_symbolic(a, b, x, y, eta, T):
    """
    Computes the value of the symbolic probability element for the inputs x=1, y=3, leaving Ba and Bz
    as free variables
        
        a     --      Alice's output
        b     --      Bob's output
        x     --      Alice's input
        y     --      Bob's input
        eta   --      local efficiecny
        T     --      beamsplitter transmissivity
        
    """
    
    # Define symbolic variables
    A_sym = {}
    A_sym[1] = sp.symbols('A1a A1z A1ap A1zp')
    A_sym[2] = sp.symbols('A2a A2z A2ap A2zp')
    B_sym = {}
    B_sym[1] = sp.symbols('B1a B1z B1ap B1zp')
    B_sym[2] = sp.symbols('B2a B2z B2ap B2zp')
    B_sym[3] = sp.symbols('B3a B3z B3ap B3zp')
    

    # Create symbolic expressions
    id = sp.eye(2)
    POVM = id + ((-1)**b) * OBS(B_sym[y])        # Unnormalized POVM
    kron_expr = sp.kronecker_product((id + ((-1)**a) * OBS(A_sym[x])), POVM)
    p_expr = 0.25 * sp.trace(state(eta, T) * kron_expr)
    
    return p_expr


def cond_ent_symbolic(eta, T):
    """
    Computes the symbolic expression for H(A|B) = H(AB) - H(B) 
    
    """
    
    ##### Computing the probabilities to define the conditional entropy expression

    # Compute marginal, they are the same with and without preproc since they are the sum over a
    marg = []
    for b in range(2):
        marg_tmp = 0.0
        for a in range(2):
            marg_tmp += p_symbolic(a, b, 1, 3, eta, T)
        marg.append(marg_tmp)
    
    
    q = sp.symbols('q')             # Bit-flip probability
    
    # Compute noisy joint
    joint_q = {}
    joint_q['00'] = (1 - q) * p_symbolic(0, 0, 1, 3, eta, T) + q * p_symbolic(1, 0, 1, 3, eta, T)
    joint_q['01'] = (1 - q) * p_symbolic(0, 1, 1, 3, eta, T) + q * p_symbolic(1, 1, 1, 3, eta, T)
    
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
    
    return sp.Abs(err_term)
    # return err_term



import csv
import numpy as np
from math import sqrt, log2, log, pi, cos, sin

from sympy.physics.quantum.dagger import Dagger


import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize


# Beamsplitter trasmissivity (necessary for comoputing the state)
t_BS = 0.005

# Data to be loaded
data = './../More_general_meas/t'+str(t_BS)+'_from_CHSH.csv'
# Column corresponding to the eta selected
col = 36

# load data from the csv file
eta, abs_key, q_guess, Apar, Bpar  = load_params(data, col)



a1_guess = [1.00000229e+00]
a2_guess = [1.00000009e+00]
gamma_guess = [0.790883274662436]
t_var_guess = [2.34413366e-06]



keys = []
best_vars = []
errors = []

guess = [sp.flatten([a1_guess, a2_guess, gamma_guess, t_var_guess, q_guess,sp.flatten(Apar.values()),sp.flatten(Bpar.values())])]

N_list = np.concatenate([np.logspace(20,13,1), np.logspace(10,9,1), np.logspace(9,7,30)])

"""
Cycling over N
"""

i=0
for N in N_list:  
    print('******************\nN = ', N)
    # Run the optimization
    err, val, variabs, stats = optimize_key(eta, t_BS, N, i)
    
    print('key = ', val)
    
    # Correct convergence check
    if N == 1e20:
        if np.abs(val - abs_key)>1e-5:
            print("Asymtotic key error")   
            err += 1
            
    # Store the values obtained
    keys += [val]
    errors += [err]
    # Add the optimized values to the input params in order to start the next optimization from the previously optimized point
    guess += [variabs]    
        
    i+=1            # Index to be updated


print(errors)

# # Creating the vector to save the optimized parameters
a1_opt = [guess[x][0] for x in range(1, len(keys)+1)]
a2_opt = [guess[x][1] for x in range(1, len(keys)+1)]
t_opt = [guess[x][2] for x in range(1,len(keys)+1)]
gamma_opt = [guess[x][3] for x in range(1,len(keys)+1)]
q_opt = [guess[x][4] for x in range(1,len(keys)+1)]
A1a_opt = [guess[x][5] for x in range(1, len(keys)+1)]
A1z_opt = [guess[x][6] for x in range(1, len(keys)+1)]
A1ap_opt = [guess[x][7] for x in range(1, len(keys)+1)]
A1zp_opt = [guess[x][8] for x in range(1, len(keys)+1)]
A2a_opt = [guess[x][9] for x in range(1, len(keys)+1)]
A2z_opt = [guess[x][10] for x in range(1, len(keys)+1)]
A2ap_opt = [guess[x][11] for x in range(1, len(keys)+1)]
A2zp_opt = [guess[x][12] for x in range(1, len(keys)+1)]
B1a_opt = [guess[x][13] for x in range(1, len(keys)+1)]
B1z_opt = [guess[x][14] for x in range(1, len(keys)+1)]
B1ap_opt = [guess[x][15] for x in range(1, len(keys)+1)]
B1zp_opt = [guess[x][16] for x in range(1, len(keys)+1)]
B2a_opt = [guess[x][17] for x in range(1, len(keys)+1)]
B2z_opt = [guess[x][18] for x in range(1, len(keys)+1)]
B2ap_opt = [guess[x][19] for x in range(1, len(keys)+1)]
B2zp_opt = [guess[x][20] for x in range(1, len(keys)+1)]
B3a_opt = [guess[x][21] for x in range(1, len(keys)+1)]
B3z_opt = [guess[x][22] for x in range(1, len(keys)+1)]
B3ap_opt = [guess[x][23] for x in range(1, len(keys)+1)]
B3zp_opt = [guess[x][24] for x in range(1, len(keys)+1)]



save_data = [N_list, keys, a1_opt, a2_opt, t_opt, gamma_opt, q_opt, A1a_opt, A1z_opt, A1ap_opt, A1zp_opt, A2a_opt, A2z_opt, A2ap_opt, A2zp_opt, B1a_opt, B1z_opt, B1ap_opt, B1zp_opt, B2a_opt, B2z_opt, B2ap_opt, B2zp_opt, B3a_opt, B3z_opt, B3ap_opt, B3zp_opt]

file_path = './eta'+str(eta)+'_t'+str(t_BS)+'_finite_size.csv'

headers = np.array(["N", "keys", "a1", "a2", "t", "gamma", "q",
"A1a", "A1z","A1ap", "A1zp", "A2a", "A2z","A2ap", "A2zp",
"B1a", "B1z","B1ap", "B1zp", "B2a", "B2z","B2ap", "B2zp",
"B3a", "B3z","B3ap", "B3zp"])

# Combine headers and data
output_data = np.hstack((headers[:, None], save_data))

# Save data with headers vertically
np.savetxt(file_path, output_data, fmt='%.8f', delimiter=',')



"""
 Plot 
"""

plt.plot(N_list[:len(keys)], keys, label='eta='+str(eta))
plt.axhline(y=abs_key, linestyle='--', label='asymptotic')
plt.legend()
plt.xlabel("N")
plt.ylabel("R")
plt.xscale('log')
plt.ylim(0,abs_key+1e-2)
plt.xlim(1e4,1e20)
# plt.savefig('./finite_size'+str(eta)+'gamma'+str(gamma_fix)+'.png')
plt.show()

