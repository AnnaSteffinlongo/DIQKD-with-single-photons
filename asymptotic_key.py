"""
Script to compute the parameters that maximize the analytical expression for the
asymptotic key rate with noisy preprocessing.
This script in particular uses states and measurements from appendix B.
This script is used to compute the key rates used for Fig. 5 in appendix B.
"""


def load_params(namefile):  
    """
    Opens a cvs file with the noisy preprocessing parameters and loads it. Outputs the vectors with the data
    
        namefile    --    name of the file from which to load the data
        
    """
    csv_data = []
    with open(namefile, 'r') as file:

        csv_reader = csv.reader(file)

        
        # Iterate over the rows and append them to the list
        for line in csv_reader:
            csv_data += [line[1:]]
                       
            
        
    l= len(csv_data[0])
    eta_list = [float(csv_data[0][x]) for x in range(l)]
    
    qs = [0.0]
    
    A = {}
    B = {}
    
    for x in range(1,3):
        A[str(x) + 'a'] = [float(csv_data[2+4*(x-1)][0])]
        A[str(x) + 'z'] = [float(csv_data[3+4*(x-1)][0])]
        A[str(x) + 'ap'] = [float(csv_data[4+4*(x-1)][0])]
        A[str(x) + 'zp'] = [float(csv_data[5+4*(x-1)][0])]
    for y in range(1,3):
        B[str(y) + 'a'] = [float(csv_data[10+4*(y-1)][0])]
        B[str(y) + 'z'] = [float(csv_data[11+4*(y-1)][0])]
        B[str(y) + 'ap'] = [float(csv_data[12+4*(y-1)][0])]
        B[str(y) + 'zp'] = [float(csv_data[13+4*(y-1)][0])]
            
    B['3a'] = [1e-5]
    B['3z'] = [1e-5]
    B['3ap'] = [1e-5]
    B['3zp'] = [1e-5]
 
    
    return  eta_list, A, B


def state(eta, t):
    """
    Computes the state 
    
        eta   --    local efficiecny
        t     --    beamsplitter transmissivity
    
    """
    
    alpha = 2* (1-eta+t*eta)
    beta = eta*(1-t)
    

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
    theta = params[2]
    phi = params[3]
    
    
    sech = 1 / sp.cosh(z)
    
    P = sp.Matrix([[2*sp.exp(a**2*(sp.cos(phi - 2*theta)*sp.tanh(z) - 1))*sech - 1, 2*a*sp.exp(a**2*sp.cos(phi - 2*theta)*sp.tanh(z) - a**2 - 1j*theta)*sech**2], [2*a*sp.exp(a**2*sp.cos(phi - 2*theta)*sp.tanh(z) - a**2 + 1j*theta)*sech**2, 2*a**2*sp.exp(a**2*(sp.cos(phi - 2*theta)*sp.tanh(z) - 1))*sech**3 - 1]]
        )
    return P


def CHSH(eta,t):
    """
    Computes the symbolic expression for the CHSH
    
        eta   --    local efficiecny
        t     --    beamsplitter transmissivity
        
    """
    A_sym = {}
    A_sym[1] = sp.symbols('A1a A1z A1ap A1zp')
    A_sym[2] = sp.symbols('A2a A2z A2ap A2zp')
    B_sym = {}
    B_sym[1] = sp.symbols('B1a B1z B1ap B1zp')
    B_sym[2] = sp.symbols('B2a B2z B2ap B2zp')
    B_sym[3] = sp.symbols('B3a B3z B3ap B3zp')
    
    S = sp.trace(state(eta,t) @ (sp.kronecker_product( (OBS(A_sym[1])+OBS(A_sym[2])), OBS(B_sym[1]) )+sp.kronecker_product( (OBS(A_sym[1])-OBS(A_sym[2])), (OBS(B_sym[2])) )))

    return sp.Abs(S)
    

def h_func(X):
    """
    Computes the binary entropy of X, where 0 log 0 = 0 and 1 log 1 = 0
        
    """
    f = - X * sp.log(X, 2) - (1-X) * sp.log(1-X, 2)
    piecewise_expr = sp.Piecewise((0, sp.Or(sp.Eq(X, 0), sp.Eq(X, 1))), (f, True))

    return piecewise_expr
    

def analytical(eta, t):
    """
    Computes the symbolic expression for the analytical key rate
    
        eta   --    local efficiecny
        t     --    beamsplitter transmissivity
        
    """
    q = sp.symbols('q')
    r = 1 - h_func((1+sp.sqrt((CHSH(eta, t)/2)**2-1))/2) + h_func((1+sp.sqrt(1-q*(1-q)*(8-CHSH(eta, t)**2)))/2) - cond_ent_symbolic(eta, t) 
    
    return r

def opt_analytical(eta, t, i):
    """
    Computes optimal value of the analytical key rate
    
        eta   --    local efficiecny
        t     --    beamsplitter transmissivity
        i     --    index corresponding to eta
        
    """
    # Definition of the symbolic variables over which to optimizze
    A_sym = {}
    A_sym[1] = sp.symbols('A1a A1z A1ap A1zp')
    A_sym[2] = sp.symbols('A2a A2z A2ap A2zp')
    B_sym = {}
    B_sym[1] = sp.symbols('B1a B1z B1ap B1zp')
    B_sym[2] = sp.symbols('B2a B2z B2ap B2zp')
    B_sym[3] = sp.symbols('B3a B3z B3ap B3zp')
    q = sp.symbols('q')

    # Set initial values
    initial_values = guess[i]
    
    # Define objective function of the optimization problem
    objective_function = -analytical(eta, t)
    
    # Redefinition of the variables
    q_sym  = sp.symbols('q')
    A_sym_flat = [sym for sublist in A_sym.values() for sym in sublist]
    B_sym_flat = [sym for sublist in B_sym.values() for sym in sublist]
    
    # Define a function that returns the value of the objective function
    def objective_function_wrapper(values):
        q_val, *rest_values = values
        A_values = rest_values[:len(A_sym_flat)]
        B_values = rest_values[len(A_sym_flat):]
        
        # Substitute values into the symbolic expression
        objective_function_numeric = sp.lambdify((q_sym, ) + tuple(A_sym_flat) + tuple(B_sym_flat), objective_function, 'numpy')
        
        # # Evaluate the objective function
        result = objective_function_numeric(q_val, *A_values, *B_values)
         
        print("Objective function value:", result)
    
        return float(result)

    ########## Perform the optimization ###############
    
    # Specify bounds for specific variables
    bounds = [(None, None)] * len(initial_values)  # Default bounds, no constraints
    bounds[0] = (0,0.35)    
    for ii in range(1, len(initial_values),4):
        bounds[ii] = (-1,1)
        bounds[ii+1] = (-1,1)
        bounds[ii+2] = (-np.pi,np.pi)
        bounds[ii+3] = (-np.pi,np.pi)
        
    # Run the optimization
    result = minimize(objective_function_wrapper, initial_values, method='SLSQP', bounds=bounds, options={'ftol': 1e-6,'disp': True, 'maxiter': 1000})
   
    # Save the value of the optimized objective function and the corresponding parameters
    opt_variables = result.x 
    opt_val = -result.fun
     
    print("eta:", eta)
    print("Optimized Values:", opt_val)
    
    return opt_variables, opt_val


def p_symbolic(a, b, x, y, eta, t):
    """
    Computes the value of the symbolic probability element for the inputs x=1, y=3, leaving Ba and Bz
    as free variables
        
        a     --      Alice's output
        b     --      Bob's output
        x     --      Alice's input
        y     --      Bob's input
        eta   --    local efficiecny
        t     --    beamsplitter transmissivity
        
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
    p_expr = 0.25 * sp.trace(state(eta, t) * kron_expr)
    
    return p_expr


def cond_ent_symbolic(eta, t):
    """
    Computes the symbolic expression for H(A|B) = H(AB) - H(B) 
    
        eta   --    local efficiecny
        t     --    beamsplitter transmissivity
    
    """
    
    ##### Computing the probabilities to define the conditional entropy expression

    # Compute marginal, they are the same with and without preproc since they are the sum over a
    marg = []
    for b in range(2):
        marg_tmp = 0.0
        for a in range(2):
            marg_tmp += p_symbolic(a, b, 1, 3, eta, t)
        marg.append(marg_tmp)
    
    q = sp.symbols('q')             # Bit-flip probability
    
    # Compute noisy joint
    joint_q = {}
    joint_q['00'] = (1 - q) * p_symbolic(0, 0, 1, 3, eta, t) + q * p_symbolic(1, 0, 1, 3, eta, t)
    joint_q['01'] = (1 - q) * p_symbolic(0, 1, 1, 3, eta, t) + q * p_symbolic(1, 1, 1, 3, eta, t)
    
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




def flatten_sum(matrix):
    return sum(matrix, [])

import csv
import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import sympy as sp
from scipy.optimize import minimize



# load data from the csv file

# location of the file from which to load the data
data = './../More_general_meas/t0.01_CHSH.csv'
# Apar and Bpar will be used as initial points for the optimization with efficiecy eta=1
eta_list, Apar, Bpar  = load_params(data)

# Initial guess for the noisy preprocesing probability
q_guess = [0]
# List of initial values for the optimization
guess = [sp.flatten([q_guess,sp.flatten(Apar.values()),sp.flatten(Bpar.values())])]

# Beamsplitter trasmissivity (necessary for comoputing the state)
T = 1e-2

"""
Cycling over etas 
"""

keys = []
# Change if we want different values of eta
eta_list = np.concatenate([np.linspace(1,0.94,13), np.linspace(0.939,0.9,40)])

i=0
for eta in eta_list:  
    print('******************\neta = ', eta)
    # Run the optimization
    variabs, val = opt_analytical(eta, T, i)
    print('key = ', val)
    
    # Store the values obtained
    keys += [val]
    
    # Add the optimized values to the input params in order to start the next optimization from the previously optimized point    
    guess += [variabs]
    i+=1            # Index to be updated

# reorder parameters to be saved
q_opt = [guess[x][0] for x in range(1,len(keys)+1)]
A1a_opt = [guess[x][1] for x in range(1, len(keys)+1)]
A1z_opt = [guess[x][2] for x in range(1, len(keys)+1)]
A1ap_opt = [guess[x][3] for x in range(1, len(keys)+1)]
A1zp_opt = [guess[x][4] for x in range(1, len(keys)+1)]
A2a_opt = [guess[x][5] for x in range(1, len(keys)+1)]
A2z_opt = [guess[x][6] for x in range(1, len(keys)+1)]
A2ap_opt = [guess[x][7] for x in range(1, len(keys)+1)]
A2zp_opt = [guess[x][8] for x in range(1, len(keys)+1)]
B1a_opt = [guess[x][9] for x in range(1, len(keys)+1)]
B1z_opt = [guess[x][10] for x in range(1, len(keys)+1)]
B1ap_opt = [guess[x][11] for x in range(1, len(keys)+1)]
B1zp_opt = [guess[x][12] for x in range(1, len(keys)+1)]
B2a_opt = [guess[x][13] for x in range(1, len(keys)+1)]
B2z_opt = [guess[x][14] for x in range(1, len(keys)+1)]
B2ap_opt = [guess[x][15] for x in range(1, len(keys)+1)]
B2zp_opt = [guess[x][16] for x in range(1, len(keys)+1)]
B3a_opt = [guess[x][17] for x in range(1, len(keys)+1)]
B3z_opt = [guess[x][18] for x in range(1, len(keys)+1)]
B3ap_opt = [guess[x][19] for x in range(1, len(keys)+1)]
B3zp_opt = [guess[x][20] for x in range(1, len(keys)+1)]

# Creating the vector to save the optimized parameters
save_data = np.array([eta_list, keys, q_opt, A1a_opt, A1z_opt, A1ap_opt, A1zp_opt, A2a_opt, A2z_opt, A2ap_opt, A2zp_opt, B1a_opt, B1z_opt, B1ap_opt, B1zp_opt, B2a_opt, B2z_opt, B2ap_opt, B2zp_opt, B3a_opt, B3z_opt, B3ap_opt, B3zp_opt])

file_path = './t' + str(T) + '_from_CHSH.csv'

# Headers for each line
headers = np.array([
    "eta", "keys", "q",
    "A1a", "A1z","A1ap", "A1zp", "A2a", "A2z","A2ap", "A2zp",
    "B1a", "B1z","B1ap", "B1zp", "B2a", "B2z","B2ap", "B2zp",
    "B3a", "B3z","B3ap", "B3zp"
])


# Combine headers and data
output_data = np.hstack((headers[:, None], save_data))

# Save data with headers vertically
# np.savetxt(file_path, output_data, fmt='%.8f', delimiter=',')


"""
Plot
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

# plt.plot(eta_list, keys, color = colors[0], label=R'$t=$'+str(T),linewidth=2)
plt.plot(eta_list, keys, label=R'$T=$'+str(T),linewidth=2)
plt.legend(fontsize=12, prop={'family': 'serif', 'size': 11})
plt.minorticks_on()
# plt.xaxis.set_tick_params(which='minor', bottom=False)
plt.xlabel(R"$\tilde{\eta}_L$", fontdict={'family': 'serif', 'size': 12})
plt.ylabel("r (bits/s)", fontdict={'family': 'serif', 'size': 12})
plt.xticks(fontfamily='serif', fontsize=12)
plt.yticks(fontfamily='serif', fontsize=12)
plt.yscale('log')
plt.ylim(1e-5,1)
plt.xlim(0.80, 1)
# plt.savefig('./Asymptotic.pdf')
plt.show()
