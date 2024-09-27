import gurobipy as gp
from gurobipy import GRB
import scipy.stats as stats
import scipy.optimize as spo
import scipy.integrate as integrate
import math
import numpy as np

import helper_methods as hm

####################################   
# BVAP Base constraints
#################################### 
def add_bvap_vap(m, G, k, U, bounds = None, comparison = False, bvap_ordering = False):
    '''
    Adding bvap and vap parameters and variables.
    '''
    BVAP = {i : int(G.nodes[i]["BVAP"]) for i in G.nodes}
    VAP = {i : int(G.nodes[i]["VAP"]) for i in G.nodes}
    VAP_TOTAL = sum(VAP[i] for i in G.nodes) # the total voting age population, serves as an upper bound
    BVAP_TOTAL = sum(BVAP[i] for i in G.nodes) # the total black voting age population, serves as an upper bound
    
     ## population variables
        # bvap = bvap in district j
        # vap = voting age population in district j
        # Option to add specific bounds rather than a more general bound
    if bounds == None:
        m._vap = {j : m.addVar(name = f"vap{j}", ub = U)   for j in range(k)} 
        m._bvap = {j : m.addVar(name = f"bvap{j}", ub = U)  for j in range(k)} 
    else:
        m._vap = {j : m.addVar(name = f"vap{j}", lb = bounds['vap']['lb'][j], ub = bounds['vap']['ub'][j])   for j in range(k)}
        m._bvap = {j : m.addVar(name = f"bvap{j}", lb = bounds['bvap']['lb'][j], ub = bounds['bvap']['ub'][j]) for j in range(k)} 
    
    
    ##  VOTING AGE POPULATION AND BVAP POPULATION
    m.addConstrs(( sum(VAP[i]*m._X[i,j]    for i in G.nodes) ==  m._vap[j]   for j in range(k)  ), name="VAP_Block_j")
    m.addConstrs(( sum(BVAP[i]*m._X[i,j]   for i in G.nodes) ==  m._bvap[j]  for j in range(k)  ), name="BVAP_Block_j")
    
    # Natural comparison bounds on vap vs bvap
    if comparison:
        m.addConstrs((  m._vap[j]  >=  m._bvap[j]  for j in range(k)  ), name="VAP_BVAP_compare_j")
    
    # Order bvap variables
    if bvap_ordering:
        print("Adding bvap ordering")
        m.addConstrs( (m._bvap[j] <= m._bvap[j+1]  for j in range(k-1)), name='ordering_bvap' )
    
    return BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, m._vap, m._bvap    

def add_bvap_vap_continuous(m, G, k, U, bounds = None, comparison = False, bvap_ordering = False):
    '''
    Adding bvap and vap parameters and variables.
    '''
    BVAP = {i : int(G.nodes[i]["BVAP"]) for i in G.nodes}
    VAP = {i : int(G.nodes[i]["VAP"]) for i in G.nodes}
    VAP_TOTAL = sum(VAP[i] for i in G.nodes) # the total voting age population, serves as an upper bound
    BVAP_TOTAL = sum(BVAP[i] for i in G.nodes) # the total black voting age population, serves as an upper bound
    
     ## population variables
        # bvap = bvap in district j
        # vap = voting age population in district j
        # Option to add specific bounds rather than a more general bound
    if bounds == None:
        vap = {j : m.addVar(name = f"vap{j}", ub = U)   for j in range(k)} 
        bvap = {j : m.addVar(name = f"bvap{j}", ub = U)  for j in range(k)} 
    else:
        vap = {j : m.addVar(name = f"vap{j}", lb = bounds['vap']['lb'][j], ub = bounds['vap']['ub'][j])   for j in range(k)}
        bvap = {j : m.addVar(name = f"bvap{j}", lb = bounds['bvap']['lb'][j], ub = bounds['bvap']['ub'][j]) for j in range(k)} 
    
    
    ##  VOTING AGE POPULATION AND BVAP POPULATION
    m.addConstr( VAP_TOTAL ==  sum(vap[j]   for j in range(k)) , name="VAP_Total")
    m.addConstr(  BVAP_TOTAL ==  sum(bvap[j]  for j in range(k))  , name="BVAP_Total")
    
    # Natural comparison bounds on vap vs bvap
    if comparison:
        m.addConstrs((  vap[j]  >=  bvap[j]  for j in range(k)  ), name="VAP_BVAP_compare_j")
    
    # Order bvap variables
    if bvap_ordering:
        print("Adding bvap ordering")
        m.addConstrs( (bvap[j] <= bvap[j+1]  for j in range(k-1)), name='ordering_bvap' )
    
    return BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap  






####################################   
# Compute BVAP Bounds
#################################### 

def add_bvap_bounds_objective(m, G, k, R,U,bounds,obj_order,index):
    '''
    function that is used to help generate strong bounds on the bvap and vap variables.  
    the bounds are then processed in the file "Observing bounds information.ipynb"
    these bounds are stored in a json in the Validi/data/bounds folder
    these bounds can then be used in any of the formulations to give tighter initial formulations
    '''
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k,U, bvap_ordering = True)
    
    objective_options = {
        'bvap_max': (GRB.MAXIMIZE, 'bvap'),
        'bvap_min': (GRB.MINIMIZE, 'bvap'),
        'vap_max': (GRB.MAXIMIZE, 'vap'),
        'vap_min': (GRB.MINIMIZE, 'vap'),
    }

    if obj_order in objective_options:
        obj_option, var_prefix = objective_options[obj_order]
        obj_variable = locals()[f"{var_prefix}"]
        m.setObjective(obj_variable[index], obj_option)
        

    

####################################   
# BVAP Objectives
#################################### 


## Stepwise supporting functions
def add_step_simple(m, vap, bvap, VAP_TOTAL, cdf, delta, RATIO_BREAKPOINTS,CDF_VALUES, PWL_PARTS, k):
    '''
    This function is applied to multiple stepwise objectives.
    It takes in information about the breakpoints of interest and the needed variables and implements the simple threshold constraints on the objective function.
    This version includes the constraints that order the delta variables within a district.
    This version also assumes that the breakpoints for each district are the same.
    '''

    
    ## Decide value of cdf[j]
    ## We focus on the fact that the cdf is an increasing function
    ## We model  "if ratio <= RATIO_BREAKPOINT, then cdf <= CDF_VALUE@RATIO_BREAKPOINT"
    ## For this, we use an indicator variable \delta, where \delta = 1 means ratio <= RATIO_BREAKPOINT
    
    ### if bvap/vap <= ratio, then delta = 1
    ### --->  if bvap - ratio*vap <= 0, then delta = 1
    ### --->  if ratio*vap - bvap > 0, then delta = 1
    ### --->  ratio*vap - bvap <= M*delta
    m.addConstrs(( RATIO_BREAKPOINTS[l]*vap[j] - bvap[j] <= VAP_TOTAL*delta[j][l] for j in range(k)  for l in range(PWL_PARTS+1)  ), name="breakpoint_bound")
    
    ### if delta = 1, then cdf_j <= CDF_VALUE: (note: 1 acts as the big M here)
    m.addConstrs(( cdf[j] <= CDF_VALUES[l] + (1-delta[j][l])   for j in range(k)  for l in range(PWL_PARTS+1)  ), name="cdf_bound")
    
    ### delta variables are ordered (i.e., if ratio[j] <= RATIO_BREAKPOINT[l], then ratio[j] <= RATIO_BREAKPOINT[l+1])
    ### thus, if \delta[l] = 1, then \delta[l+1] = 1
    m.addConstrs(( delta[j][l] <= delta[j][l+1]  for j in range(k)  for l in range(PWL_PARTS)  ), name="cdf_bound")


def add_step_improved(m, cdf, delta, RATIO_BREAKPOINTS,CDF_VALUES, CDFMAX,CDF_DELTAS, PWL_PARTS, k):
    '''
    this is an improved formulation of the step function that gives a tighter convexification using fewer inequalities.
    '''
    ## Decide value of cdf[j]
    m.addConstrs(( RATIO_BREAKPOINTS[l]*vap[j] - bvap[j] <=  VAP_TOTAL*delta[j][l] for j in range(k)  for l in range(PWL_PARTS+1)  ), name="breakpoint_bound")
    
    ### if delta = 1, then cdf_j <= CDF_VALUE: (note: 1 acts as the big M here)
    m.addConstrs(( cdf[j] <= CDFMAX-sum(CDF_DELTAS[l]*delta[j][l]  for l in range(len(CDF_VALUES)-1)) for j in range(k) ), name="cdf_bound")
    
    ### delta variables are ordered (i.e., if ratio[j] <= RATIO_BREAKPOINT[l], then ratio[j] <= RATIO_BREAKPOINT[l+1])
    m.addConstrs(( delta[j][l] <= delta[j][l+1]  for j in range(k)  for l in range(PWL_PARTS)  ), name="cdf_bound")
    

# Stepwise objective functions
def add_step_exp_objective(m, G, k, R,U, bounds = None):
    '''
    Stepwise objective using breakpoints generated from minimizing the expected value of the error assuming uniformly sampled values.
    '''
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k,U)
    
    # Calculate breakpoints
    node_ratios = [BVAP[i]/VAP[i]  for i in G.nodes]
    inf = min(node_ratios)
    sup = max(node_ratios)
    
    def rho(r):
        return -hm.pdf_fun(r).sum()
    def grad(r):
        return -hm.pdf_fun(r)
    
    RATIO_BREAKPOINTS, PWL_PARTS, CDF_VALUES, maxErr = hm.calculate_breakpoints_exp(inf, sup, R, hm.cdf_fun, rho, grad)
    
    # variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    delta = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"cdf{j},{l}")  for l in range(PWL_PARTS+1)}  for j in range(k)}
    
    add_step_simple(m, vap, bvap,  VAP_TOTAL, cdf, delta, RATIO_BREAKPOINTS,CDF_VALUES, PWL_PARTS, k)
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)
    return maxErr#(expErr, maxErr)


def add_step_max_objective(m, G, k, R,U, bounds = None):
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k,U)
    
    # Calculate breakpoints
    node_ratios = [BVAP[i]/VAP[i]  for i in G.nodes]
    inf = min(node_ratios)
    sup = max(node_ratios)
    
    RATIO_BREAKPOINTS, PWL_PARTS, CDF_VALUES, maxErr =  hm.calculate_breakpoints_max(inf, sup, R, hm.cdf_fun)
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} 
    delta = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"cdf{j},{l}")  for l in range(PWL_PARTS+1)}  for j in range(k)}
    
    add_step_simple(m, vap, bvap,  VAP_TOTAL, cdf, delta, RATIO_BREAKPOINTS,CDF_VALUES, PWL_PARTS, k)
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)
    return maxErr#(expErr, maxErr)

def add_step_alt_objective(m, G, k, R, U):
    '''
    This alternative stepwise formulation uses the improved inequalities that give a tighter bound on the convex hull and actually using fewer inequalities.
    '''
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k,U)
    
    node_ratios = [BVAP[i]/VAP[i]  for i in G.nodes]
    inf = min(node_ratios)
    sup = max(node_ratios)
    
    RATIO_BREAKPOINTS, PWL_PARTS, CDF_VALUES, CDFMAX, CDF_DELTAS, maxErr = hm.calculate_breakpoints_alt(inf, sup,R,hm.cdf_fun, hm.rho, hm.grad)
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    delta = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"cdf{j},{l}")  for l in range(PWL_PARTS+1)}  for j in range(k)}
    
    # MODELING THE OBJECTIVE
    add_step_improved(m, cdf, delta, RATIO_BREAKPOINTS,CDF_VALUES, CDFMAX,CDF_DELTAS, PWL_PARTS, k)
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)
    return (expErr, maxErr)

# LogE formulations
def add_LogEPWL_objective(m,G,k,L,U, bounds = None):
    '''
    Our implementation of the LogE piecewise linear formulation.   This uses only logarithmically many binary variables with respect to the number of piecewise linear sections L.
    This implementation is sensitive to the Rinner and Router values.
    '''
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U) 
    
   
    Rinner = min([math.sqrt(BVAP[i]**2+VAP[i]**2)  for i in G.nodes]) # inner radius
    Router = math.sqrt(BVAP_TOTAL**2+U**2) # outer radius
    
    node_ratios = [BVAP[i]/VAP[i]  for i in G.nodes]
    l = min(node_ratios)  # lower bound on ratio
    u = max(node_ratios)   # upper bound on ratio
    

    factor = Router/Rinner
    step = (u-l)/L

    myRatios = np.arange(l, u, step).tolist()

    B = [[math.sqrt(Rinner**2/(1+r**2)), math.sqrt(Rinner**2 - Rinner**2/(1+r**2))] for r in myRatios] # set of inner radius vertices
    
    nu = math.ceil(math.log(len(myRatios),2))
    code = hm.generateGrayarr(nu)
    S = hm.summation_sets(code, nu,L-1)

    D = range(k) # set of districts

    lam = m.addVars(D,range(L), ['in','out'], name = 'lam')
    delta = m.addVars(D,range(nu), vtype = GRB.BINARY, name = 'delta')
    
    y = vap 
    z = bvap
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    
    f = cdf
    

    m.addConstrs(sum(lam[d,i,j] for j in ['in','out'] for i in range(L)) == 1 for d in D)

    for d in D:
        #m.addSOS(GRB.SOS_TYPE2, [delta[d,i] for i in range(L)])
        for q in range(nu):
            m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in S[q][1]) <= 1-delta[d,q])
            m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in S[q][0]) <= delta[d,q])

    m.addConstrs(z[d] == sum(lam[d,i,'in']*B[i][1] for i in range(L)) + sum(lam[d,i,'out']*B[i][1]*factor for i in range(L)) for d in D)
    m.addConstrs(y[d] == sum(lam[d,i,'in']*B[i][0] for i in range(L)) + sum(lam[d,i,'out']*B[i][0]*factor for i in range(L)) for d in D)


    m.addConstrs(f[d] == sum((lam[d,i,'in']+lam[d,i,'out'])*hm.calculate_black_rep(B[i][1]/B[i][0]) for i in range(L)) for d in D)
    
    # set objective
    m.setObjective(sum(f[k] for k in D), sense=GRB.MAXIMIZE)

    
def add_LogEPWL_objective_extra_bounds_old(m,G,k,L,U, bounds):
    '''
    Perhaps an improved implementation of the LogE formulation. 
    This version allows for improved bounds to be usd for each of the districts based on an ordering of the bvap variables
    '''
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U, bounds = bounds, bvap_ordering = True)
    
    
#     # Special to SC and testing if added inequalities help
#     # This is the gradient at the optimal solution
#     grad_y = [1.97338363, 2.09482622, 2.60365293, 2.0817186, 2.78971841,
#         5.67583369, 5.78411099]
#     grad_z = [-0.34014631, -0.37274136, -0.52628305, -0.39175563, -0.61444055,
#         -1.96069322, -2.03916219]
    
#     m.addConstr(sum(bvap[i]*grad_y[i] + vap[i]*grad_z[i] for i in range(k)) <= 3.508319191561e+05)
    
#     #m.addConstr(sum(bvap[i]*grad_y[i] + vap[i]*grad_z[i] for i in range(k)) >= -1)
    
#     m.addConstr(bvap[5] + bvap[6] - bvap[1] - bvap[2] <= 3.110470000000e+05)
    
#     #m.addConstr(bvap[5] + bvap[6] <= 4.899240000000e+05)
    
#     m.addConstr(bvap[5] + bvap[6] <= 440101.0)

    # write out bounds in different variables for later usage
    vap_lb, vap_ub, bvap_lb, bvap_ub = bounds['vap']['lb'], bounds['vap']['ub'],bounds['bvap']['lb'],bounds['bvap']['ub']
    
     ### Some value values for SC
    nu = math.ceil(math.log(L,2))
    code = hm.generateGrayarr(nu)
    S = hm.summation_sets(code, nu,L-1)
    
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} 
    lam = m.addVars(range(k),range(L), ['in','out'], name = 'lam')
    delta = m.addVars(range(k),range(nu), vtype = GRB.BINARY, name = 'delta')

    # Convex combination multipliers sum to 1
    m.addConstrs(sum(lam[d,i,j] for j in ['in','out'] for i in range(L)) == 1 for d in range(k))

    for d in range(k):
        for q in range(nu):
            m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in S[q][1]) <= 1-delta[d,q])
            m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in S[q][0]) <= delta[d,q])
            
            
    import matplotlib.pyplot as plt
    # add breakpoints specific to each district based on the ordering of bvap in each district.
    for d in range(k): 
        Rinner = math.sqrt(vap_lb[d]**2+bvap_lb[d]**2) # inner radius
        Router = math.sqrt(vap_ub[d]**2+bvap_ub[d]**2) # outer radius

        l = bvap_lb[d]/vap_ub[d]
        u = bvap_ub[d]/vap_lb[d]

        factor = Router/Rinner
        step = (u-l)/L
        
        #myRatios = hm.calculate_breakpoints_max(l, u, L, hm.cdf_fun)[0]
        #print(hm.calculate_breakpoints_max(l, u, L, hm.cdf_fun))
        breakpoints  = hm.calculate_breakpoints_max_error(l, u, 0.001, hm.cdf_fun)
        myRatios = breakpoints[0]
        L = breakpoints[1]
        maxErr = breakpoints[3]
        print(breakpoints)
        #myRatios = np.arange(l, u, step).tolist()
        
        print("District ", d)
        print(myRatios)

        B = [[math.sqrt(Rinner**2/(1+r**2)), math.sqrt(Rinner**2 - Rinner**2/(1+r**2))] for r in myRatios] # set of inner radius vertices
        Binner = []
        Bouter = []
        #print(B)
        
                # Loop over each ratio in the list
        for i in range(len(myRatios)):
            r = myRatios[i]
            x, y = B[i]

            # Plot the points and the scaled points
            plt.plot(x, y, 'bo')
            plt.plot(factor*x, factor*y, 'bo')
            plt.plot([x,factor*x], [y,factor*y], 'b')
            
            p1, p2 = hm.liang_barsky(x, y, factor*x, factor*y, vap_lb[d], vap_ub[d], bvap_lb[d], bvap_ub[d])
            Binner.append(p1)
            Bouter.append(p2)
            print(r,p1,p2)
            plt.plot(p1[0],p1[1], 'ro')
            plt.plot(p2[0], p2[1], 'ro')
            plt.plot([p1[0],p2[0]], [p1[1],p2[1]], 'r')
            
            # Connect the points with lines
            if i > 0:
                x_prev, y_prev = B[i-1]
                plt.plot([x_prev, x], [y_prev, y], 'b-')
                plt.plot([factor*x_prev, factor*x], [factor*y_prev, factor*y], 'b-')

        # Set axis labels and title
        plt.xlabel(f'vap{d}')
        plt.ylabel(f'bvap{d}')
        plt.title(f'Plot of Points and Scaled Points for district {d}, L = {L}, maxErr = {maxErr}')
        # Set the limits of the x and y axes
        # Set the limits of the x and y axes
        plt.xlim(min(vap_lb.values()), max(vap_ub.values()))
        plt.ylim(min(bvap_lb.values()), max(bvap_ub.values()))

        # Get the limits of the x and y axes
        x0, x1 = plt.xlim()
        y0, y1 = plt.ylim()

        # Plot the box
        plt.plot([x0, x1], [y0, y0], 'r--')
        plt.plot([x1, x1], [y0, y1], 'r--')
        plt.plot([x1, x0], [y1, y1], 'r--')
        plt.plot([x0, x0], [y1, y0], 'r--')
        
       # Get the limits of the x and y axes
        x0, x1 = vap_lb[d], vap_ub[d]
        y0, y1 = bvap_lb[d], bvap_ub[d]

        # Plot the box
        plt.plot([x0, x1], [y0, y0], 'g--')
        plt.plot([x1, x1], [y0, y1], 'g--')
        plt.plot([x1, x0], [y1, y1], 'g--')
        plt.plot([x0, x0], [y1, y0], 'g--')
        
        plt.xlim(vap_lb[0]*0.9, vap_ub[k-1]*1.05)
        plt.ylim(bvap_lb[0]*0.6, bvap_ub[k-1]*1.05)
        
        
        
        
        # Show the plot
        plt.show()
        
        print(Binner)
        print(Bouter)
        
        # uses rays intersecting the bounds
        m.addConstr(bvap[d] == sum(lam[d,i,'in']*Binner[i][1] for i in range(L)) + sum(lam[d,i,'out']*Bouter[i][1] for i in range(L)))
        m.addConstr(vap[d] == sum(lam[d,i,'in']*Binner[i][0] for i in range(L)) + sum(lam[d,i,'out']*Bouter[i][0] for i in range(L)))
        m.addConstr(cdf[d] == sum((lam[d,i,'in']+lam[d,i,'out'])*hm.calculate_black_rep(B[i][1]/B[i][0]) for i in range(L)))
        
        # old - uses full rays, even if they are far away from bounds.
#         m.addConstr(bvap[d] == sum(lam[d,i,'in']*B[i][1] for i in range(L)) + sum(lam[d,i,'out']*B[i][1]*factor for i in range(L)))
#         m.addConstr(vap[d] == sum(lam[d,i,'in']*B[i][0] for i in range(L)) + sum(lam[d,i,'out']*B[i][0]*factor for i in range(L)))
#         m.addConstr(cdf[d] == sum((lam[d,i,'in']+lam[d,i,'out'])*hm.calculate_black_rep(B[i][1]/B[i][0]) for i in range(L)))

    # set objective
    m.setObjective(sum(cdf[d] for d in range(k)), sense=GRB.MAXIMIZE)
    
def add_LogEPWL_objective_extra_bounds_great(m,G,k,L,U, bounds):
    '''
    Perhaps an improved implementation of the LogE formulation. 
    This version allows for improved bounds to be usd for each of the districts based on an ordering of the bvap variables
    '''
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U, bounds = bounds, bvap_ordering = True)

    # write out bounds in different variables for later usage
    vap_lb, vap_ub, bvap_lb, bvap_ub = bounds['vap']['lb'], bounds['vap']['ub'],bounds['bvap']['lb'],bounds['bvap']['ub']
    

    
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} 
    lam = {}#m.addVars(range(k),range(L), ['in','out'], name = 'lam')
    delta = {}#m.addVars(range(k),range(nu), vtype = GRB.BINARY, name = 'delta')
            
            
    import matplotlib.pyplot as plt
    # add breakpoints specific to each district based on the ordering of bvap in each district.
    for d in range(k): 
        Rinner = math.sqrt(vap_lb[d]**2+bvap_lb[d]**2) # inner radius
        Router = math.sqrt(vap_ub[d]**2+bvap_ub[d]**2) # outer radius

        l = bvap_lb[d]/vap_ub[d]
        u = bvap_ub[d]/vap_lb[d]

        factor = Router/Rinner
        step = (u-l)/L
        
        #myRatios = hm.calculate_breakpoints_max(l, u, L, hm.cdf_fun)[0]
        #print(hm.calculate_breakpoints_max(l, u, L, hm.cdf_fun))
        breakpoints  = hm.calculate_breakpoints_max_error(l, u, 0.01, hm.cdf_fun)
        myRatios = breakpoints[0]
        L = breakpoints[1]
        maxErr = breakpoints[3]
        
        ### Some value values for SC
        nu = math.ceil(math.log(L,2))
        code = hm.generateGrayarr(nu)
        S = hm.summation_sets(code, nu,L-1)
        
        #print(breakpoints)
        #myRatios = np.arange(l, u, step).tolist()
        
        for j in ['in','out']:
            for i in range(L):
                lam[(d,i,j)] = m.addVar(name = f'lam{d,i,j}', lb = 0, ub = 1)
        for q in range(nu):
            delta[(d,q)] = m.addVar( vtype = GRB.BINARY, name = f'delta{d,q}')

        # Convex combination multipliers sum to 1
        m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in range(L)) == 1)
        
        for q in range(nu):
            m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in S[q][1]) <= 1-delta[d,q])
            m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in S[q][0]) <= delta[d,q])
        
        #print("District ", d)
        #print(myRatios)

        B = [[math.sqrt(Rinner**2/(1+r**2)), math.sqrt(Rinner**2 - Rinner**2/(1+r**2))] for r in myRatios] # set of inner radius vertices
        Binner = []
        Bouter = []
        #print(B)
        
        # Loop over each ratio in the list
        for i in range(len(myRatios)):
            r = myRatios[i]
            x, y = B[i]

            # Plot the points and the scaled points
            plt.plot(x, y, 'bo')
            plt.plot(factor*x, factor*y, 'bo')
            plt.plot([x,factor*x], [y,factor*y], 'b')
            
            p1, p2 = hm.liang_barsky(x, y, factor*x, factor*y, vap_lb[d], vap_ub[d], bvap_lb[d], bvap_ub[d])
            Binner.append(p1)
            Bouter.append(p2)
            #print(r,p1,p2)
            plt.plot(p1[0],p1[1], 'ro')
            plt.plot(p2[0], p2[1], 'ro')
            plt.plot([p1[0],p2[0]], [p1[1],p2[1]], 'r')
            
            # Connect the points with lines
            if i > 0:
                x_prev, y_prev = B[i-1]
                plt.plot([x_prev, x], [y_prev, y], 'b-')
                plt.plot([factor*x_prev, factor*x], [factor*y_prev, factor*y], 'b-')

        # Set axis labels and title
        plt.xlabel(f'vap{d}')
        plt.ylabel(f'bvap{d}')
        plt.title(f'Plot of Points and Scaled Points for district {d}, L = {L}, maxErr = {maxErr}')
        # Set the limits of the x and y axes
        # Set the limits of the x and y axes
        plt.xlim(min(vap_lb.values()), max(vap_ub.values()))
        plt.ylim(min(bvap_lb.values()), max(bvap_ub.values()))

        # Get the limits of the x and y axes
        x0, x1 = plt.xlim()
        y0, y1 = plt.ylim()

        # Plot the box
        plt.plot([x0, x1], [y0, y0], 'r--')
        plt.plot([x1, x1], [y0, y1], 'r--')
        plt.plot([x1, x0], [y1, y1], 'r--')
        plt.plot([x0, x0], [y1, y0], 'r--')
        
       # Get the limits of the x and y axes
        x0, x1 = vap_lb[d], vap_ub[d]
        y0, y1 = bvap_lb[d], bvap_ub[d]

        # Plot the box
        plt.plot([x0, x1], [y0, y0], 'g--')
        plt.plot([x1, x1], [y0, y1], 'g--')
        plt.plot([x1, x0], [y1, y1], 'g--')
        plt.plot([x0, x0], [y1, y0], 'g--')
        
        plt.xlim(vap_lb[0]*0.9, vap_ub[k-1]*1.05)
        plt.ylim(bvap_lb[0]*0.6, bvap_ub[k-1]*1.05)
        
        
        
        
        # Show the plot
        plt.show()
        
        #print(Binner)
        #print(Bouter)
        
        # uses rays intersecting the bounds
        print("d = ", d, " L=", L)
        m.addConstr(bvap[d] == sum(lam[d,i,'in']*Binner[i][1] for i in range(L)) + sum(lam[d,i,'out']*Bouter[i][1] for i in range(L)))
        m.addConstr(vap[d] == sum(lam[d,i,'in']*Binner[i][0] for i in range(L)) + sum(lam[d,i,'out']*Bouter[i][0] for i in range(L)))
        m.addConstr(cdf[d] == sum((lam[d,i,'in']+lam[d,i,'out'])*hm.calculate_black_rep(B[i][1]/B[i][0]) for i in range(L)))
        
        # old - uses full rays, even if they are far away from bounds.
#         m.addConstr(bvap[d] == sum(lam[d,i,'in']*B[i][1] for i in range(L)) + sum(lam[d,i,'out']*B[i][1]*factor for i in range(L)))
#         m.addConstr(vap[d] == sum(lam[d,i,'in']*B[i][0] for i in range(L)) + sum(lam[d,i,'out']*B[i][0]*factor for i in range(L)))
#         m.addConstr(cdf[d] == sum((lam[d,i,'in']+lam[d,i,'out'])*hm.calculate_black_rep(B[i][1]/B[i][0]) for i in range(L)))

    # set objective
    m.setObjective(sum(cdf[d] for d in range(k)), sense=GRB.MAXIMIZE)    



# BNPWL Forumulation
def add_BNPWL_objective(m, G, K, R, U):
    '''
    
    '''
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U)
    
    BIGNORM = math.sqrt(BVAP_TOTAL**2+U**2)
    node_norms = [math.sqrt(BVAP[i]**2+VAP[i]**2)  for i in G.nodes]
    
    def cos(j):
        return round(math.cos(math.pi/(2**(j+2))),12) # 0th rotation is 45 degrees
    def sin(j):
        return round(math.sin(math.pi/(2**(j+2))),12) # 0th rotation is 45 degrees
    def tan(j):
        return math.tan(math.pi/(2**j))
    norm = stats.norm(0,1)

    # def rot(theta):
    #     return np.matrix([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    
    def my_func(i,nu=3):
        '''
        function value 
        '''
        n = 2**(nu+2)
        j = 2**nu - i - 1
        value1 = math.cos( (j+1) * (math.pi) / n )
        value2 = math.sin( (j+1) * (math.pi) / n )
        return f_bvap(value2/value1)
    
    '''
    Input:  coordinates y_value,z_value >= 0, 
            nu in Z_+   -  the number of layers of the Ben Tal - Nemirovski approximation to use
    Output:  Gray code of length nu that identifies the slice containing (y_value,z_value)
            (xi, eta) values that are the transformations of (y_value,z_value)
            
    This function is build with GUROBI to solve equaitons.   
    It models the Ben Tal - Nemirovski LP approximation of the SOCP, but it adds binary variables 
    that track when the rotation needs to be flipped back to the positive orthant. 
    This then identifies the slice if the unit circle that contains (y_value,z_value)
    
    Set so that the initial rotation is 45 degrees, then decreases by half.  This is assuming that y_value \leq z_value
    
    '''
    nu = R
    gray = hm.generateGrayarr(nu)
    
    norm_bound = BIGNORM
    F_bound = 100
    inner = min(node_norms)
    outer = norm_bound
    
    # Code with environment here suppresses gurobi outputs
            
    # Build model m
    
    # Sets
    J = list(range(nu+1))
    J1 = list(range(1,nu+1))
    D = range(K)
    # Bounds
    lb_eta = -norm_bound
    ub_eta = norm_bound
    
    # Variables
    y = vap
    z = bvap
    f = m.addVars(D, vtype=GRB.CONTINUOUS, name="f")
    l = m.addVars(D,range(4), lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name="l")
    xi = m.addVars(D,J, lb = 0, vtype=GRB.CONTINUOUS, name="xi")
    eta_rhs = m.addVars(D,J, lb = lb_eta, ub = ub_eta, vtype=GRB.CONTINUOUS, name="eta_rhs")
    eta = m.addVars(D,J, lb = 0, ub = ub_eta, vtype=GRB.CONTINUOUS, name="eta")
    delta = m.addVars(D,J1, vtype=GRB.BINARY, name="delta")

    
    # set objective
    m.setObjective(sum(f[k] for k in D), sense=GRB.MAXIMIZE)
    
    for k in D:
        # con: initializations
        m.addConstr(xi[k,0] == y[k])
        m.addConstr(eta[k,0] == z[k])
        m.addConstr(eta_rhs[k,0] == z[k])

        # con: Transformation  (xi, eta_rhs) = Rot(theta)*(xi, eta)   
        m.addConstrs(xi[k,j] == cos(j)*xi[k,j - 1] + sin(j)*eta[k,j - 1] for j in J1)
        m.addConstrs(eta_rhs[k,j] == -1*sin(j)*xi[k,j - 1] + cos(j)*eta[k,j - 1] for j in J1)

        # con: eta = |eta_rhs|    (This could be done in a better way with lambdas.  See Hongbo's paper)
        ##    Lower bounds
        m.addConstrs(eta[k,j] >= eta_rhs[k,j] for j in J1)
        m.addConstrs(eta[k,j] >= -1*eta_rhs[k,j] for j in J1)

        ##    Upper bounds (assigns the binary variable z which counts flips)
        m.addConstrs(eta[k,j] <= -eta_rhs[k,j] + ub_eta*(1-delta[k,j]) for j in J1)
        m.addConstrs(eta[k,j] <= eta_rhs[k,j] + ub_eta*(delta[k,j]) for j in J1)

        # z = 1 if eta_rhs >= 0,   
        # z = 0 if eta_rhs < 0
        m.addConstrs(ub_eta*(1-delta[k,j]) >= eta_rhs[k,j] for j in J1)
        m.addConstrs(lb_eta*(delta[k,j]) <= eta_rhs[k,j] for j in J1)


        # con: bound xi[nu], which should now be nearly the norm of (x1, x2) for large enough nu
        m.addConstr(xi[k,nu] <= norm_bound)

        # con: bound eta[nu] based on xi[nu]
        m.addConstr(eta[k,nu] <= tan(nu)*xi[k,nu])

        # add inequaltiies on objective function
        ## we assume that f is an increasing function of y/z, and that we are maximizing f, so we only need upper bounds

        m.addConstr(sum(l[k,j] for j in range(4)) == 1)

        # single convex combination constraint
        #(xi[nu], eta[nu]) =  sum(l[j]v[j] for j in range(4))
        n = 2**(nu+2)
        x1 = math.cos( (0) * (math.pi) / n )
        y1 = math.sin( (0) * (math.pi) / n ) 
        x2 = math.cos( (1) * (math.pi) / n )
        y2 = math.sin( (1) * (math.pi) / n ) 


        m.addConstr(xi[k,nu] == l[k,0]*outer*x1 + l[k,1]*inner*x1 + l[k,2]*outer*x2 + l[k,3]*inner*x2)
        m.addConstr(eta[k,nu] == l[k,0]*outer*y1 + l[k,1]*inner*y1 + l[k,2]*outer*y2 + l[k,3]*inner*y2)

        for i,grayCode in enumerate(gray):
            if (i % 2) == 0:
                  f_value = l[k,0]*my_func(i+1,nu) + l[k,1]*my_func(i+1,nu) + l[k,2]*my_func(i,nu) + l[k,3]*my_func(i,nu)
            else:
                  f_value = l[k,0]*my_func(i,nu) + l[k,1]*my_func(i,nu) + l[k,2]*my_func(i+1,nu) + l[k,3]*my_func(i+1,nu)
            s = bin(i)[2:].zfill(nu)
            t = len(s) - len(s.rstrip('0'))
            J_i = range(1,nu+1-t)
            m.addConstr(f[k] <= f_value + F_bound*sum(delta[k,j] for j in J_i if grayCode[j-1] < 0.1) + F_bound*sum(grayCode[j-1]- delta[k,j] for j in J_i if grayCode[j-1] > 0.9), f'label{i}')







####
# Old bvap formulations?
####
def add_disctrete_PWL_objective(m, G, k, R, U):
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U)
    
    node_ratios = [BVAP[i]/VAP[i]  for i in G.nodes]
    inf = min(node_ratios)
    sup = max(node_ratios)
    
    vstep = (cdf_fun(sup)-cdf_fun(inf))/R
    CDF_VALUES = [cdf_fun(inf)+vstep*i  for i in range(R+1)]
    
    RATIO_BREAKPOINTS = []
    for v in CDF_VALUES:
        def func(x):
            return cdf_fun(x)-v
        RATIO_BREAKPOINTS += [spo.fsolve(func,.5)[0]]
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    
    
    L = 10
    y = m.addVars(k, L, vtype=GRB.CONTINUOUS, lb=0, name='y')
    m.addConstrs( bvap[j] >= sum(2**(-l-1)*y[j,l]  for l in range(L))  for j in range(k))
    
    rho = m.addVars(k, L, vtype=GRB.BINARY, name='rho')
    m.addConstrs( y[j,l] <= rho[j,l]*VAP_TOTAL               for j in range(k)  for l in range(L))
    m.addConstrs( y[j,l] <= vap[j]                           for j in range(k)  for l in range(L))
    m.addConstrs( y[j,l] >= vap[j]-(1-rho[j,l])*VAP_TOTAL    for j in range(k)  for l in range(L))
    
    #r = {j : m.addVar(lb = 0, ub = 1000, name = f"r{j}", vtype=GRB.INTEGER)   for j in range(k)}
    ratio = {j : m.addVar(lb = 0, ub = 1, name = f"ratio{j}", vtype=GRB.CONTINUOUS)   for j in range(k)}
    m.addConstrs(ratio[j] <= sum(2**(-l-1)*rho[j,l]  for l in range(5))  for j in range(k))
    
    #m.addConstrs( bvap[j] >= vap[j]*ratio[j]   for j in range(k))
    #m.params.NonConvex = 2
    
    #m.setAttr("BranchPriority", r, 1)
    for j in range(k):
        m.addGenConstrPWL( ratio[j], cdf[j], RATIO_BREAKPOINTS, CDF_VALUES)
    
    #m.addConstrs( ratio[j] <= ratio[j+1]  for j in range(k-1) )
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)



def add_step_ordered_objective(m, G, k, R,U):
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U, bvap_ordering = True)
    
    node_ratios = [BVAP[i]/VAP[i]  for i in G.nodes]
    inf = min(node_ratios)
    sup = max(node_ratios)
    
    print("inf and sup ratios: " + str(inf) + ", " + str(sup))
    r_initial = [ inf+i*(sup-inf)/R  for i in range(R+1) ]
    
    cons = ({ 'type':'eq', 'fun': lambda r: r[-1] - r_initial[-1] }, { 'type':'eq', 'fun': lambda r: r[0] - r_initial[0] })
    
    result = spo.minimize(  hm.rho, r_initial, constraints=cons, jac=hm.grad, tol=.00000001  )
    
    integral = integrate.quad(  hm.cdf_fun, r_initial[0], r_initial[-1]  )[0]
    
    
    if result.success:
        print( 'Breakpoints successfully generated!' )
        print( f'Breakpoints = {result.x}' )
        expErr = (result.fun-integral)/(r_initial[-1]-r_initial[0])
        print( f'ExpectError = {expErr}' )
    else:
        print('Breakpoints failed to generate!')
    RATIO_BREAKPOINTS = result.x
    
    
    PWL_PARTS = len(RATIO_BREAKPOINTS)-1
    CDF_VALUES = [cdf_fun(r)  for r in RATIO_BREAKPOINTS] # function values at breakpoints
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    delta = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta{j},{l}")  for l in range(PWL_PARTS+1)}  for j in range(k)}
    
    
    ## Decide value of cdf[j]
    ## We focus on the fact that the cdf is an increasing function
    ## We model  "if ratio <= RATIO_BREAKPOINT, then cdf <= CDF_VALUE@RATIO_BREAKPOINT"
    ## For this, we use an indicator variable \delta, where \delta = 1 means ratio <= RATIO_BREAKPOINT
    
    ### if bvap/vap <= ratio, then delta = 1
    ### --->  if bvap - ratio*vap <= 0, then delta = 1
    ### --->  if ratio*vap - bvap > 0, then delta = 1
    ### --->  ratio*vap - bvap <= M*delta
    #m.addConstrs(( RATIO_BREAKPOINTS[l]*vap[j] - bvap[j] <= VAP_TOTAL*delta[j][l]  for j in range(k)  for l in range(PWL_PARTS+1)  ), name="breakpoint_bound")
    m.addConstrs(( RATIO_BREAKPOINTS[l]*vap[j] - bvap[j] <= U*delta[j][l]  for j in range(k)  for l in range(PWL_PARTS+1)  ), name="breakpoint_bound")
    print(VAP_TOTAL)
    ### if delta = 1, then cdf_j <= CDF_VALUE: (note: 1 acts as the big M here)
    #m.addConstrs(( cdf[j] <= CDF_VALUES[j][PWL_PARTS] - sum((CDF_VALUES[j][l+1]-CDF_VALUES[j][l])*delta[j][l]  for l in range(PWL_PARTS))   for j in range(k)    ), name="cdf_bound_awesome")
    m.addConstrs(( cdf[j] <= CDF_VALUES[PWL_PARTS] - sum((CDF_VALUES[l+1]-CDF_VALUES[l])*delta[j][l]  for l in range(PWL_PARTS))   for j in range(k)    ), name="cdf_bound_awesome")
    
    #m.addConstrs(( cdf[j] <= CDF_VALUES[l] + (1-delta[j][l])   for j in range(k)  for l in range(PWL_PARTS+1)  ), name="cdf_bound")
    
    ### delta variables are ordered (i.e., if ratio[j] <= RATIO_BREAKPOINT[l], then ratio[j] <= RATIO_BREAKPOINT[l+1])
    ### thus, if \delta[l] = 1, then \delta[l+1] = 1
    m.addConstrs(( delta[j][l] <= delta[j][l+1]  for j in range(k)  for l in range(PWL_PARTS)  ), name="ordering_delta_for_district")
    
    
  

    
    #m.addConstrs(( delta[j][l] >= delta[j+1][l]  for j in range(k-1)  for l in range(PWL_PARTS+1)  ), name="ordering_delta_for_objective")
    
    # This applies when breakpoints are the same for each district.
    #m.addConstrs(( sum(delta[j][l] for l in range(PWL_PARTS+1))   >= sum(delta[j+1][l] for l in range(PWL_PARTS+1) ) for j in range(k-1)   ), name="ordering_delta_for_objective")
    
    m.Params.SolFiles = 'ordering test'
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)
    m.setParam("NonConvex", 2)

def add_cumulative_objective(m, G, k, U):
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U)

    
    PWL_PARTS = len(RATIO_BREAKPOINTS)-1
    CDF_VALUES = [cdf_fun(r)  for r in RATIO_BREAKPOINTS] # function values at breakpoints
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    delta = {j : {l : m.addVar(vtype = GRB.BINARY)  for l in range(PWL_PARTS+1)}  for j in range(k)}

    ## Decide value of cdf[j]
    ## We focus on the fact that the cdf is an increasing function
    ## We model  "if ratio <= RATIO_BREAKPOINT, then cdf <= CDF_VALUE@RATIO_BREAKPOINT"
    ## For this, we use an indicator variable \delta, where \delta = 1 means ratio <= RATIO_BREAKPOINT
    
    ### if bvap/vap <= ratio, then delta = 1
    ### --->  if bvap - ratio*vap <= 0, then delta = 1
    ### --->  if ratio*vap - bvap > 0, then delta = 1
    ### --->  ratio*vap - bvap <= M*delta
    m.addConstrs(( RATIO_BREAKPOINTS[l]*vap[j] - bvap[j] <=  VAP_TOTAL*delta[j][l] for j in range(k)  for l in range(PWL_PARTS+1)  ), name="breakpoint_bound")
    
    ### cdf_j <= sum \Delta CDF_VALUE * \delta_{j,l} : (note: no big M formulation here.  Also, could try equality as well as an approximation)
    m.addConstrs(( cdf[j] <= CDF_VALUES[0] + sum((1-delta[j][l])*(CDF_VALUES[l+1]-CDF_VALUES[l])   for l in range(PWL_PARTS)  ) for j in range(k) ), name="cdf_bound")
    
    ### delta variables are ordered (i.e., if ratio[j] <= RATIO_BREAKPOINT[l], then ratio[j] <= RATIO_BREAKPOINT[l+1])
    ### thus, if \delta[l] = 1, then \delta[l+1] = 1
    m.addConstrs(( delta[j][l] <= delta[j][l+1]  for j in range(k)  for l in range(PWL_PARTS)  ), name="cdf_bound")
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)


def add_PWL_objective(m, G, k, R, U):
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U, bvap_ordering = True)
    
    node_ratios = [BVAP[i]/VAP[i]  for i in G.nodes]
    inf = min(node_ratios)
    sup = max(node_ratios)
    
    vstep = (cdf_fun(sup)-cdf_fun(inf))/R
    CDF_VALUES = [cdf_fun(inf)+vstep*i  for i in range(R+1)]
    
    RATIO_BREAKPOINTS = []
    for v in CDF_VALUES:
        def func(x):
            return cdf_fun(x)-v
        RATIO_BREAKPOINTS += [spo.fsolve(func,.5)[0]]
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)

    
    r = {j : m.addVar(lb = 0, ub = 1000, name = f"r{j}", vtype=GRB.INTEGER)   for j in range(k)}
    ratio = {j : m.addVar(lb = 0, ub = 1, name = f"ratio{j}", vtype=GRB.CONTINUOUS)   for j in range(k)}
    m.params.NonConvex = 2
    m.setAttr("BranchPriority", r, 1)
    m.addConstrs( r[j]/1000*vap[j] <= bvap[j]  for j in range(k))
    m.addConstrs( ratio[j] == r[j]/1000  for j in range(k))
    for j in range(k):
        m.addGenConstrPWL( ratio[j], cdf[j], RATIO_BREAKPOINTS, CDF_VALUES)
    
    #m.addConstrs( ratio[j] <= ratio[j+1]  for j in range(k-1) )
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)

def add_PWL_approx_objective(m, G, k, R, U):
    BVAP = {i : int(G.nodes[i]["BVAP"]) for i in G.nodes}
    VAP = {i : int(G.nodes[i]["VAP"]) for i in G.nodes}
    approx_VAP = sum(VAP.values())/k
    
    node_ratios = [BVAP[i]/VAP[i]  for i in G.nodes]
    inf = min(node_ratios)
    sup = max(node_ratios)
    
    vstep = (cdf_fun(sup)-cdf_fun(inf))/R
    CDF_VALUES = [cdf_fun(inf)+vstep*i  for i in range(R+1)]
    
    RATIO_BREAKPOINTS = []
    for v in CDF_VALUES:
        def func(x):
            return cdf_fun(x)-v
        RATIO_BREAKPOINTS += [spo.fsolve(func,.5)[0]]
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    
    ## population variables
    bvap = {j : m.addVar()  for j in range(k)} # bvap in district j
    
    ##  VOTING AGE POPULATION AND BVAP POPULATION
    m.addConstrs((   sum(BVAP[i]*m._X[i,j]   for i in G.nodes)             ==  bvap[j]             for j in range(k)  ), name="BVAP_Block_j")
    
    ratio = {j : m.addVar(lb = 0, ub = 1, name = f"ratio{j}", vtype=GRB.CONTINUOUS)   for j in range(k)}
    m.addConstrs( ratio[j]*approx_VAP <= bvap[j]  for j in range(k))
    for j in range(k):
        m.addGenConstrPWL( ratio[j], cdf[j], RATIO_BREAKPOINTS, CDF_VALUES)
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)

def add_conv_objective(m, G, k, U, dist_bounds=False):
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, U)
        
    #RATIO_BREAKPOINTS = [i/PWL_PARTS  for i in range(PWL_PARTS+1)]
    # the below ratio breakpoints were computed in Mathematica for some reason.
    RATIO_BREAKPOINTS = [0, 0.574291, 0.582676, 0.601897, 0.625041, 0.655121, 0.701284, 1]
    CDF_VALUES = [cdf_fun(r)  for r in RATIO_BREAKPOINTS] # function values at breakpoints
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    
    
    r = {j : m.addVar(lb = 0, ub = 1000, name = f"r{j}", vtype=GRB.INTEGER)   for j in range(k)}
    ratio = {j : m.addVar(lb = 0, ub = 1, name = f"ratio{j}", vtype=GRB.CONTINUOUS)   for j in range(k)}
    m.params.NonConvex = 2
    m.setAttr("BranchPriority", r, 1)
    m.addConstrs( r[j]/1000*vap[j] <= bvap[j]  for j in range(k))
    m.addConstrs( ratio[j] == r[j]/1000  for j in range(k))
    for j in range(k):
        m.addGenConstrPWL( ratio[j], cdf[j], RATIO_BREAKPOINTS, CDF_VALUES)
    
    if dist_bounds != False:
        m.addconstrs( ratio[j] >= dist_bounds[j][0]  for j in range(k) )
        m.addconstrs( ratio[j] <= dist_bounds[j][1]  for j in range(k) )
    m.addConstrs( ratio[j] <= ratio[j+1]  for j in range(k-1) )
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)

def add_ratio_objective(m, G, k, n, sense, U, dist_bounds=False):
    # Add bvap and vap
    BVAP, VAP, VAP_TOTAL, BVAP_TOTAL, vap, bvap = add_bvap_vap(m, G, k, R,U, bvap_ordering = True)
    
    #r = {j : m.addVar(lb = 0, ub = 1000, name = f"r{j}", vtype=GRB.INTEGER)   for j in range(k)}
    #ratio = {j : m.addVar(lb = 0, ub = 1, name = f"ratio{j}", vtype=GRB.CONTINUOUS)   for j in range(k)}
    ub = 10000
    r =  m.addVar(lb = 0, ub = ub, name = "r", vtype=GRB.INTEGER)
    ratio = m.addVar(lb = 0, ub = 1, name = "ratio", vtype=GRB.CONTINUOUS)
    m.addConstr( ratio == r/ub)

    m.params.NonConvex = 2
    m.setAttr("BranchPriority", r, 10)

    if sense == 'maximize':
        m.addConstrs( r/ub*vap[j] <= bvap[j]  for j in range(n))
        m.setObjective(ratio, GRB.MAXIMIZE)
        
    else:
        m.addConstrs( r/ub*vap[j] >= bvap[j]  for j in range(n))
        m.setObjective(ratio, GRB.MINIMIZE)
    
    
def add_conv_approx_objective(m, G, k, U):
    
    
    BVAP = {i : G.nodes[i]["BVAP"] for i in G.nodes}
    VAP = {i : G.nodes[i]["VAP"] for i in G.nodes}
    approx_VAP = sum(VAP.values())/k
    
    #RATIO_BREAKPOINTS = [i/PWL_PARTS  for i in range(PWL_PARTS+1)]
    # the below ratio breakpoints were computed in Mathematica for some reason.
    RATIO_BREAKPOINTS = [0, 0.574291, 0.582676, 0.601897, 0.625041, 0.655121, 0.701284, 1]
    CDF_VALUES = [cdf_fun(r)  for r in RATIO_BREAKPOINTS] # function values at breakpoints
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} # r = cdf(8.26 q - 3.271)
    
    ## population variables
    bvap = {j : m.addVar(name = f"bvap{j}")  for j in range(k)} # bvap in district j
    
    ##  VOTING AGE POPULATION AND BVAP POPULATION
    m.addConstrs((   sum(BVAP[i]*m._X[i,j]   for i in G.nodes)             ==  bvap[j]             for j in range(k)  ), name="BVAP_Block_j")
    
    ratio = {j : m.addVar(lb = 0, ub = 1, name = f"ratio{j}", vtype=GRB.CONTINUOUS)   for j in range(k)}
    m.addConstrs( ratio[j]*approx_VAP <= bvap[j]  for j in range(k))
    for j in range(k):
        m.addGenConstrPWL( ratio[j], cdf[j], RATIO_BREAKPOINTS, CDF_VALUES)
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)
    