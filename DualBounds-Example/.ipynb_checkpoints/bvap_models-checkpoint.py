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
    m._BVAP = {i : int(G.nodes[i]["BVAP"]) for i in G.nodes}
    m._VAP = {i : int(G.nodes[i]["VAP"]) for i in G.nodes}
    m._VAP_TOTAL = sum(m._VAP[i] for i in G.nodes) # the total voting age population, serves as an upper bound
    m._BVAP_TOTAL = sum(m._BVAP[i] for i in G.nodes) # the total black voting age population, serves as an upper bound
    
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
    m.addConstrs(( sum(m._VAP[i]*m._X[i,j]    for i in G.nodes) ==  m._vap[j]   for j in range(k)  ), name="VAP_Block_j")
    m.addConstrs(( sum(m._BVAP[i]*m._X[i,j]   for i in G.nodes) ==  m._bvap[j]  for j in range(k)  ), name="BVAP_Block_j")
    
    # Natural comparison bounds on vap vs bvap
    if comparison:
        m.addConstrs((  m._vap[j]  >=  m._bvap[j]  for j in range(k)  ), name="VAP_BVAP_compare_j")
    
    # Order bvap variables
    if bvap_ordering:
        print("Adding bvap ordering")
        m.addConstrs( (m._bvap[j] <= m._bvap[j+1]  for j in range(k-1)), name='ordering_bvap' )
    
    return m._BVAP, m._VAP, m._VAP_TOTAL, m._BVAP_TOTAL, m._vap, m._bvap    

def add_bvap_vap_continuous(m, G, k, U, bounds = None, comparison = False, bvap_ordering = False):
    '''
    Adding bvap and vap parameters and variables.
    '''
    m._BVAP = {i : int(G.nodes[i]["BVAP"]) for i in G.nodes}
    m._VAP = {i : int(G.nodes[i]["VAP"]) for i in G.nodes}
    m._VAP_TOTAL = sum(m._VAP[i] for i in G.nodes) # the total voting age population, serves as an upper bound
    m._BVAP_TOTAL = sum(m._BVAP[i] for i in G.nodes) # the total black voting age population, serves as an upper bound
    
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
    m.addConstr( m._VAP_TOTAL ==  sum(m._vap[j]   for j in range(k)) , name="VAP_Total")
    m.addConstr(  m._BVAP_TOTAL ==  sum(m._bvap[j]  for j in range(k))  , name="BVAP_Total")
    
    # Natural comparison bounds on vap vs bvap
    if comparison:
        m.addConstrs((  m._vap[j]  >=  m._bvap[j]  for j in range(k)  ), name="VAP_BVAP_compare_j")
    
    # Order bvap variables
    if bvap_ordering:
        print("Adding bvap ordering")
        m.addConstrs( (m._bvap[j] <= m._bvap[j+1]  for j in range(k-1)), name='ordering_bvap' )
    
    return m._BVAP, m._VAP, m._VAP_TOTAL, m._BVAP_TOTAL, m._vap, m._bvap  


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

    
def add_LogEPWL_objective_extra_bounds(m,G,k,L,U, bounds):
    '''
    Perhaps an improved implementation of the LogE formulation. 
    This version allows for improved bounds to be usd for each of the districts based on an ordering of the bvap variables
    '''
    # Add bvap and vap
    add_bvap_vap(m, G, k, U, bounds = bounds, bvap_ordering = True)

    # Add the objective function
    add_logE_objective(m, bounds, L, k)   


def add_bvap_bounds_objective_specific(m, G, k, R,U,bounds):
    '''
    function that is used to help generate strong bounds on the bvap and vap variables in specific directions.  
    these bounds can then be used in any of the formulations to give tighter initial formulations
    '''
    # Add bvap and vap
    add_bvap_vap(m, G, k,U, bvap_ordering = True)

    # Setting an objective, but this might be overwritten in an update outside of here.
    m.setObjective(sum(m._bvap[i]- m._vap[i] for i in range(k)), GRB.MINIMIZE)

    
def add_LogEPWL_objective_extra_bounds_continuous(m,G,k,L,U, bounds):
    '''
    Modifying this version for a continuous optimization only
    '''

    print("Running Continuous Optimization")
    ## Continuous Optimization
    add_bvap_vap_continuous(m, G, k, U, bounds = bounds, bvap_ordering = True)
    
    print("Adding diff bounds specific to SC")
    diff_lb = {6: 272073, 5: 286284.000, 0:368181, 4:306041}
    diff_ub = {6: 474464, 5: 539950, 0:541226, 4: 534351}

    for d in diff_lb.keys():
         m.addConstr(vap[d] - bvap[d] >= diff_lb[d])
    for d in diff_ub.keys():
            m.addConstr(vap[d] - bvap[d] <= diff_ub[d])
            
    # Add the objective function
    add_logE_objective(m, bounds, L, k)
    
    
def add_logE_objective(m, bounds, L, k):
    
    # write out bounds in different variables for later usage
    vap_lb, vap_ub, bvap_lb, bvap_ub = bounds['vap']['lb'], bounds['vap']['ub'],bounds['bvap']['lb'],bounds['bvap']['ub']
    
    ## variables for modeling the objective function
    m._cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)} 
    lam = {}
    delta = {}           
           
    # add breakpoints specific to each district based on the ordering of bvap in each district.
    for d in range(k): 
        Rinner = math.sqrt(vap_lb[d]**2+bvap_lb[d]**2) # inner radius
        Router = math.sqrt(vap_ub[d]**2+bvap_ub[d]**2) # outer radius

        l = bvap_lb[d]/vap_ub[d]
        u = bvap_ub[d]/vap_lb[d]

        factor = Router/Rinner
        step = (u-l)/L
        
        breakpoints  = hm.calculate_breakpoints_max_error(l, u, 0.01, hm.cdf_fun)
        myRatios, L, maxErr = breakpoints[0], breakpoints[1], breakpoints[3]

        nu = math.ceil(math.log(L,2))
        B = [[math.sqrt(Rinner**2/(1+r**2)), math.sqrt(Rinner**2 - Rinner**2/(1+r**2))] for r in myRatios] # set of inner radius vertices
        Binner, Bouter = [], []
        
        # Loop over each ratio in the list
        for i in range(len(myRatios)):
            r = myRatios[i]
            x, y = B[i]
            # Find intersection of ray with bounds box
            p1, p2 = hm.liang_barsky(x, y, factor*x, factor*y, vap_lb[d], vap_ub[d], bvap_lb[d], bvap_ub[d])
            Binner.append(p1)
            Bouter.append(p2)

        # setup gray code
        code = hm.generateGrayarr(nu)
        S = hm.summation_sets(code, nu,L-1)
        
        # create variables for the model
        for j in ['in','out']:
            for i in range(L):
                lam[(d,i,j)] = m.addVar(name = f'lam{d,i,j}', lb = 0, ub = 1)
        for q in range(nu):
            delta[(d,q)] = m.addVar( vtype = GRB.BINARY, name = f'delta{d,q}')

        # Convex combination multipliers sum to 1
        m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in range(L)) == 1)
        
        # LogE version of SOS2 constraints
        for q in range(nu):
            m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in S[q][1]) <= 1-delta[d,q])
            m.addConstr(sum(lam[d,i,j] for j in ['in','out'] for i in S[q][0]) <= delta[d,q])
        
        # uses rays intersecting the bounds
        print("d = ", d, " L=", L)
        m.addConstr(m._bvap[d] == sum(lam[d,i,'in']*Binner[i][1] for i in range(L)) + sum(lam[d,i,'out']*Bouter[i][1] for i in range(L)))
        m.addConstr(m._vap[d] == sum(lam[d,i,'in']*Binner[i][0] for i in range(L)) + sum(lam[d,i,'out']*Bouter[i][0] for i in range(L)))
        m.addConstr(m._cdf[d] == sum((lam[d,i,'in']+lam[d,i,'out'])*hm.calculate_black_rep(B[i][1]/B[i][0]) for i in range(L)))

    # set objective
    m.setObjective(sum(m._cdf[d] for d in range(k)), sense=GRB.MAXIMIZE)

