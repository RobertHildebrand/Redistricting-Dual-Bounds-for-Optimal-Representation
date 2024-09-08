import gurobipy as gp
from gurobipy import GRB
import scipy.stats as stats
import scipy.optimize as spo
import scipy.integrate as integrate
import math
import numpy as np

import helper_methods as hm

from partisan import * # has all the partisan objectives
from bvap import *
from bvap_models import *

####################################   
# Base constraints
#################################### 

def add_base_constraints(m, population, L, U, k):
    DG = m._DG # bidirected version of G
    m._population = population
    m._U = U
    m._L = L
    m._k = k
    # Each vertex i assigned to one district
    m.addConstrs((gp.quicksum(m._X[i,j] for j in range(k)) == 1 for i in DG.nodes), name = 'vertex to district')
     
    # Population balance: population assigned to district j should be in [L,U]
    m.addConstrs((gp.quicksum(population[i] * m._X[i,j] for i in DG.nodes) <= U for j in range(k)), name = 'pop UB')
    m.addConstrs((gp.quicksum(population[i] * m._X[i,j] for i in DG.nodes) >= L for j in range(k)), name = 'pop LB') 
    
    
####################################   
# BVAP + HVAP Objectives
#################################### 

def add_step_hvap_objective(m, G, k, R, state, U):
    BVAP = {i : int(G.nodes[i]["BVAP"]) for i in G.nodes}
    VAP = {i : int(G.nodes[i]["VAP"]) for i in G.nodes}
    HVAP = {i : int(G.nodes[i]["HVAP"]) for i in G.nodes}
    VAP_TOTAL = sum(VAP[i] for i in G.nodes) # the total voting age population, serves as an upper bound
    
    
    if state in ["AL","GA","LA","MS","SC"]: # Deep South
        a = 10.44
        b = 3
        c = -4.729
        
    if state in ["MD","VA","NC","TN"]: # Rim South
        a = 9.75
        b = 3
        c = -4.194
    
    def cdf_fun(r):
        return stats.norm.cdf(r+c)
    
    inf = 0
    sup = max(a,b)
    
    r_initial = [ inf+i*(sup-inf)/R  for i in range(R+1) ]
    
    cons = ({ 'type':'eq', 'fun': lambda r: r[-1] - r_initial[-1] }, { 'type':'eq', 'fun': lambda r: r[0] - r_initial[0] })
     
    brkpts = spo.minimize(  rho, r_initial, constraints=cons, jac=grad, tol=.00001  )
    
    integral = integrate.quad(  stats.norm.cdf, r_initial[0], r_initial[-1]  )[0]
    
    if brkpts.success:
        print( 'Breakpoints successfully generated!' )
        print( f'Breakpoints = {brkpts.x}' )
        expErr = (brkpts.fun-integral)/(r_initial[-1]-r_initial[0])
        print( f'ExpectError = {expErr}' )
    else:
        print('Breakpoints failed to generate!')
    RATIO_BREAKPOINTS = brkpts.x
    
    PWL_PARTS = len(RATIO_BREAKPOINTS)-1
    CDF_VALUES = [cdf_fun(r)  for r in RATIO_BREAKPOINTS] # function values at breakpoints
    CDFMAX = CDF_VALUES[-1]
    CDF_DELTAS = [CDF_VALUES[i+1]-CDF_VALUES[i]  for i in range(len(CDF_VALUES)-1) ]
    maxErr = max( [CDF_VALUES[i+1]-CDF_VALUES[i]  for i in range(len(CDF_VALUES)-1)] )
    
    ## population variables
    vap = {j : m.addVar(name = f"vap{j}")   for j in range(k)} # voting age population in district j
    bvap = {j : m.addVar(name = f"bvap{j}")  for j in range(k)} # bvap in district j
    hvap = {j : m.addVar(name = f"hvap{j}")  for j in range(k)} # bvap in district j
    
    w = m.addVars(range(k), vtype=GRB.CONTINUOUS, lb=0, name="w")
    
    ##  VOTING AGE POPULATION AND BVAP POPULATION
    m.addConstrs((   sum(VAP[i]*m._X[i,j]   for i in G.nodes)  ==  vap[j]   for j in range(k)  ), name="VAP_Block_j"  )
    m.addConstrs((   sum(BVAP[i]*m._X[i,j]  for i in G.nodes)  ==  bvap[j]  for j in range(k)  ), name="BVAP_Block_j" )
    m.addConstrs((   sum(HVAP[i]*m._X[i,j]  for i in G.nodes)  ==  hvap[j]  for j in range(k)  ), name="HVAP_Block_j" )
    m.addConstrs((           w[j] <= a*bvap[j]+b*hvap[j]                    for j in range(k)  ), name="w_Block_j"    )
    
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf[{j}]")   for j in range(k)} 
    delta = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta[{j},{l}]")  for l in range(PWL_PARTS+1)}  for j in range(k)}
    
    m.addConstrs(( RATIO_BREAKPOINTS[l]*vap[j] - w[j] <=  10*VAP_TOTAL*delta[j][l] for j in range(k)  for l in range(PWL_PARTS+1)  ), name="breakpoint_bound")
    
    ### if delta = 1, then cdf_j <= CDF_VALUE: (note: 1 acts as the big M here)
    m.addConstrs(( cdf[j] <= CDFMAX-sum(CDF_DELTAS[l]*delta[j][l]  for l in range(len(CDF_VALUES)-1)) for j in range(k) ), name="cdf_bound")
    
    ### delta variables are ordered (i.e., if ratio[j] <= RATIO_BREAKPOINT[l], then ratio[j] <= RATIO_BREAKPOINT[l+1])
    m.addConstrs(( delta[j][l] <= delta[j][l+1]  for j in range(k)  for l in range(PWL_PARTS)  ), name="delta-ordering")
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)

    



####################################   
# Compactness Objectives
#################################### 

def add_perimax_objective(m, G, K):
    m._Y = {i : {j : {k : m.addVar(vtype = GRB.BINARY, name = f'Y[{i},{j},{k}]')  for k in range(K)}  for j in G.neighbors(i)}  for i in G.nodes}
    perim = m.addVar(vtype = GRB.CONTINUOUS, name = "perim")
    
    BOUND_PERIM = {i : 0  for i in G.nodes}
    for i in G.nodes:
        if G.nodes[i]['boundary_node']:
            BOUND_PERIM[i] = G.nodes[i]['boundary_perim']
    SHARED_PERIM = {i : {j : G.edges[i,j]['shared_perim']  for j in G.neighbors(i)}  for i in G.nodes}
    
    #Edge Constraints (y[i,j,k] = x[i,k](1-x[j,k]) via McCormick inequalities)
    m.addConstrs( m._Y[i][j][k] <= m._X[i,k]            for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    m.addConstrs( m._Y[i][j][k] <= 1-m._X[j,k]          for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    m.addConstrs( m._Y[i][j][k] >= m._X[i,k]-m._X[j,k]  for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    
    #Perimiter Constraint
    m.addConstr( perim == sum(BOUND_PERIM[i]*m._X[i,0]  for i in G.nodes)+sum(SHARED_PERIM[i][j]*m._Y[i][j][0]  for i in G.nodes  for j in G.neighbors(i)) )
    
    m.setObjective( perim, GRB.MAXIMIZE)
    
    
def add_perimin_objective(m, G, K):
    m._Y = {i : {j : {k : m.addVar(vtype = GRB.BINARY, name = f'Y[{i},{j},{k}]')  for k in range(K)}  for j in G.neighbors(i)}  for i in G.nodes}
    perim = m.addVar(vtype = GRB.CONTINUOUS, name = "perim")
    
    BOUND_PERIM = {i : 0  for i in G.nodes}
    for i in G.nodes:
        if G.nodes[i]['boundary_node']:
            BOUND_PERIM[i] = G.nodes[i]['boundary_perim']
    SHARED_PERIM = {i : {j : G.edges[i,j]['shared_perim']  for j in G.neighbors(i)}  for i in G.nodes}
    
    #Edge Constraints (y[i,j,k] = x[i,k](1-x[j,k]) via McCormick inequalities)
    m.addConstrs( m._Y[i][j][k] <= m._X[i,k]            for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    m.addConstrs( m._Y[i][j][k] <= 1-m._X[j,k]          for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    m.addConstrs( m._Y[i][j][k] >= m._X[i,k]-m._X[j,k]  for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    
    #Perimiter Constraint
    m.addConstr( perim == sum(BOUND_PERIM[i]*m._X[i,0]  for i in G.nodes)+sum(SHARED_PERIM[i][j]*m._Y[i][j][0]  for i in G.nodes  for j in G.neighbors(i)) )
    
    m.setObjective( perim, GRB.MINIMIZE)


def add_compact_objective(m, G, K, R, state):
    m._Y = {i : {j : {k : m.addVar(vtype = GRB.BINARY, name = f'Y[{i},{j},{k}]')  for k in range(K)}  for j in G.neighbors(i)}  for i in G.nodes}
    area = {k : m.addVar(vtype = GRB.CONTINUOUS, name = f"area[{k}]")  for k in range(K)}
    perim = {k : m.addVar(vtype = GRB.CONTINUOUS, name = f"perim[{k}]")  for k in range(K)}
    delta = {k : {l : m.addVar(vtype = GRB.BINARY, name = f"delta[{k},{l}]")  for l in range(R+1)}  for k in range(K)}
    ratio = {k : m.addVar(vtype = GRB.CONTINUOUS, name = f"ratio[{k}]")  for k in range(K)}
    
    perim_bounds = {'AL' : [0.1, 19.06987486], 
                    'MS' : [0.1, 54.05893036],
                    'LA' : [0.1, 41.06938075],
                    'SC' : [0.1, 13.27057034]}
    
    AREA = {i : G.nodes[i]['area']  for i in G.nodes}
    BOUND_PERIM = {i : 0  for i in G.nodes}
    for i in G.nodes:
        if G.nodes[i]['boundary_node']:
            BOUND_PERIM[i] = G.nodes[i]['boundary_perim']
    SHARED_PERIM = {i : {j : G.edges[i,j]['shared_perim']  for j in G.neighbors(i)}  for i in G.nodes}
    
    inf = perim_bounds[state][0]
    sup = perim_bounds[state][1]
    
    def step(i,sup,inf,R):
        return (sup-inf)*(i/R)**2
    
    brkpts = [inf+step(l,sup,inf,R)*l  for l in range(R+1)]
    print(brkpts)
    
    #Edge Constraints (y[i,j,k] = x[i,k](1-x[j,k]) via McCormick inequalities)
    m.addConstrs( m._Y[i][j][k] <= m._X[i,k]            for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    m.addConstrs( m._Y[i][j][k] <= 1-m._X[j,k]          for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    m.addConstrs( m._Y[i][j][k] >= m._X[i,k]-m._X[j,k]  for i in G.nodes  for j in G.neighbors(i)  for k in range(K))
    
    #Area Constraint
    m.addConstrs( area[k] == sum(AREA[i]*m._X[i,k]  for i in G.nodes)  for k in range(K))
    
    #Perimiter Constraint
    m.addConstrs( perim[k] == sum(BOUND_PERIM[i]*m._X[i,k]  for i in G.nodes)+sum(SHARED_PERIM[i][j]*m._Y[i][j][k]  for i in G.nodes  for j in G.neighbors(i))  for k in range(K))
    m.addConstrs( perim[k] <= brkpts[l]+sup*delta[k][l]  for k in range(K)  for l in range(R+1))
    m.addConstrs( delta[k][l] <= delta[k][l-1]  for k in range(K) for l in range(1,R+1) )
    
    #Ratio Constraint
    m.addConstrs(ratio[k] <= area[k]/brkpts[l]**2+(1-delta[k][l])  for k in range(K)  for l in range(R+1))
    
    m.setObjective(sum(ratio[k]  for k in range(K)), GRB.MAXIMIZE)


####################################   
# Extensions
#################################### 
    
def add_extended_objective(m, G, k):
    # Z[i,j,v] = 1 if edge (i,j) is cut because i->v but j!->v
    m._Z = m.addVars(G.edges, range(k), vtype=GRB.BINARY)
    m.addConstrs( m._X[i,v]-m._X[j,v] <= m._Z[i,j,v] for i,j in G.edges for v in range(k))
    m.setObjective( gp.quicksum(m._Z), GRB.MINIMIZE)


def add_orbitope_extended_formulation(m, G, k, ordering):
    s = m.addVars(G.nodes, range(k), vtype=GRB.CONTINUOUS) 
    u = m.addVars(G.nodes, range(k), vtype=GRB.CONTINUOUS) 
    w = m.addVars(G.nodes, range(k), vtype=GRB.CONTINUOUS) 
    
    m.addConstrs(m._X[i,j] == s[i,j]-s[i,j+1] for i in G.nodes for j in range(k-1))
    m.addConstrs(m._X[i,k-1] == s[i,k-1] for i in G.nodes)
    
    m.addConstrs(m._R[ordering[0],j] == w[ordering[0],j] for j in range(k))
    m.addConstrs(m._R[ordering[i],j] == w[ordering[i],j] - w[ordering[i-1],j] for i in range(1,G.number_of_nodes()) for j in range(k))
    
    m.addConstrs(m._R[i,j] <= m._X[i,j] for i in G.nodes for j in range(k))
    m.addConstrs(s[i,j] <= w[i,j] for i in G.nodes for j in range(k))
    
    m.addConstrs(u[ordering[i],j]+m._R[ordering[i],j] == u[ordering[i+1],j] + m._R[ordering[i+1],j+1] for i in range(0,G.number_of_nodes()-1) for j in range(k-1))
    m.addConstrs(u[ordering[i],k-1]+m._R[ordering[i],k-1] == u[ordering[i+1],k-1] for i in range(0,G.number_of_nodes()-1))
    m.addConstrs(u[ordering[G.number_of_nodes()-1],j]+m._R[ordering[G.number_of_nodes()-1],j] == 0 for j in range(k-1))
    
    m._R[ordering[0],0].LB=1
    m.addConstr( u[ordering[G.number_of_nodes()-1],k-1] + m._R[ordering[G.number_of_nodes()-1],k-1]==1 )  
   
            
def most_possible_nodes_in_one_district(population, U):
    cumulative_population = 0
    num_nodes = 0
    for ipopulation in sorted(population):
        cumulative_population += ipopulation
        num_nodes += 1
        if cumulative_population > U:
            return num_nodes - 1
   
    
def add_shir_constraints(m, symmetry):
    DG = m._DG
    k = m._k
        
    # g[i,j] = amount of flow generated at node i of type j
    g = m.addVars(DG.nodes, range(k), vtype=GRB.CONTINUOUS)
    
    # f[j,u,v] = amount of flow sent across arc uv of type j
    f = m.addVars(range(k), DG.edges, vtype=GRB.CONTINUOUS)

    # compute big-M    
    M = most_possible_nodes_in_one_district(m._population, m._U) - 1
    
    # the following constraints are weaker than some in the orbitope EF
    if symmetry != 'orbitope':
        m.addConstrs( gp.quicksum(m._R[i,j] for i in DG.nodes)==1 for j in range(k) )
        m.addConstrs( m._R[i,j] <= m._X[i,j] for i in DG.nodes for j in range(k) )
    
    # flow can only be generated at roots
    m.addConstrs( g[i,j] <= (M+1)*m._R[i,j] for i in DG.nodes for j in range(k) )
    
    # flow balance
    m.addConstrs( g[i,j] - m._X[i,j] == gp.quicksum(f[j,i,u]-f[j,u,i] for u in DG.neighbors(i)) for i in DG.nodes for j in range(k) )
    
    # flow type j can enter vertex i only if (i is assigned to district j) and (i is not root of j)
    m.addConstrs( gp.quicksum(f[j,u,i] for u in DG.neighbors(i)) <= M*(m._X[i,j]-m._R[i,j]) for i in DG.nodes for j in range(k) )
           

def add_scf_constraints(m, G, extended, symmetry):
    DG = m._DG
    k = m._k
    
    # f[u,v] = amount of flow sent across arc uv
    f = m.addVars(DG.edges, vtype=GRB.CONTINUOUS)
    
    # compute big-M    
    M = most_possible_nodes_in_one_district(m._population, m._U) - 1
    
    # the following constraints are weaker than some in the orbitope EF
    if symmetry != 'orbitope':
        m.addConstrs( gp.quicksum(m._R[i,j] for i in DG.nodes)==1 for j in range(k) )
        m.addConstrs( m._R[i,j] <= m._X[i,j] for i in DG.nodes for j in range(k) )  
    
    # if not a root, consume some flow.
    # if a root, only send out so much flow.
    m.addConstrs( gp.quicksum(f[u,v]-f[v,u] for u in DG.neighbors(v)) >= 1 - M * gp.quicksum(m._R[v,j] for j in range(k)) for v in G.nodes)
    
    # do not send flow across cut edges
    if extended:
        m.addConstrs( f[i,j] + f[j,i] <= M*(1 - gp.quicksum( m._Z[i,j,v] for v in range(k) )) for (i,j) in G.edges)
    else:
        m.addConstrs( f[i,j] + f[j,i] <= M*(1 - m._Y[i,j]) for (i,j) in G.edges )
            
      