import gurobipy as gp
from gurobipy import GRB
import scipy.stats as stats
import scipy.optimize as spo
import scipy.integrate as integrate
import math
import numpy as np

import helper_methods as hm


####################################   
# Partisan Data and Variables
#################################### 
def add_partisan(m, G, k, U, bounds = None, ordering = False, simple_bounds = False):
    
    Dem16 = {i : G.nodes[i]["D16"] for i in G.nodes}
    Rep16 = {i : G.nodes[i]["R16"] for i in G.nodes}
    Dem20 = {i : G.nodes[i]["D20"] for i in G.nodes}
    Rep20 = {i : G.nodes[i]["R20"] for i in G.nodes}
    TotalVotes = sum(Dem16[i]+Rep16[i]+Dem20[i]+Rep20[i]  for i in G.nodes)
    
    # Add variables:
    if bounds == None:
        m._D16 = {j : m.addVar(name = f"D16{j}", ub = U)   for j in range(k)}
        m._R16 = {j : m.addVar(name = f"R16{j}", ub = U)   for j in range(k)}
        m._T16 = {j : m.addVar(name = f"T16{j}", ub = U)   for j in range(k)}
        m._D20 = {j : m.addVar(name = f"D20{j}", ub = U)   for j in range(k)}
        m._R20 = {j : m.addVar(name = f"R20{j}", ub = U)   for j in range(k)}
        m._T20 = {j : m.addVar(name = f"T20{j}", ub = U)   for j in range(k)}
        m._diff20 = {j: m.addVar(lb=-gp.GRB.INFINITY, name = f"diff_D_T20{j}") for j in range(k)}
        m._diff16 = {j: m.addVar(lb=-gp.GRB.INFINITY, name = f"diff_D_T16{j}") for j in range(k)}
                
        

        
    else:
        print("Loading bounds for continuous model.")
        print(bounds)
        m._D16 = {j : m.addVar(name = f"D16{j}", lb = bounds[f'D16']['lb'][j], ub = bounds[f'D16']['ub'][j])   for j in range(k)}
        m._R16 = {j : m.addVar(name = f"R16{j}", lb = bounds[f'R16']['lb'][j], ub = bounds[f'R16']['ub'][j])   for j in range(k)}
        m._T16 = {j : m.addVar(name = f"T16{j}", lb = bounds[f'T16']['lb'][j], ub = bounds[f'T16']['ub'][j])   for j in range(k)}
        m._D20 = {j : m.addVar(name = f"D20{j}", lb = bounds[f'D20']['lb'][j], ub = bounds[f'D20']['ub'][j])   for j in range(k)}
        m._R20 = {j : m.addVar(name = f"R20{j}", lb = bounds[f'R20']['lb'][j], ub = bounds[f'R20']['ub'][j])   for j in range(k)}
        m._T20 = {j : m.addVar(name = f"T20{j}", lb = bounds[f'T20']['lb'][j], ub = bounds[f'T20']['ub'][j])   for j in range(k)}
        
        m._diff20 = {j: m.addVar( lb = bounds[f'diff_D_T20']['lb'][j], ub = bounds[f'diff_D_T20']['ub'][j], name = f"diff_D_T20{j}") for j in range(k)}
        m._diff16 = {j: m.addVar( lb = bounds[f'diff_D_T16']['lb'][j], ub = bounds[f'diff_D_T16']['ub'][j], name = f"diff_D_T16{j}") for j in range(k)}
                
        
        
    m.addConstrs(m._diff20[j] == m._D20[j] - m._T20[j] for j in range(k))
    m.addConstrs(m._diff16[j] == m._D16[j] - m._T16[j] for j in range(k))
        
    m.addConstrs((  m._T20[j] == m._D20[j]+ m._R20[j]  for j in range(k)  ), name="total_votes_2020_Dist"  )
    m.addConstrs((  m._T16[j] == m._D16[j]+ m._R16[j]  for j in range(k)  ), name="total_votes_2016_Dist"  )
        
    m.addConstrs((  m._D16[j] == sum(Dem16[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="Dem_Votes_2016_Dist"  )
    m.addConstrs((  m._R16[j] == sum(Rep16[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="Rep_Votes_2016_Dist"  )

    
    m.addConstrs((  m._D20[j] == sum(Dem20[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="Dem_Votes_2020_Dist"  )
    m.addConstrs((  m._R20[j] == sum(Rep20[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="Rep_Votes_2020_Dist"  )

    
    # adding simple bounds
    if simple_bounds:
        m.addConstrs((  m._D16[j] <= m._T16[j] for j in range(k)), name="total_bound_dem16_j"  )
        m.addConstrs((  m._D20[j] <= m._T20[j] for j in range(k)), name="total_bound_dem20_j"  )
    
    # adding symmetry breaking (ordering) on D20
    if ordering:
        m.addConstrs((  m._D20[j] <= m._D20[j+1] for j in range(k-1)), name="symmetry_breaking_dem20_j"  )
    
    return Dem16, Rep16, Dem20, Rep20, TotalVotes, m._D16, m._R16, m._D20, m._R20



def add_partisan_continuous(m, G, k, U, bounds = None, ordering = True, simple_bounds = False):
    
    Dem16 = {i : G.nodes[i]["D16"] for i in G.nodes}
    Rep16 = {i : G.nodes[i]["R16"] for i in G.nodes}
    Dem20 = {i : G.nodes[i]["D20"] for i in G.nodes}
    Rep20 = {i : G.nodes[i]["R20"] for i in G.nodes}
    TotalVotes = sum(Dem16[i]+Rep16[i]+Dem20[i]+Rep20[i]  for i in G.nodes)
    
    # Add variables:
    if bounds == None:
        m._D16 = {j : m.addVar(name = f"D16{j}", ub = U)   for j in range(k)}
        m._R16 = {j : m.addVar(name = f"R16{j}", ub = U)   for j in range(k)}
        m._T16 = {j : m.addVar(name = f"T16{j}", ub = U)   for j in range(k)}
        m._D20 = {j : m.addVar(name = f"D20{j}", ub = U)   for j in range(k)}
        m._R20 = {j : m.addVar(name = f"R20{j}", ub = U)   for j in range(k)}
        m._T20 = {j : m.addVar(name = f"T20{j}", ub = U)   for j in range(k)}
        m._diff20 = {j: m.addVar(lb=-gp.GRB.INFINITY, name = f"diff_D_T20{j}") for j in range(k)}
        m._diff16 = {j: m.addVar(lb=-gp.GRB.INFINITY, name = f"diff_D_T16{j}") for j in range(k)}

        
    else:
        print("Loading bounds for continuous model.")
        print(bounds)
        m._D16 = {j : m.addVar(name = f"D16{j}", lb = bounds[f'D16']['lb'][j], ub = bounds[f'D16']['ub'][j])   for j in range(k)}
        m._R16 = {j : m.addVar(name = f"R16{j}", lb = bounds[f'R16']['lb'][j], ub = bounds[f'R16']['ub'][j])   for j in range(k)}
        m._T16 = {j : m.addVar(name = f"T16{j}", lb = bounds[f'T16']['lb'][j], ub = bounds[f'T16']['ub'][j])   for j in range(k)}
        m._D20 = {j : m.addVar(name = f"D20{j}", lb = bounds[f'D20']['lb'][j], ub = bounds[f'D20']['ub'][j])   for j in range(k)}
        m._R20 = {j : m.addVar(name = f"R20{j}", lb = bounds[f'R20']['lb'][j], ub = bounds[f'R20']['ub'][j])   for j in range(k)}
        m._T20 = {j : m.addVar(name = f"T20{j}", lb = bounds[f'T20']['lb'][j], ub = bounds[f'T20']['ub'][j])   for j in range(k)}
        
        m._diff20 = {j: m.addVar( lb = bounds[f'diff_D_T20']['lb'][j], ub = bounds[f'diff_D_T20']['ub'][j], name = f"diff_D_T20{j}") for j in range(k)}
        m._diff16 = {j: m.addVar( lb = bounds[f'diff_D_T16']['lb'][j], ub = bounds[f'diff_D_T16']['ub'][j], name = f"diff_D_T16{j}") for j in range(k)}
                
        
        
    m.addConstrs(m._diff20[j] == m._D20[j] - m._T20[j] for j in range(k))
    m.addConstrs(m._diff16[j] == m._D16[j] - m._T16[j] for j in range(k))
        
    m.addConstrs((  m._T20[j] == m._D20[j]+ m._R20[j]  for j in range(k)  ), name="total_votes_2020_Dist_j"  )
    m.addConstrs((  m._T16[j] == m._D16[j]+ m._R16[j]  for j in range(k)  ), name="total_votes_2016_Dist_j"  )
    
    # adding simple bounds
    if simple_bounds:
        m.addConstrs((  m._D16[j] <= m._T16[j] for j in range(k)), name="total_bound_dem16_j"  )
        m.addConstrs((  m._D20[j] <= m._T20[j] for j in range(k)), name="total_bound_dem20_j"  )
    
    # adding symmetry breaking (ordering) on D20
    if ordering:
        m.addConstrs((  m._D20[j] <= m._D20[j+1] for j in range(k-1)), name="symmetry_breaking_dem20_j"  )
    
    return Dem16, Rep16, Dem20, Rep20, TotalVotes, m._D16, m._R16, m._D20, m._R20
    
####################################   
# Partisan Bounds
####################################         
        
def add_partisan_dem_objective_ordering(m, G, k, R,U,obj_order,index):
    
    # Add poitical variables with an ordering   
    Dem16, Rep16, Dem20, Rep20, TotalVotes, D16, T16, D20, T20 = add_partisan(m, G, k, U, ordering = True)
    
    objective_options = {
        'D20_max': (GRB.MAXIMIZE, 'D20'),
        'D20_min': (GRB.MINIMIZE, 'D20'),
        'R20_max': (GRB.MAXIMIZE, 'R20'),
        'R20_min': (GRB.MINIMIZE, 'R20'),
        'D16_max': (GRB.MAXIMIZE, 'D16'),
        'D16_min': (GRB.MINIMIZE, 'D16'),
        'R16_max': (GRB.MAXIMIZE, 'R16'),
        'R16_min': (GRB.MINIMIZE, 'R16'),
    }

    if obj_order in objective_options:
        obj_option, var_prefix = objective_options[obj_order]
        obj_variable = globals()[f"{var_prefix}[{index}]"]
        m.setObjective(obj_variable, obj_option)

####################################   
# Partisan Objectives
#################################### 


def add_partisan_dem_objective(m, G, k, R,U):
    # add partisan data and variables
    Dem16, Rep16, Dem20, Rep20, TotalVotes, D16, T16, D20, T20 = add_partisan(m, G, k, U)
    
    Ratios16 = [Dem16[i]/(Dem16[i]+Rep16[i])  for i in G.nodes]
    Ratios20 = [Dem20[i]/(Dem20[i]+Rep20[i])  for i in G.nodes]
    inf = min(min(Ratios16),min(Ratios20))
    sup = max(max(Ratios16),max(Ratios20))*1.01
    
    r_initial = [ inf+i*(sup-inf)/R  for i in range(R+1) ]
    
    def rho(r):
        return sum(sum(  CPVI_fun(r[i]+r[j])*(r[i]-r[i-1])*(r[j]-r[j-1])  for i in range(1,len(r))  )  for j in range(1,len(r))  )

    cons = ({ 'type':'eq', 'fun': lambda r: r[-1] - r_initial[-1] }, { 'type':'eq', 'fun': lambda r: r[0] - r_initial[0] })

    result = spo.minimize(  rho, r_initial, constraints=cons  )

    integral = integrate.dblquad(  lambda x, y: CPVI_fun(x+y), r_initial[0], r_initial[-1], lambda x: r_initial[0], lambda x: r_initial[-1]  )[0]

    if result.success:
        print( 'Breakpoints successfully generated!' )
        print( f'Breakpoints = {result.x}' )
        print( f'ExpectError = {(result.fun-integral)/(r_initial[-1]-r_initial[0])**2}' )
    else:
        print('Breakpoints failed to generate!')
    RATIO_BREAKPOINTS = result.x
    
    CDF_VALUES = [[CPVI_fun(l+r)  for l in RATIO_BREAKPOINTS]  for r in RATIO_BREAKPOINTS]  ##only need upper triangle?
    
    
    delta16 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta16{j}{l}")  for l in range(1,R+1)}  for j in range(k)}
    delta20 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta20{j}{l}")  for l in range(1,R+1)}  for j in range(k)}

    m._cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)}
    
    
    ### DPVI[j] == (D16[j]/T16[j]+D20[j]/T20[j])*50-51.69  for j in range(k)
    m.addConstrs((  RATIO_BREAKPOINTS[l]*m._T16[j]-m._D16[j] <= U*delta16[j][l]  for l in range(1,R+1)  for j in range(k)  ),  name='ratio16')
    m.addConstrs((  RATIO_BREAKPOINTS[l]*m._T20[j]-m._D20[j] <= U*delta20[j][l]  for l in range(1,R+1)  for j in range(k)  ),  name='ratio20')
    
    m.addConstrs((  m._cdf[j] <= CDF_VALUES[l][r]+(1-delta16[j][l])+(1-delta20[j][r])  for l in range(1,R+1)  for r in range(1,R+1)  for j in range(k)  ), name='cdf')
    
    m.setObjective( sum(m._cdf[j] for j in range(k)), GRB.MAXIMIZE)
    
def add_partisan_dem_max_objective(m, G, k, R,U):
    # add partisan data and variables
    Dem16, Rep16, Dem20, Rep20, TotalVotes, D16, T16, D20, T20 = add_partisan(m, G, k, U)
    
    Ratios16 = [Dem16[i]/(Dem16[i]+Rep16[i])  for i in G.nodes]
    Ratios20 = [Dem20[i]/(Dem20[i]+Rep20[i])  for i in G.nodes]
    inf = min(min(Ratios16),min(Ratios20))
    sup = 1.01*max(max(Ratios16),max(Ratios20))
    
    print(f"\n inf = {inf} \n sup = {sup} \n")
    
    vstep = (CPVI_fun(sup)-CPVI_fun(inf))/R
    CPVI_VALUES = [CPVI_fun(inf)+vstep*i  for i in range(R+1)]
    
    RATIO_BREAKPOINTS = []
    for v in CPVI_VALUES:
        def func(x):
            return CPVI_fun(x)-v
        RATIO_BREAKPOINTS += [spo.fsolve(func,.5)[0]]
    print("\n Breakpoints \n")
    print(RATIO_BREAKPOINTS)
    print("\n")
    
    integral = integrate.quad(  cdf_fun, inf, sup  )[0]
    expErr = (sum(CPVI_VALUES[i+1]*(RATIO_BREAKPOINTS[i+1]-RATIO_BREAKPOINTS[i])  for i in range(len(CPVI_VALUES)-1) )-integral)/(sup-inf)
    
    print(f"\n MaxError={vstep} \n ExpError={expErr} \n")
    
    CPVI_VALUES = [[CPVI_fun(l+r)  for l in RATIO_BREAKPOINTS]  for r in RATIO_BREAKPOINTS]  ##only need upper triangle??
    

    delta16 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta16{j}{l}")  for l in range(1,R+1)}  for j in range(k)}
    delta20 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta20{j}{l}")  for l in range(1,R+1)}  for j in range(k)}

    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)}
    
    m.addConstrs((  RATIO_BREAKPOINTS[l]*m._T16[j]-m._D16[j] <= U*delta16[j][l]  for l in range(1,R+1)  for j in range(k)  ),  name='ratio16')
    m.addConstrs((  RATIO_BREAKPOINTS[l]*m._T20[j]-m._D20[j] <= U*delta20[j][l]  for l in range(1,R+1)  for j in range(k)  ),  name='ratio20')
    
    m.addConstrs((  m._cdf[j] <= CPVI_VALUES[l][r]+(1-delta16[j][l])+(1-delta20[j][r])  for l in range(1,R+1)  for r in range(1,R+1)  for j in range(k)  ), name='cdf')
    
    m.setObjective( sum(m._cdf[j] for j in range(k)), GRB.MAXIMIZE)
    
    
def add_partisan_dem_max_objective_bounds(m, G, k, R, U,bounds):
#     # add partisan data and variables
#     Dem16, Rep16, Dem20, Rep20, TotalVotes, D16, T16, D20, T20 = add_partisan(m, G, k, U)
    
    delta16 = {}
    delta20 = {}
    Dem16 = {i : G.nodes[i]["D16"] for i in G.nodes}
    Rep16 = {i : G.nodes[i]["R16"] for i in G.nodes}
    Dem20 = {i : G.nodes[i]["D20"] for i in G.nodes}
    Rep20 = {i : G.nodes[i]["R20"] for i in G.nodes}
    TotalVotes = sum(Dem16[i]+Rep16[i]+Dem20[i]+Rep20[i]  for i in G.nodes)
    Ratios16 = [Dem16[i]/(Dem16[i]+Rep16[i])  for i in G.nodes]
    Ratios20 = [Dem20[i]/(Dem20[i]+Rep20[i])  for i in G.nodes]
    inf_min = min(min(Ratios16),min(Ratios20))
    sup_max = max(max(Ratios16),max(Ratios20))*1.01


    m._cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)}
   
    
    for j in range(k):
        # create uniform bounds on ratios for 2016 and 2020 (since they are probably simialr)
        ## The uniform bounds helps with reducing the number of breakpoints to consider by choosing a uniform spacing
        print(bounds)
        inf = min([bounds[f'D{year}']['lb'][j]/(bounds[f'D{year}']['ub'][j] + bounds[f'R{year}']['ub'][j]) for year in ['16','20']])
        sup = max([bounds[f'D{year}']['ub'][j]/(bounds[f'D{year}']['lb'][j] + bounds[f'R{year}']['lb'][j]) for year in ['16','20']])
        print(f"\n inf = {inf} \n sup = {sup} \n")
        inf = 0.07
        sup = 0.9
        inf = max(inf, inf_min)
        sup = min(sup, sup_max)
        print("Update inf sup")
        print(f"\n inf = {inf} \n sup = {sup} \n")
        
        r_initial = [ inf+i*(sup-inf)/R  for i in range(R+1) ]
        RATIO_BREAKPOINTS = r_initial
#         def rho(r):
#             return sum(sum(  hm.CPVI_fun(r[i]+r[j])*(r[i]-r[i-1])*(r[j]-r[j-1])  for i in range(1,len(r))  )  for j in range(1,len(r))  )

#         cons = ({ 'type':'eq', 'fun': lambda r: r[-1] - r_initial[-1] }, { 'type':'eq', 'fun': lambda r: r[0] - r_initial[0] })

#         result = spo.minimize(  rho, r_initial, constraints=cons  )

#         integral = integrate.dblquad(  lambda x, y: hm.CPVI_fun(x+y), r_initial[0], r_initial[-1], lambda x: r_initial[0], lambda x: r_initial[-1]  )[0]

#         if result.success:
#             print( 'Breakpoints successfully generated!' )
#             print( f'Breakpoints = {result.x}' )
#             print( f'ExpectError = {(result.fun-integral)/(r_initial[-1]-r_initial[0])**2}' )
#         else:
#             print('Breakpoints failed to generate!')
#         RATIO_BREAKPOINTS = result.x
        
        
        print(f"\n Breakpoints District{j} \n")
        print(RATIO_BREAKPOINTS)
        print("\n")

        #integral = integrate.quad(  hm.cdf_fun, inf, sup  )[0]
        #expErr = (sum(CPVI_VALUES[i+1]*(RATIO_BREAKPOINTS[i+1]-RATIO_BREAKPOINTS[i])  for i in range(len(CPVI_VALUES)-1) )-integral)/(sup-inf)

        #print(f"\n MaxError={vstep} \n ExpError={expErr} \n")

        #CPVI_VALUES = {r : {l: hm.CPVI_fun(l+r)  for l in RATIO_BREAKPOINTS}  for r in RATIO_BREAKPOINTS} #only need upper triangle??
        CPVI_VALUES = [ [hm.CPVI_fun(l+r)  for l in RATIO_BREAKPOINTS]  for r in RATIO_BREAKPOINTS] #only need upper triangle??
        #print("CPVI_VALUES")
        #print(CPVI_VALUES)
        #print(j, RATIO_BREAKPOINTS)
        ## Add delta variables
        delta16[j] =  {l : m.addVar(vtype = GRB.BINARY, name = f"delta16_{j}{l}")  for l in range(1,R+1)} 
        delta20[j] =  {l : m.addVar(vtype = GRB.BINARY, name = f"delta20_{j}{l}")  for l in range(1,R+1)}
        
        # Add relevant constraints
        m.addConstrs((  RATIO_BREAKPOINTS[l]*m._T16[j]-m._D16[j] <= U*delta16[j][l]  for l in range(1,R+1)  ),  name=f'ratio_{j}_16')
        m.addConstrs((  RATIO_BREAKPOINTS[l]*m._T20[j]-m._D20[j] <= U*delta20[j][l]  for l in range(1,R+1)  ),  name=f'ratio20_{j}_20')
    
        m.addConstrs((  m._cdf[j] <= CPVI_VALUES[l][r]+(1-delta16[j][l])+(1-delta20[j][r])  for l in range(1,R+1)  for r in range(1,R+1) ), name=f'cdf{j}_bounds')
                     
    
    m.setObjective( sum(m._cdf[j] for j in range(k)), GRB.MAXIMIZE) 
    m.write("m.lp")                 
                     
def add_partisan_dem_objective_LOGE(m, G, k, R,U):
    # add partisan data and variables
    Dem16, Rep16, Dem20, Rep20, TotalVotes, D16, T16, D20, T20 = add_partisan(m, G, k, U)
    
    Ratios16 = [Dem16[i]/(Dem16[i]+Rep16[i])  for i in G.nodes]
    Ratios20 = [Dem20[i]/(Dem20[i]+Rep20[i])  for i in G.nodes]
    inf = min(min(Ratios16),min(Ratios20))
    sup = max(max(Ratios16),max(Ratios20))
    
    r_initial = [ inf+i*(sup-inf)/R  for i in range(R+1) ]
    
    def rho(r):
        return sum(sum(  CPVI_fun(r[i]+r[j])*(r[i]-r[i-1])*(r[j]-r[j-1])  for i in range(1,len(r))  )  for j in range(1,len(r))  )

    cons = ({ 'type':'eq', 'fun': lambda r: r[-1] - r_initial[-1] }, { 'type':'eq', 'fun': lambda r: r[0] - r_initial[0] })

    result = spo.minimize(  rho, r_initial, constraints=cons  )

    integral = integrate.dblquad(  lambda x, y: CPVI_fun(x+y), r_initial[0], r_initial[-1], lambda x: r_initial[0], lambda x: r_initial[-1]  )[0]

    if result.success:
        print( 'Breakpoints successfully generated!' )
        print( f'Breakpoints = {result.x}' )
        print( f'ExpectError = {(result.fun-integral)/(r_initial[-1]-r_initial[0])**2}' )
    else:
        print('Breakpoints failed to generate!')
    RATIO_BREAKPOINTS = result.x
    
    CDF_VALUES = [[CPVI_fun(l+r)  for l in RATIO_BREAKPOINTS]  for r in RATIO_BREAKPOINTS]  ##only need upper triangle??
   
    
    
    delta16 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta16{j}{l}")  for l in range(1,R+1)}  for j in range(k)}
    delta20 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta20{j}{l}")  for l in range(1,R+1)}  for j in range(k)}

    m._cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)}
    
    # specific breakpoints for each district
    # for j in districts....
    
    ### DPVI[j] == (D16[j]/T16[j]+D20[j]/T20[j])*50-51.69  for j in range(k)
    m.addConstrs((  RATIO_BREAKPOINTS[l]*m._T16[j]-m._D16[j] <= U*delta16[j][l]  for l in range(1,R+1)  for j in range(k)  ),  name='ratio16')
    m.addConstrs((  RATIO_BREAKPOINTS[l]*m._T20[j]-m._D20[j] <= U*delta20[j][l]  for l in range(1,R+1)  for j in range(k)  ),  name='ratio20')
    
    m.addConstrs((  m._cdf[j] <= CDF_VALUES[l][r]+(1-delta16[j][l])+(1-delta20[j][r])  for l in range(1,R+1)  for r in range(1,R+1)  for j in range(k)  ), name='cdf')
    
    m.setObjective( sum(m._cdf[j] for j in range(k)), GRB.MAXIMIZE)    
    
def add_partisan_rep_objective(m, G, k, R, U):
    Dem16 = {i : G.nodes[i]["D16"] for i in G.nodes}
    Rep16 = {i : G.nodes[i]["R16"] for i in G.nodes}
    Dem20 = {i : G.nodes[i]["D20"] for i in G.nodes}
    Rep20 = {i : G.nodes[i]["R20"] for i in G.nodes}
    
    TotalVotes = sum(Dem16[i]+Rep16[i]+Dem20[i]+Rep20[i]  for i in G.nodes)
    
    Ratios16 = [Rep16[i]/(Dem16[i]+Rep16[i])  for i in G.nodes]
    Ratios20 = [Rep20[i]/(Dem20[i]+Rep20[i])  for i in G.nodes]
    inf = min(min(Ratios16),min(Ratios20))
    sup = max(max(Ratios16),max(Ratios20))
    
    r_initial = [ inf+i*(sup-inf)/R  for i in range(R+1) ]
    
    def rho(r):
        return sum(sum(  CPVI_fun(r[i]+r[j])*(r[i]-r[i-1])*(r[j]-r[j-1])  for i in range(1,len(r))  )  for j in range(1,len(r))  )

    cons = ({ 'type':'eq', 'fun': lambda r: r[-1] - r_initial[-1] }, { 'type':'eq', 'fun': lambda r: r[0] - r_initial[0] })

    result = spo.minimize(  rho, r_initial, constraints=cons  )

    integral = integrate.dblquad(  lambda x, y: CPVI_fun(x+y), r_initial[0], r_initial[-1], lambda x: r_initial[0], lambda x: r_initial[-1]  )[0]

    if result.success:
        print( 'Breakpoints successfully generated!' )
        print( f'Breakpoints = {result.x}' )
        print( f'ExpectError = {(result.fun-integral)/(r_initial[-1]-r_initial[0])**2}' )
    else:
        print('Breakpoints failed to generate!')
    RATIO_BREAKPOINTS = result.x
    
    CDF_VALUES = [[CPVI_fun(l+r)  for l in RATIO_BREAKPOINTS]  for r in RATIO_BREAKPOINTS]  ##only need upper triangle??
    
    R16 = {j : m.addVar(name = f"D16{j}")   for j in range(k)}
    T16 = {j : m.addVar(name = f"T16{j}")   for j in range(k)}
    R20 = {j : m.addVar(name = f"D20{j}")   for j in range(k)}
    T20 = {j : m.addVar(name = f"T20{j}")   for j in range(k)}
    
    delta16 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta16{j}{l}")  for l in range(1,R+1)}  for j in range(k)}
    delta20 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta20{j}{l}")  for l in range(1,R+1)}  for j in range(k)}

    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)}
    
    m.addConstrs((  R16[j] == sum(Rep16[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="Rep_Votes_2016_Dist_j"  )
    m.addConstrs((  T16[j] == R16[j]+sum(Dem16[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="total_votes_2016_Dist_j"  )
    m.addConstrs((  R20[j] == sum(Rep20[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="Dem_Votes_2020_Dist_j"  )
    m.addConstrs((  T20[j] == R20[j]+sum(Dem20[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="total_votes_2020_Dist_j"  )
    
    ### DPVI[j] == (D16[j]/T16[j]+D20[j]/T20[j])*50-51.69  for j in range(k)
    m.addConstrs((  RATIO_BREAKPOINTS[l]*T16[j]-R16[j] <= TotalVotes*delta16[j][l]  for l in range(1,R+1)  for j in range(k)  ),  name='ratio16')
    m.addConstrs((  RATIO_BREAKPOINTS[l]*T20[j]-R20[j] <= TotalVotes*delta20[j][l]  for l in range(1,R+1)  for j in range(k)  ),  name='ratio20')
    m.addConstrs((  cdf[j] <= CDF_VALUES[l][r]+(1-delta16[j][l])+(1-delta20[j][r])  for l in range(1,R+1)  for r in range(1,R+1)  for j in range(k)  ), name='cdf')
    
    m.setObjective( sum(cdf[j] for j in range(k)), GRB.MAXIMIZE)



def add_comp_objective(m, G, k, U):
    Dem16 = {i : G.nodes[i]["D16"] for i in G.nodes}
    Rep16 = {i : G.nodes[i]["R16"] for i in G.nodes}
    Dem20 = {i : G.nodes[i]["D20"] for i in G.nodes}
    Rep20 = {i : G.nodes[i]["R20"] for i in G.nodes}
    
    TotalVotes = sum(Dem16[i]+Rep16[i]+Dem20[i]+Rep20[i]  for i in G.nodes)
    
    RATIO_BREAKPOINTS = [0.149275,0.163929,0.178582,0.193236,0.20789,0.222544,0.237198,0.251852,0.266505,0.281159,0.295813,0.310467,0.325121,0.339775,0.354428,0.369082,0.383736,0.39839,0.413044,0.427698,0.442351,0.457005,0.471659,0.486313,0.500967,0.515621,0.530274,0.544928,0.559582,0.574236,0.58889,0.603544,0.618197,0.632851,0.647505,0.662159,0.676813,0.691467,0.70612,0.720774]
    R = range(len(RATIO_BREAKPOINTS))
    ETA_VALUES = [[Comp_fun( min(l+r, 2*1.0338-l-r) )  for l in RATIO_BREAKPOINTS]  for r in RATIO_BREAKPOINTS]  ##only need upper triangle??
    
    # Somethign strange here...
    R16 = {j : m.addVar(name = f"D16{j}")   for j in range(k)}
    T16 = {j : m.addVar(name = f"T16{j}")   for j in range(k)}
    R20 = {j : m.addVar(name = f"D20{j}")   for j in range(k)}
    T20 = {j : m.addVar(name = f"T20{j}")   for j in range(k)}
    
    delta16 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta16{j}{l}")  for l in R}  for j in range(k)}
    delta20 = {j : {l : m.addVar(vtype = GRB.BINARY, name = f"delta20{j}{l}")  for l in R}  for j in range(k)}

    h = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in range(k)}
    
    m.addConstrs((  R16[j] == sum(Rep16[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="Dem_Votes_2016_Dist_j"  )
    m.addConstrs((  T16[j] == R16[j]+sum(Dem16[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="total_votes_2016_Dist_j"  )
    m.addConstrs((  R20[j] == sum(Rep20[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="Dem_Votes_2020_Dist_j"  )
    m.addConstrs((  T20[j] == R20[j]+sum(Dem20[i]*m._X[i,j]  for i in G.nodes)  for j in range(k)  ), name="total_votes_2020_Dist_j"  )
    
    m.addConstrs((  RATIO_BREAKPOINTS[l]*T16[j]-R16[j] <= TotalVotes*delta16[j][l]  for l in R  for j in range(k)  ),  name='ratio16')
    m.addConstrs((  RATIO_BREAKPOINTS[l]*T20[j]-R20[j] <= TotalVotes*delta20[j][l]  for l in R  for j in range(k)  ),  name='ratio20')
    m.addConstrs((  h[j] <= ETA_VALUES[l][r]+(1-delta16[j][l])+(1-delta20[j][r])  for l in R  for r in R  for j in range(k)  ), name='cdf')
    
    m.setObjective( sum(h[j] for j in range(k)), GRB.MAXIMIZE)

                                                                              
