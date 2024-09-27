import gurobipy as gp
from gurobipy import GRB 
import scipy.stats as stats

def add_base_constraints(m, population, L, U, k):
    DG = m._DG
    # Each vertex i assigned to one district
    m.addConstrs(gp.quicksum(m._X[i,j] for j in DG.nodes) == 1 for i in DG.nodes)
     
    # Pick k centers
    m.addConstr(gp.quicksum(m._X[j,j] for j in DG.nodes) == k)
    
    # Population balance: population assigned to vertex j should be in [L,U], if j is a center
    m.addConstrs(gp.quicksum(population[i] * m._X[i,j] for i in DG.nodes) <= U * m._X[j,j] for j in DG.nodes)
    m.addConstrs(gp.quicksum(population[i] * m._X[i,j] for i in DG.nodes) >= L * m._X[j,j] for j in DG.nodes)
    
    # Add coupling inequalities for added model strength
    couplingConstrs = m.addConstrs(m._X[i,j] <= m._X[j,j] for i in DG.nodes for j in DG.nodes)
    
    # Make them user cuts
    for i in DG.nodes:
        for j in DG.nodes:
            couplingConstrs[i,j].Lazy = -1
    
    # Set branch priority on center vars
    for j in DG.nodes:
        m._X[j,j].BranchPriority=1         

        
#def add_objective(m, G):
    # Y[i,j] = 1 if edge {i,j} is cut
#    m._Y = m.addVars(G.edges, vtype=GRB.BINARY)
#    m.addConstrs( m._X[i,v]-m._X[j,v] <= m._Y[i,j] for i,j in G.edges for v in G.nodes)
#    m.setObjective( gp.quicksum(m._Y), GRB.MINIMIZE )

#Cdf function
def cdf_fun(x):
    if x < 0.07:
        return 0
    return stats.norm.cdf(8.26*x-3.271)

def add_objective(m, G):
    BVAP = {i : G.nodes[i]["BVAP"] for i in G.nodes}
    VAP = {i : G.nodes[i]["VAP"] for i in G.nodes}
    VAP_TOTAL = sum(VAP[i] for i in G.nodes) # the total voting age population, serves as an upper bound
    BVAP_TOTAL = sum(BVAP[i] for i in G.nodes) # the total voting age population, serves as an upper bound
    
    #RATIO_BREAKPOINTS = [i/PWL_PARTS  for i in range(PWL_PARTS+1)]
    # the below ratio breakpoints were computed in Mathematica for some reason.
    RATIO_BREAKPOINTS = [0.158721, 0.19687, 0.221727, 0.240853, 0.256737, 0.270529, 0.282858, 0.294114, 0.30455, 0.314347, 0.323637, 0.332518, 0.34107, 0.349356, 0.357429, 0.365333, 0.373109, 0.380792, 0.388413, 0.396005, 0.403596,  0.411218, 0.418901, 0.426676, 0.434581, 0.442654, 0.45094, 0.459492,  0.468373, 0.477662, 0.487459, 0.497896, 0.509151, 0.521481, 0.535272,  0.551156, 0.570282, 0.59514, 0.633289, 1]
    PWL_PARTS = len(RATIO_BREAKPOINTS)-1
    CDF_VALUES = [cdf_fun(r)  for r in RATIO_BREAKPOINTS] # function values at breakpoints
    
    ## variables for modeling the objective function
    cdf = {j : m.addVar(lb = 0, ub = 1, name = f"cdf{j}")   for j in G.nodes} # r = cdf(8.26 q - 3.271)
    delta = {j : {l : m.addVar(vtype = GRB.BINARY)  for l in range(PWL_PARTS+1)}  for j in G.nodes}
    
    ## population variables
    vap = {j : m.addVar(name = f"vap{j}")   for j in G.nodes} # voting age population in block k
    bvap = {j : m.addVar()  for j in G.nodes} # bvap in block k
    
    # MODELING THE OBJECTIVE
    ##  if j is not a center, then cdf[j] = 0.
    ##  Note that 1 is a big M here.
    m.addConstrs((   m._X[j,j]      >=  cdf[j]                for j in G.nodes    ), name="cdf_force_zero")
    ### if j not a center, then delta variables are 0 (and hence inactive)
    ### this constraint is not really necessary, but should be good valid inequalties
    m.addConstrs(( m._X[j,j]  >= delta[j][l]   for j in G.nodes  for l in range(PWL_PARTS+1)  ), name="delta_force_zero")
    
    ##  VOTING AGE POPULATION AND BVAP POPULATION
    m.addConstrs((   sum(VAP[i]*m._X[i,j]    for i in G.nodes)             ==  vap[j]              for j in G.nodes  ), name="VAP_Block_j")
    m.addConstrs((   sum(BVAP[i]*m._X[i,j]   for i in G.nodes)             ==  bvap[j]             for j in G.nodes  ), name="BVAP_Block_j")

    ## Decide value of cdf[j]
    ## We focus on the fact that the cdf is an increasing function
    ## We model  "if ratio <= RATIO_BREAKPOINT, then cdf <= CDF_VALUE@RATIO_BREAKPOINT"
    ## For this, we use an indicator variable \delta, where \delta = 1 means ratio <= RATIO_BREAKPOINT
    
    ### if bvap/vap <= ratio, then delta = 1
    ### --->  if bvap - ratio*vap <= 0, then delta = 1
    ### --->  if ratio*vap - bvap > 0, then delta = 1
    ### --->  ratio*vap - bvap <= M*delta
    m.addConstrs(( RATIO_BREAKPOINTS[l]*vap[j] - bvap[j] <=  VAP_TOTAL*delta[j][l] for j in G.nodes  for l in range(PWL_PARTS+1)  ), name="breakpoint_bound")
    
    ### if delta = 1, then cdf_j <= CDF_VALUE: (note: 1 acts as the big M here)
    m.addConstrs(( cdf[j] <= CDF_VALUES[l] + (1-delta[j][l])   for j in G.nodes  for l in range(PWL_PARTS+1)  ), name="cdf_bound")
    
    ### delta variables are ordered (i.e., if ratio[j] <= RATIO_BREAKPOINT[l], then ratio[j] <= RATIO_BREAKPOINT[l+1])
    ### thus, if \delta[l] = 1, then \delta[l+1] = 1
    m.addConstrs(( delta[j][l] <= delta[j][l+1]  for j in G.nodes  for l in range(PWL_PARTS)  ), name="cdf_bound")
    
    #m.addConstr( sum(cdf[j] for j in G.nodes) <= 2.13 )
    
    m.setObjective( sum(cdf[j] for j in G.nodes), GRB.MAXIMIZE)




def add_extended_objective(m, G):
    # Z[i,j,v] = 1 if edge (i,j) is cut because i->v but j!->v
    m._Z = m.addVars(G.edges, G.nodes, vtype=GRB.BINARY) 
    m.addConstrs( m._X[i,v]-m._X[j,v] <= m._Z[i,j,v] for i,j in G.edges for v in G.nodes)
    m.setObjective( gp.quicksum(m._Z), GRB.MINIMIZE )
    
    
def most_possible_nodes_in_one_district(population, U):
    cumulative_population = 0
    num_nodes = 0
    for ipopulation in sorted(population):
        cumulative_population += ipopulation
        num_nodes += 1
        if cumulative_population > U:
            return num_nodes - 1
   
    
def add_shir_constraints(m):
    DG = m._DG
    
    # F[j,u,v] tells how much flow (from source j) is sent across arc (u,v)
    F = m.addVars( DG.nodes, DG.edges, vtype=GRB.CONTINUOUS)
    
    # compute big-M    
    M = most_possible_nodes_in_one_district(m._population, m._U) - 1
    
    m.addConstrs( gp.quicksum( F[j,u,j] for u in DG.neighbors(j)) == 0 for j in DG.nodes)
    m.addConstrs( gp.quicksum( F[j,u,i]-F[j,i,u] for u in DG.neighbors(i) ) == m._X[i,j] for i in DG.nodes for j in DG.nodes if i!=j)
    m.addConstrs( gp.quicksum( F[j,u,i] for u in DG.neighbors(i) ) <= M * m._X[i,j] for i in DG.nodes for j in DG.nodes if i!=j)
    m.update()
      
        
def add_scf_constraints(m, G, extended):
    DG = m._DG

    m._Y = m.addVars(G.edges, vtype=GRB.BINARY)
    
    # F[u,v] tells how much flow is sent across arc (u,v)
    F = m.addVars( DG.edges, vtype=GRB.CONTINUOUS )
    
    # compute big-M
    M = most_possible_nodes_in_one_district(m._population, m._U) - 1
    
    m.addConstrs( gp.quicksum(m._X[i,j] for i in DG.nodes) == gp.quicksum(F[j,u]-F[u,j] for u in DG.neighbors(j)) + 1 for j in DG.nodes)
    m.addConstrs( gp.quicksum(F[u,j] for u in DG.neighbors(j)) <= M * (1-m._X[j,j]) for j in DG.nodes)
        
    if extended:
        m.addConstrs( F[i,j] + F[j,i] <= M * (1 - gp.quicksum(m._Z[i,j,v] for v in G.nodes)) for i,j in G.edges)
    else:
        m.addConstrs( F[i,j] + F[j,i] <= M * (1 - m._Y[i,j]) for i,j in G.edges)
        
