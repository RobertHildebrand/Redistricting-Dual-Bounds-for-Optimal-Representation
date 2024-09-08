import json

def reorder_districts(G, district_indices, ordering):
    if ordering in ('bvap', 'D20'):
        print("Ordering warm start with respect to ", ordering)
        quantity = {j: sum(int(G.nodes[i][ordering.upper()]) for i in district_indices[j]) for j in district_indices.keys()}
        sorted_quantity = sorted(quantity.items(), key=lambda x: x[1])
        district_map = {original_label: new_label for new_label, (original_label, _) in enumerate(sorted_quantity)}
        relabeled_indices = {}
        for original_label, county_indices in district_indices.items():
            relabeled_indices[district_map[original_label]] = county_indices
        relabeled_quantity = {}
        for original_label, quantity_value in quantity.items():
            relabeled_quantity[district_map[original_label]] = quantity_value
        return relabeled_indices, relabeled_quantity
    else:
        print("Invalid ordering parameter")
        return None, None
def quantity(G,district_indices, label):
    return {j: sum(int(G.nodes[i][label.upper()]) for i in district_indices[j]) for j in district_indices.keys()}

def load_warm_start(state, obj, G, data, start_labels):
    """
    Load warm start solution from a file if available and format it for use in the current run.

    Parameters:
    -----------
    state : str
        The two-letter state abbreviation.
    obj : str
        The objective function being optimized.
    G : networkx.classes.graph.Graph
        The graph object representing the state.
    data : dict
        A dictionary of metadata about the run.

    Returns:
    --------
    heuristic_districts : dict or None
        A dictionary mapping district labels to lists of graph node indices or None if no warm start solution is found.
    """
        
    heuristic_districts = None
    
    def get_filename(state, string):
        obj_name = string.split("start_", 1)[1]
        filename = f"data/objective_sorted/{state}_{obj_name}_sorted.csv"

        with open(filename) as f:
            # skip header
            next(f)
            # return first non-empty line
            for line in f:
                if line.strip():
                    return line.split(",")[0].strip(), [elem.strip() for elem in line.split(",")[2:4]]


        # if no non-empty line was found, return None
        return None


    if obj in start_labels:
        filename_warm_start, obj_values = get_filename(state, data['start_label'])
        heuristic_districts = None
        if filename_warm_start is not None:
            print(f"Loading best known solution from {filename_warm_start}")
            with open(filename_warm_start, 'r') as f:
                data = json.load(f)

            # create a dictionary with district keys and a list of GEOID values
            districts = {node['district']: [] for node in data['nodes']}
            for node in data['nodes']:
                districts[node['district']].append(node['GEOID'])

            # create a dictionary with district keys and a list of graph indices as values
            district_indices = {district: [node for node in G.nodes if G.nodes[node]['GEOID'] in geoids] for district, geoids in districts.items()}

            if 'bvap' == 'bvap': # Fix this!!! ordering:
                relabeled_indices, relabeled_bvap = reorder_districts(G, district_indices, 'bvap')
                relabeled_vap = quantity(G,district_indices, 'VAP')
                return relabeled_indices, relabeled_bvap, relabeled_vap, obj_values[0]
                
#             if 'D20' in ordering:
#                 relabeled_indices, relabeled_D20 = reorder_districts(G, district_indices, 'BVAP')
#                 relabeled_D16 = quantity(G,district_indices, 'D16')
#                 relabeled_T20 = quantity(G,district_indices, 'T20')
#                 relabeled_T16 = quantity(G,district_indices, 'T16')
#                 return relabeled_indices, relabeled_D20, relabeled_D16, relabeled_T20, relabeled_T16,obj_value[1]
                
                
               
            
            heuristic_districts = district_indices

    return heuristic_districts


def load_warm_start_dem(state, G):
    """
    Load warm start solution from a file if available and format it for use in the current run.

    Parameters:
    -----------
    state : str
        The two-letter state abbreviation.
    G : networkx.classes.graph.Graph
        The graph object representing the state.

    Returns:
    --------
    heuristic_districts : dict or None
        A dictionary mapping district labels to lists of graph node indices or None if no warm start solution is found.
    """
    data = {'nodes':list(G.nodes)}
    heuristic_districts = None
    
    def get_filename(state):
        filename = f"data/objective_sorted/{state}_dem_sorted.csv"

        with open(filename) as f:
            # skip header
            next(f)
            # return first non-empty line
            for line in f:
                if line.strip():
                    return line.split(",")[0].strip(), [elem.strip() for elem in line.split(",")[2:4]]


        # if no non-empty line was found, return None
        return None



    filename_warm_start, obj_values = get_filename(state)
    heuristic_districts = None
    if filename_warm_start is not None:
        print(f"Loading best known solution from {filename_warm_start}")
        with open(filename_warm_start, 'r') as f:
            data = json.load(f)

        # create a dictionary with district keys and a list of GEOID values
        districts = {node['district']: [] for node in data['nodes']}
        for node in data['nodes']:
            districts[node['district']].append(node['GEOID'])

        # create a dictionary with district keys and a list of graph indices as values
        district_indices = {district: [node for node in G.nodes if G.nodes[node]['GEOID'] in geoids] for district, geoids in districts.items()}


        relabeled_indices, relabeled_D20 = reorder_districts(G, district_indices, 'D20')
        relabeled_D16 = quantity(G,district_indices, 'D16')
        relabeled_R20 = quantity(G,district_indices, 'R20')
        relabeled_R16 = quantity(G,district_indices, 'R16')
        
        print("Test warm start")
        print(quantity(G,district_indices,'D20'))
        print(relabeled_D20)
        
        return relabeled_indices, relabeled_D20, relabeled_D16, relabeled_R20, relabeled_R16,obj_values




    return heuristic_districts
def inject_warm_start(m,k,X_ws):
    for j in range(k):
        for i in X_ws[j]:
            m._X[i,j].start = 1
            
def inject_heuristic_warm_start(m, heuristic, base, order, k, position, vertex_ordering, heuristic_districts):
    """
    Inject a heuristic warm start solution into a Gurobi model object.

    Parameters:
    -----------
    m : gurobipy.Model
        The Gurobi model object to inject the warm start into.
    heuristic : bool
        Whether to use a heuristic warm start solution.
    base : str
        The algorithm used to generate the base solution.
    order : str
        The ordering used to partition the state into k districts.
    k : int
        The number of districts to partition the state into.
    position : dict
        A dictionary mapping each node in the graph to its position in the vertex ordering.
    vertex_ordering : list
        The vertex ordering used to partition the state into k districts.
    heuristic_districts : dict
        A dictionary mapping district labels to lists of graph node indices for the warm start solution.

    Returns:
    --------
    None
    """
    if heuristic and base == 'hess':
        for district in heuristic_districts:
            p = min([position[v] for v in district])
            j = vertex_ordering[p]
            for i in district:
                m._X[i,j].start = 1

    print("\n start stuff \n")
    if heuristic and base == 'labeling':
        if order == 'B_decreasing' or order == 'default':
            center_positions = [ min( position[v] for v in heuristic_districts[j] ) for j in range(k) ]
            cplabel = { center_positions[j] : j for j in range(k) }

            # what node r will root the new district j? The one with earliest position.
            for j in range(k):
                min_cp = min(center_positions)
                r = vertex_ordering[min_cp]
                old_j = cplabel[min_cp]

                for i in heuristic_districts[old_j]:
                    m._X[i,j].start = 1

                center_positions.remove(min_cp)

        # otherwise, just load the heuristic solution determined earlier
        else:
            for j in range(k):
                for i in heuristic_districts[j]:
                    m._X[i,j].start = 1
