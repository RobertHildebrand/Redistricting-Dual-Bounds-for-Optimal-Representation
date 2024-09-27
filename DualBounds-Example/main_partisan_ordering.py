# Import necessary modules and packages
import argparse
import logging

# Define command line arguments using argparse
parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('config_file', type=str, nargs='?', default='default_config.json', help='Input file path')


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# /###########################
# Imports
###########################  

import gurobipy as gp
from gurobipy import GRB 

import matplotlib.pyplot as plt

from datetime import date
import math
import networkx as nx
import csv
import time
import json
import sys
import os

#from gerrychain import Graph # This functionality was not fully needed here.  It has been replaced by just using networkx.
import geopandas as gpd

# Model imports
import labeling, ordering, fixing, separation, partisan
# Plotting and export imports
from my_exports import *
import warm_start as ws


def load_config_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data



def main_partisan_ordering(state, config_path, summarize_and_export_results = True, return_bvap_vap_soln_and_bounds= False, load_config = True):
    # Parameters that we fix
    deviation = .2
    # Set desired tolerance for early termination
    tol = 0.01
    
#     {
#     "SC_vap_0_max": {
#         "state": "SC", 
#         "level": "county",
#         "base": "labeling",
#         "obj": "bvap_bounds_specific",
#         "R": 90,
#         "dist_bounds": false,
#         "tlimit": 600,
#         "fixing": false,
#         "contiguity": "lcut",
#         "symmetry": "none",
#         "extended": false,
#         "order": "none",
#         "heuristic": true,
#         "lp": false,
#         "obj_order": "bvap_max",
#         "index": 0
#     }
# }
    
    ###############################################
    # Reading config
    ############################################### 
    if load_config:
        if config_path == None:
            print("No config path!")
            return 0

        print("Reading config from", config_path)    
        config_filename_wo_extension = config_path.rsplit('.',1)[0]
        configs_file = open(config_path,'r')
        batch_configs = json.load(configs_file)
        configs_file.close()
    else:
        batch_configs = config_path # in this case, user input a dictionary that the config would have presented.
       
    ###############################################
    # Load common data
    ############################################### 
    
    config_data = load_config_data('config_data_bvap.json')
    
    ###############################################
    # Assign common data for use as it is coded
    ############################################### 
    #state_codes = config_data['state_codes']
    #number_of_congressional_districts = config_data['number_of_congressional_districts']
    default_config = config_data['default_config']
    available_config = config_data['available_config']
    available_config['state'] = set(config_data['state_codes'].keys())
    available_config['R'] = int
    available_config['fixing'] = set(available_config['fixing'])
    available_config['extended'] = set(available_config['extended'])
    available_config['heuristic'] = set(available_config['heuristic'])
    available_config['lp'] = set(available_config['lp'])

    ###############################################
    # Setup writing to files
    ###############################################

    # create directory for results
    path = os.path.join("..", "results_for_" + config_filename_wo_extension) 
    if not os.path.exists(path):
        os.mkdir(path) 

    # print results to csv file
    today = date.today()
    today_string = today.strftime("%Y_%b_%d") # Year_Month_Day, like 2019_Sept_16
    results_filename = "../results_for_" + config_filename_wo_extension + "/results_" + config_filename_wo_extension + "_" + today_string + ".csv" 

    # Check if the file already exists
    if os.path.isfile(results_filename):
        write_mode = 'a'  # Append mode
    else:
        write_mode = 'w'  # Write mode

    # prepare csv file by writing column headers
    my_fieldnames = config_data["my_fieldnames"]
    with open(results_filename, write_mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=my_fieldnames)
        if write_mode == 'w':
            writer.writeheader()  # Write headers only if the file is new


    ############################################################
    # Run experiments for each config in batch_config file
    ############################################################
    print("Experiments to run:")
    print(list(batch_configs.keys()))
    for key in batch_configs.keys(): 

        # get config and check for errors
        config = batch_configs[key]
        print("In run",key,"using config:",config,end='.')
        for ckey in available_config.keys():
            if not ckey =='R':
                if config[ckey] not in available_config[ckey]:
                    errormessage = "Error: the config option"+ckey+":"+config[ckey]+"is not known."
                    sys.exit(errormessage)
        print("")

        # fill-in unspecified configs using default values
        for ckey in available_config.keys():
            if ckey not in config.keys():
                print("Using default value",ckey,"=",default_config[ckey],"since no option was selected.")
                config[ckey] = default_config[ckey]

        # initialize dictionary to store this run's results
        result = config
        result['run'] = key            

        # read input data from config
        keys_to_extract = ['extended', 'obj', 'R', 'dist_bounds', 'state', 'level', 'base', 'heuristic', 'contiguity', 'order', 'obj_order', 'index']
        values = {k: config.get(k) for k in keys_to_extract}
        extended, obj, R, dist_bounds, state_not_used, level, base, heuristic, contiguity, order, obj_order, index = values.values()
        code = config_data['state_codes'][state]
        
        graph_json_file = f"../data/{level}/dual_graphs/{level}{code}.json"


        
        print(graph_json_file)

        def load_graph_from_json(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
            return nx.json_graph.adjacency_graph(data)

        G = load_graph_from_json(graph_json_file)
        print(G)

        DG = nx.DiGraph(G) # bidirected version of G


        # set parameters
        k = config_data['number_of_congressional_districts'][state]
        #k = number_of_congressional_districts[state]        
        population = {node : G.nodes[node]['POP100'] for node in G.nodes()}  
        
        L = math.ceil((1-deviation/2)*sum(population[node] for node in G.nodes())/k)
        U = math.floor((1+deviation/2)*sum(population[node] for node in G.nodes())/k)
        print("L =",L,", U =",U,", k =",k)
        
        result['k'] = k
        result['L'] = L
        result['U'] = U
        result['m'] = G.number_of_edges()
        result['n'] = G.number_of_nodes()

        # abort early for trivial or overtly infeasible instances
        maxp = max(population[i] for i in G.nodes)
        if k==1 or maxp>U:
            print("k=",k,", max{ p_v | v in V } =",maxp,", U =",U,end='.')
            sys.exit("Aborting early, either due to trivial instance or overtly infeasible instance.")

        # read heuristic solution from external file (?)
        result['heur_obj'] = 'n/a'
        result['heur_time'] = 'n/a'
        result['heur_iter'] = 'n/a'

        ############################################
        # Add (extended?) objective 
        ############################################
            
        obj_func_data = config_data["objective_functions"]

        # Create the objective_functions dictionary and the start_labels dictionary
        objective_functions = {}
        start_labels = {}
#         for name, data in obj_func_data.items():
#             objective_functions[name] = getattr(labeling, data['function'])
#             start_labels[name] = data['start_label']

        bounds = None
#         if order == "D20":
#             with open(f'../data/bounds/{state}_partisan_bounds.json', 'r') as f:
#                  bounds= json.load(f)

#             bounds = hm.convert_keys_to_int(bounds)

############################################################
# Load a prior solution  -  note: warm start is enacted later
############################################################
            
        result['LP_obj'] = 'n/a'
        result['LP_time'] = 'n/a'
        
                
    ############################
    # 
    ############################  

    ##  Gurobi Stuff  ####

    ############################
    # 
    ############################  
     
              

        
        # Start building your model
        m = gp.Model()
        
        m._DG = DG
        # X[i,j]=1 if vertex i is assigned to district j in {0,1,2,...,k-1}
        m._X = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY, name="X")
        m._R = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY) # for lcut separation

        # Add base constraints
        labeling.add_base_constraints(m, population, L, U, k)

        # Add partisan constraints with D20 ordering
        partisan.add_partisan(m, G, k, U, ordering=True)

        # Add threshold objective
        #partisan.add_threshold_objective(m, bounds, L, k)

        # Load warm start
        X_ws, D20_ws, D16_ws, R20_ws, R16_ws, obj_ws = ws.load_warm_start_dem(state, G)  # Assuming this function exists and returns warm start values

        # Inject warm start
        ws.inject_warm_start(m,k, X_ws)  # Assuming this function exists and sets the warm start values to the model

        # Set time limit
        m.Params.TimeLimit = 600
        
        m._numLazyCuts = 0
        m._numCallbacks = 0
        m.Params.lazyConstraints = 1
                    # Finish setting up parameters for MIP
        m.Params.Method = 3 # use concurrent method for root LP. Useful for degenerate models
        m._base = "labeling"

        # Load JSON file or create a new one if it doesn't exist
        bounds_path = "../data/bounds"
        os.makedirs(bounds_path, exist_ok=True)
        filename = f"{bounds_path}/{state}_partisan_bounds.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = {"Bounds": {}}




        # List of variables
        dem_list = [m._D20, m._D16]
        rep_list = [m._R20, m._R16]
        tot_list = [m._T20, m._T16]
        diff_list = [m._diff20, m._diff16]
        
        def max_var(m,var,i):
            print(i)
            print(var)
            m.update()
            print("Maximize ", var[i].VarName)
            # Set and run for MAXIMIZE
            m.setObjective(var[i], GRB.MAXIMIZE)
            m.update()
            m.optimize(separation.lcut_separation_generic)
 
            data["Bounds"][var[i].VarName]['ub'] = m.ObjBound
            data["Bounds"][var[i].VarName]['max_soln']=m.ObjVal
            m.addConstr(var[i] <= m.ObjBound)
            m.update()
            print(data)

        def min_var(m,var,i):
            # Set and run for MINIMIZE
            print(f"Minimize {var[i].VarName}")
            m.setObjective(var[i], GRB.MINIMIZE)
            m.update()
            m.optimize(separation.lcut_separation_generic)

            data["Bounds"][var[i].VarName]['lb'] = m.ObjBound
            data["Bounds"][var[i].VarName]['min_soln']=m.ObjVal
            m.addConstr(var[i] >= m.ObjBound)
            print(data)
        
        m.update()
        for i in range(k):
            for var in dem_list:
                data["Bounds"][var[i].VarName] = {}
            for var in rep_list:
                data["Bounds"][var[i].VarName] = {}
            for var in tot_list:
                data["Bounds"][var[i].VarName] = {}
            for var in diff_list:
                data["Bounds"][var[i].VarName] = {}
        
        for i in range(k-1, -1, -1):  # Assuming you meant to iterate in reverse order
#             for var in dem_list:
#                 max_var(m,var,i)

#             for var in rep_list: 
#                 min_var(m,var,i)
                
            for var in tot_list: 
                min_var(m,var,i)
                
            for var in diff_list: 
                min_var(m,var,i)

        for i in range(k):
#             for var in dem_list:
#                 min_var(m,var,i)

#             for var in rep_list: 
#                 max_var(m,var,i) 

            for var in tot_list: 
                max_var(m,var,i) 
            
            for var in diff_list: 
                max_var(m,var,i)


        # Save data to JSON file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

                 
        
        if summarize_and_export_results:
            print(fn)

            if obj == "bvap_bounds":
                fn = fn + "_" + obj_order + "_" + str(index)

            # Check if expErr and maxErr are defined
            print("Summarizing the results for export")
            if 'expErr' in locals() and 'maxErr' in locals():
                # Call the function with expErr and maxErr
                summarize_results(result, m, G, DG, k, fn, level, state, base, obj, start, end, expErr=expErr, maxErr=maxErr, results_filename=results_filename, my_fieldnames=my_fieldnames)
            else:
                # Call the function without expErr and maxErr
                summarize_results(result, m, G, DG, k, fn, level, state, base, obj, start, end, results_filename=results_filename, my_fieldnames=my_fieldnames)
        


if __name__ == '__main__':
    # Parse command line arguments
    
    
    args = parser.parse_args()

    ###############################################
    # Read configs/inputs and set parameters
    ############################################### 

    # read configs file and load into a Python dictionary

    if len(sys.argv)>1:
        # name your own config file in command line, like this: 
        #       python main.py usethisconfig.json
        # to keep logs of the experiments, redirect to file, like this:
        #       python main.py usethisconfig.json 1>>log_file.txt 2>>error_file.txt
        config_path = sys.argv[1] 
    else:
        config_path = 'default_config.json' # default
        

    #config_path = YOUR_OWN_CONFIG_PATH
    
    # Call main function
    main(config_path)


