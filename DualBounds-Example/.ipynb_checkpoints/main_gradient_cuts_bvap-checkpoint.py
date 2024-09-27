###############################################
## Command Line Setup
###############################################
import argparse
import logging

# Define command line arguments using argparse
parser = argparse.ArgumentParser(description='Description of the program')
parser.add_argument('config_file', type=str, nargs='?', default='default_config.json', help='Input file path')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###############################################
## Standard Imports
###############################################

import gurobipy as gp
from gurobipy import GRB 
import matplotlib.pyplot as plt
from datetime import date
import math
import networkx as nx
import csv
import time
import json
import os
import geopandas as gpd
import argparse
import logging
import sys
from pathlib import Path


###############################################
## Custom Imports
###############################################

# Model Imports
import labeling, ordering, fixing, separation, bvap_models, bvap
from gradient_cuts import run_gradient_cuts, apply_gradient


# Plotting and export imports
from my_exports import *
import warm_start as ws



###############################################
## Commands
###############################################

def main_bvap(config_path, summarize_and_export_results = True, return_bvap_vap_soln_and_bounds= False, load_config = True):
    # Parameters that we fix
    deviation = .2
    # Set desired tolerance for early termination
    tol = 0.01
    
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
        import json
        batch_configs = json.load(configs_file)
        configs_file.close()
    else:
        batch_configs = config_path # in this case, user input a dictionary that the config would have presented.
       
    ###############################################
    # Load common data
    ############################################### 
    
    with open('data/config_data.json', 'r') as file:
        config_data = json.load(file)
    
    
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
    import os
    # create directory for results
    path = os.path.join("results", config_filename_wo_extension) 
    if not os.path.exists(path):
        os.mkdir(path) 

    # print results to csv file
    today = date.today()
    today_string = today.strftime("%Y_%b_%d") # Year_Month_Day, like 2019_Sept_16
    results_filename = "results/" + config_filename_wo_extension + "/" + config_filename_wo_extension + "_" + today_string + ".csv" 

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
                    errormessage = "Error: the config option '"+ckey+" : "+config[ckey]+"' is not known."
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
        
        tlimit = config["tlimit"]

        # read input data from config
        keys_to_extract = ['extended', 'obj', 'R', 'dist_bounds', 'state', 'level', 'base', 'heuristic', 'contiguity', 'order', 'obj_order', 'index']
        values = {k: config.get(k) for k in keys_to_extract}
        extended, obj, R, dist_bounds, state, level, base, heuristic, contiguity, order, obj_order, index = values.values()
        code = config_data['state_codes'][state]
        
        graph_json_file = f"data/{level}/dual_graphs/{level}{code}.json"
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
        for name, data in obj_func_data.items():
            objective_functions[name] = getattr(labeling, data['function'])
            start_labels[name] = data['start_label']

        bounds = None
        if order == "bvap":
            with open(f'data/bounds/{state}_bvap_bounds.json', 'r') as f:
                 bounds= json.load(f)

            bounds = hm.convert_keys_to_int(bounds)
            print(bounds)
            
        ############################################################
        # Load a prior solution  -  note: warm start is enacted later
        ############################################################
        if 'bvap_bounds' in obj or 'bvap' in order:
            X_ws, bvap_ws, vap_ws, obj_value_ws = ws.load_warm_start(state, obj, G, data, start_labels)
            print(bvap_ws)
            print(vap_ws)
            print(obj_value_ws)
            
        else:
            X_ws = ws.load_warm_start(state, obj, G, data, start_labels)
       
        
        result['LP_obj'] = 'n/a'
        result['LP_time'] = 'n/a'
        
                
        ############################
        # 
        ############################  

        ##  Gurobi Stuff  ####

        ############################
        # 
        ############################  
    


                    

        ####################################   
        # Summarize and export results including plotting
        ####################################  
        

        
        #load objective value of heuristic solution
        def terminate_on_warmstart_fail(model, where):
            if where == gp.GRB.Callback.MIPNODE:
                # Check if the warm start produced a solution
                if model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) in [gp.GRB.Status.INF_OR_UNBD, gp.GRB.Status.INFEASIBLE]:
                    # The warm start did not produce a solution
                    print("Warm start failed to produce a solution")
                    model.terminate()
        
        
        # Run the gradient cuts function
        result, m_bounds = run_gradient_cuts(config, G, DG, population, R, L, U, k, bvap_ws, vap_ws, X_ws, state, result, tlimit, obj_value_ws, tol, summarize_results, separation, bounds, config_filename_wo_extension)
        # Run integer optimization using all gradient cuts
        ## Can reuse the m_bounds model and just change the objective
        m = m_bounds
        
        # add logE objective 
        labeling.add_logE_objective(m, bounds, L, k)

        # add gradient cuts to the model
        for cut_num in result['gradient_bounds'].keys():
            grad_y = result['gradient_bounds'][cut_num]['grad_y']
            grad_z = result['gradient_bounds'][cut_num]['grad_z']
            gradient = apply_gradient(m._bvap, m._vap, grad_y, grad_z,k)
            gradient_ws = apply_gradient(bvap_ws, vap_ws, grad_y, grad_z,k)
            
            if gradient_ws < result['gradient_bounds'][cut_num]['ub']:
                m.addConstr( gradient <= result['gradient_bounds'][cut_num]['ub'], name = f"Grad_cut ub_{cut_num}"  )
            else:
                print(f"Error! ub in {cut_num} is violated by warmstart!")
                stop
                
            m.addConstr( gradient >= result['gradient_bounds'][cut_num]['lb'], 
                                      name = f"Grad_cut_lb{cut_num}"  )  
            
            
        m_bounds.Params.TimeLimit = tlimit
        m.update()
        
        
        
        start = time.time()
        m.optimize(separation.lcut_separation_generic)  #add callback function for contiguity
        end = time.time()    
            
        if summarize_and_export_results:
            fn = f"results/{config_filename_wo_extension}/{result['state']}-{result['level']}-{config['obj']}-{config['R']}"
            
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


