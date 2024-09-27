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
import hess, labeling, ordering, fixing, separation
# Plotting and export imports
from my_exports import *
from helper_methods import *
import warm_start as ws


def load_config_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data



def main(config_path, summarize_and_export_results = True, return_bvap_vap_soln_and_bounds= False, load_config = True):

    ###############################################
    # Reading config
    ############################################### 
    if load_config:
        if config_path == None:
            print("No config path!")
            return 0

        print("Reading config from",config_path)    
        config_filename_wo_extension = config_path.rsplit('.',1)[0]
        configs_file = open(config_path,'r')
        batch_configs = json.load(configs_file)
        configs_file.close()
    else:
        batch_configs = config_path # in this case, user input a dictionary that the config would have presented.
       
        
       
    
    ###############################################
    # Load common data
    ############################################### 
    
    config_data = load_config_data('config_data.json')
    
    ###############################################
    # Assign common data for use as it is coded
    ############################################### 
    state_codes = config_data['state_codes']
    number_of_congressional_districts = config_data['number_of_congressional_districts']
    default_config = config_data['default_config']
    available_config = config_data['available_config']
    available_config['state'] = set(state_codes.keys())
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
    print(batch_configs.keys())
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
        extended, obj, R, dist_bounds, state, level, base, heuristic, contiguity, order, obj_order, index = values.values()
        code = state_codes[state]
        
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
        k = number_of_congressional_districts[state]        
        population = {node : G.nodes[node]['POP100'] for node in G.nodes()}  
        deviation = .2
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


        ############################
        # Build base model
        ############################   

        m = gp.Model()
        m._DG = DG

        if base == 'hess':
            # X[i,j]=1 if vertex i is assigned to (district centered at) vertex j
            m._X = m.addVars(DG.nodes, DG.nodes, vtype=GRB.BINARY)
            hess.add_base_constraints(m, population, L, U, k)

        if base == 'labeling':        
            # X[i,j]=1 if vertex i is assigned to district j in {0,1,2,...,k-1}
            m._X = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY, name="X")
            if config['symmetry']=='orbitope' or config['contiguity'] in {'scf', 'shir', 'lcut'}:
                m._R = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY)
            labeling.add_base_constraints(m, population, L, U, k)


        ############################################
        # Add (extended?) objective 
        ############################################


        if base == 'hess':
            if extended:
                hess.add_extended_objective(m, G)
            else:
                hess.add_objective(m, G)
        print(config)

        if base == 'labeling':
            
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
                print(bounds)
                def convert_keys_to_int(obj):
                    if isinstance(obj, dict):
                        new_obj = {}
                        for key, value in obj.items():
                            if isinstance(value, dict):
                                new_obj[key] = convert_keys_to_int(value)
                            elif isinstance(value, list):
                                new_obj[key] = [convert_keys_to_int(item) for item in value]
                            else:
                                try:
                                    new_obj[int(key)] = value
                                except ValueError:
                                    new_obj[key] = value
                        return new_obj
                    elif isinstance(obj, list):
                        return [convert_keys_to_int(item) for item in obj]
                    else:
                        return obj

                bounds = convert_keys_to_int(bounds)
                print(bounds)
            
            if order == "D20":
                bounds = None # need to fill this in still.  But these bounds haven't been created yet.
                
            if obj in objective_functions:
                if obj != 'bvap_bounds':
                    function_result = objective_functions[obj](m, G, k, R, U, bounds)
                if obj == 'bvap_bounds':
                    function_result = objective_functions[obj](m, G, k, R, U, bounds, obj_order,index)

    
    ############################################################
    # Load a prior solution  -  note: warm start is enacted later
    ############################################################
    
                            
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

        ####################################   
        # Contiguity constraints
        ####################################      

        m._callback = None
        m._population = population
        m._U = U
        m._k = k
        m._base = base
        m._numLazyCuts = 0
        m._numCallbacks = 0

        if base == 'hess':
            if contiguity == 'shir':
                hess.add_shir_constraints(m)
            elif contiguity == 'scf':
                hess.add_scf_constraints(m, G, extended)
            elif contiguity == 'lcut':
                m.Params.lazyConstraints = 1
                m._callback = separation.lcut_separation_generic

        if base == 'labeling':
            if contiguity == 'shir':
                labeling.add_shir_constraints(m, config['symmetry'])
            elif contiguity == 'scf':
                labeling.add_scf_constraints(m, G, extended, config['symmetry'])
            elif contiguity == 'lcut':
                m.Params.lazyConstraints = 1
                m._callback = separation.lcut_separation_generic 

        m.update()


        ############################################
        # Vertex ordering and max B problem 
        ############################################  

        
        # if order == "none", then nothing is done.  This is important when other orders are prescribed.
        if order == 'B_decreasing' or order == 'default':
            if order == 'B_decreasing':
                (B, result['B_q'], result['B_time'], result['B_timelimit']) = ordering.solve_maxB_problem(DG, population, L, k, heuristic_districts)

                # draw set B on map and save
                fn_B = "../" + "results_for_" + config_filename_wo_extension + "/" + result['state'] + "-" + result['level'] + "-maxB.png"

                # read shapefile
                gdf = gpd.read_file("../data/"+level+"/shape_files/"+state+"_"+level+".shp")
                print("Order is B decreasing, so we made a plot...")
                export_B_to_png(G, gdf, B, fn_B)
            else:
                (B, result['B_q'], result['B_time'], result['B_timelimit']) = (list(),'n/a','n/a', 'n/a')

            result['B_size'] = len(B)

            vertex_ordering = ordering.find_ordering(order, B, DG, population)
            position = ordering.construct_position(vertex_ordering)

            print("Vertex ordering =", vertex_ordering)  
            print("Position vector =", position)
            print("Set B =", B)

        ####################################   
        # Symmetry handling
        ####################################    

        symmetry = config['symmetry']

        if symmetry == 'aggressive':
            m.Params.symmetry = 2
        elif symmetry == 'orbitope':
            if base == 'labeling':
                labeling.add_orbitope_extended_formulation(m, G, k, vertex_ordering)
            else:
                sys.exit("Error: orbitope only available for labeling base model.")     

        #m.Params.symmetry = 0

        ####################################   
        # Variable fixing
        ####################################    

        do_fixing = config['fixing']

        if do_fixing and base == 'hess':
            result['DFixings'] = fixing.do_Hess_DFixing(m, G, position)
            result['UFixings_R'] = 'n/a'

            if contiguity == 'none':
                result['LFixings'] = fixing.do_Hess_LFixing_without_Contiguity(m, G, population, L, vertex_ordering)
                result['UFixings_X'] = fixing.do_Hess_UFixing_without_Contiguity(m, G, population, U)
            else:
                result['LFixings'] = fixing.do_Hess_LFixing(m, G, population, L, vertex_ordering)
                result['UFixings_X'] = fixing.do_Hess_UFixing(m, DG, population, U, vertex_ordering)         

            if extended:
                result['ZFixings'] = fixing.do_Hess_ZFixing(m, G)
            else:
                result['ZFixings'] = 0


        if do_fixing and base == 'labeling':
            result['DFixings'] = fixing.do_Labeling_DFixing(m, G, vertex_ordering, k)

            if contiguity == 'none':
                if symmetry == 'orbitope':
                    result['LFixings'] = fixing.do_Labeling_LFixing_without_Contiguity(m, G, population, L, vertex_ordering, k)
                else:
                    result['LFixings'] = 0
                (result['UFixings_X'], result['UFixings_R']) = fixing.do_labeling_UFixing_without_Contiguity()
            else:
                result['LFixings'] = fixing.do_Labeling_LFixing(m, G, population, L, vertex_ordering, k)
                (result['UFixings_X'], result['UFixings_R']) = fixing.do_Labeling_UFixing(m, DG, population, U, vertex_ordering, k)

            if extended:
                result['ZFixings'] = fixing.do_Labeling_ZFixing(m, G, k)
            else:
                result['ZFixings'] = 0

        if not do_fixing:
            result['DFixings'] = 0
            result['UFixings_R'] = 0
            result['LFixings'] = 0
            result['UFixings_X'] = 0
            result['ZFixings'] = 0


        ######################################################################################
        # Solve root LP? Used only for reporting purposes. Not used for MIP solve.
        ######################################################################################  

        if config['lp']:
            r = m.relax() # LP relaxation of MIP model m
            r.Params.LogToConsole = 0 # keep log to a minimum
            r.Params.Method = 3 # use concurrent LP solver
            r.Params.TimeLimit = 3600 # one-hour time limit for solving LP
            print("To get the root LP bound, now solving a (separate) LP model.")

            lp_start = time.time()
            r.optimize()
            lp_end = time.time()

            if r.status == GRB.OPTIMAL:
                result['LP_obj'] = '{0:.2f}'.format(r.objVal)
            elif r.status == GRB.TIME_LIMIT:
                result['LP_obj'] = 'TL'
            else:
                result['LP_obj'] = '?'
            result['LP_time'] = '{0:.2f}'.format(lp_end - lp_start)

        else:
            result['LP_obj'] = 'n/a'
            result['LP_time'] = 'n/a'


        ####################################   
        # Inject heuristic warm start
        ####################################    

        ws.inject_X_ws(m,k,X_ws)

        ####################################   
        # Solve MIP
        ####################################  

        result['MIP_timelimit'] = config['tlimit'] 
        m.Params.TimeLimit = result['MIP_timelimit']
        m.Params.Method = 3 # use concurrent method for root LP. Useful for degenerate models

        fn = "../" + "results_for_" + config_filename_wo_extension + "/" + result['state'] + "-" + result['level'] + "-" + config['obj'] + "-" + str(config['R'])
        m.params.LogFile= fn+".log"

        #m.setParam('LogToConsole', 0)
        start = time.time()
        m.optimize(m._callback)
        end = time.time()
        

        ####################################   
        # Summarize and export results including plotting
        ####################################  
        
        
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
        
        
        ####################################   
        # Returns for subproblems for bvap optimization
        ####################################  
        
        if return_bvap_vap_soln_and_bounds:

            result.update({'MIP_time': '{0:.2f}'.format(end-start), 
                           'MIP_status': int(m.status), 
                           'MIP_nodes': int(m.NodeCount), 
                           'MIP_bound': m.objBound, 
                           'callbacks': m._numCallbacks, 
                           'lazy_cuts': m._numLazyCuts})
    

            # report best solution found
            if m.SolCount > 0:
                print("Recording found solution")
                result['MIP_obj'] = m.objVal
                result['bvap'] = {i: m.getVarByName(f'bvap{i}').x for i in range(k)}
                result['vap'] = {i: m.getVarByName(f'vap{i}').x for i in range(k)}
            return result
        
        
#         vap_soln = [629679.0, 537662.0, 595591.0, 583282.0, 631354.0, 517860.0, 519040.0]
#         bvap_soln = [89262.0, 92355.0, 108679.0, 109767.0, 131180.0, 217520.0, 222581.0]
#         import helper_methods as hm
#         import numpy as np
#         vap_values = [m.getVarByName(f'vap{i}').x for i in range(7)]
#         bvap_values =  [m.getVarByName(f'bvap{i}').x for i in range(7)]
#         print("vap_value = ",vap_values)
#         print("bvap_value = ",bvap_values)
#         grad_y, grad_z = hm.grad_bvap(bvap_values, vap_values)
#         grad_y =list(np.array(grad_y)*10**7)
#         grad_z = list(np.array(grad_z)*10**7)
#         print(f" \"grad_y\" : {grad_y}, \"grad_z\" : {grad_z}")
#         print("Soln val: ", sum(bvap_values[i]*grad_y[i] + vap_values[i]*grad_z[i] for i in range(k)))
#         print("Warm start val: ", sum(bvap_soln[i]*grad_y[i] + vap_soln[i]*grad_z[i] for i in range(k)))
#         print("ratios", [m.getVarByName(f'bvap{i}').x/m.getVarByName(f'vap{i}').x for i in range(7)])
        
#         vap_difference = [m.getVarByName(f'vap{i}').x - vap_soln[i] for i in range(7)]
#         bvap_difference = [m.getVarByName(f'bvap{i}').x - bvap_soln[i] for i in range(7)]
#         print("vap difference", vap_difference)
#         print("bvap difference", bvap_difference)
#         print("ratios", [m.getVarByName(f'bvap{i}').x/m.getVarByName(f'vap{i}').x for i in range(7)])
        print("func", [hm.f_bvap(m.getVarByName(f'bvap{i}').x/m.getVarByName(f'vap{i}').x) for i in range(7)])
        #print("cdf", [m.getVarByName(f'cdf{i}').x for i in range(7)])   
    return result


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


