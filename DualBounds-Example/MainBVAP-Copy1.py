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



###############################################
## Custom Imports
###############################################

# Model Imports
import labeling, ordering, fixing, separation, bvap_models, bvap

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
        

        
        def build_relax_model(G, DG, population, R, k, bvap_ws, vap_ws):
            ''' Explain what this is .... !!!!'''
            
            
            m = gp.Model()
           
            bvap_models.add_bvap_vap_continuous(m, G, k, U, bounds, comparison = False, bvap_ordering = True)
            # Adds the LogE Objective [Change this?]
            bvap_models.add_logE_objective(m, bounds, R, k)

            # Inject heuristic warm start

            inject_bvap_ws(m,k,bvap_ws, vap_ws)
              
            # Finish setting up parameters for MIP
            m.Params.Method = 3 # use concurrent method for root LP. Useful for degenerate models

            fn = f"results/{config_filename_wo_extension}/{result['state']}-{result['level']}-{config['obj']}-{config['R']}"
            m.params.LogFile = f"{fn}.log"
            return m

        # Inject heuristic warm start
        def inject_X_ws(m,k,X_ws):            
            for j in range(k):
                for i in X_ws[j]:
                    m._X[i,j].start = 1
        # Inject heuristic warm start
        def inject_bvap_ws(m,k,bvap_ws, vap_ws):            
            for j in range(k):
                m._bvap[j].start = bvap_ws[j]
                m._vap[j].start = vap_ws[j]
                    
        def build_bounds_model(G, DG, population, L, U, k, heuristic_districts):
            ''' Explain what this is .... !!!!'''
            
            m = gp.Model()
            m._DG = DG
            # X[i,j]=1 if vertex i is assigned to district j in {0,1,2,...,k-1}
            m._X = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY, name="X")
            m._R = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY) # for lcut separation

            labeling.add_base_constraints(m, population, L, U, k)            
            labeling.add_bvap_vap(m, G, k, U, bounds, comparison = False, bvap_ordering = True)
            #labeling.add_logE_objective(m, bvap, vap, bounds, L, k)

            m._numLazyCuts = 0
            m._numCallbacks = 0
            m.Params.lazyConstraints = 1
            m._base = "labeling"
            inject_X_ws(m,k,X_ws)
              
            # Finish setting up parameters for MIP
            m.Params.Method = 3 # use concurrent method for root LP. Useful for degenerate models

            fn = f"results/{config_filename_wo_extension}/{result['state']}-{result['level']}-{config['obj']}-{config['R']}"
            m.params.LogFile = f"{fn}.log"
            return m
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
        
        
        #########################################
        ###         Gradient Cuts             ###
        #########################################
        def apply_gradient(bvap, vap, grad_y, grad_z,k):
            return sum(bvap[i]*grad_y[i] + vap[i]*grad_z[i] for i in range(k))
        
        #Set number of gradient cuts
        num_grad_cuts = config["numGCuts"]
        
        # Build continuous relaxation in just y,z space.
        obj_cts = "LogEPWL_bvap_bounds_continuous"
        
        m_relax = build_relax_model(G, DG, population, R, k, bvap_ws, vap_ws)
        m_relax.Params.TimeLimit = tlimit
        
        # Build bounds optimization model
        m_bounds = build_bounds_model(G, DG, population, L, U, k, X_ws)
        


        # Define the path to the file
        filename = f"data/bounds/{state}_bvap_gradient_bounds.json"

        # Check if the file exists
        if os.path.isfile(filename):
            # Print
            print("Loading existing gradient cuts!!")
            # Load the existing data
            with open(filename, 'r') as f:
                existing_data = json.load(f)

            # Find the maximum key
            max_key = len(existing_data.keys())
            print(f"Loading {max_key} many prior gradient cuts!")

            # Start the counter at the next key
            counter = int(max_key)
            
            # initial with prior information
            result['gradient_bounds'] = existing_data
            
            # Save a copy of the existing data with a timestamp in the filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"data/bounds/{state}_bvap_gradient_bounds_{timestamp}.json"
            with open(backup_filename, 'w') as f:
                json.dump(existing_data, f)

        else:
            # If the file does not exist, start the counter at 0
            print("Not loading any prior cuts")
            counter = 0
            result['gradient_bounds'] = {}

        if counter > 0:
            for cut_num in existing_data.keys():
                # note: grad_y and grad_z should be stored from computation below for cut_num >= 1. 
                
                grad_y= result['gradient_bounds'][cut_num]['grad_y']
                grad_z= result['gradient_bounds'][cut_num]['grad_z']
                gradient_relax = apply_gradient(m_relax._bvap, m_relax._vap, grad_y, grad_z,k)
                
                m_relax.addConstr( gradient_relax <= result['gradient_bounds'][cut_num]['ub'], 
                                          name = f"Grad_cut ub_{cut_num}"  )
                m_relax.addConstr( gradient_relax >= result['gradient_bounds'][cut_num]['lb'], 
                                          name = f"Grad_cut_lb{cut_num}"  )  
                m_relax.update()
                
                gradient_bounds = apply_gradient(m_bounds._bvap, m_bounds._vap, grad_y, grad_z, k)

                m_bounds.addConstr(gradient_bounds <= result['gradient_bounds'][cut_num]['ub'], 
                                    name=f"Grad_cut_ub_{cut_num}")
                m_bounds.addConstr(gradient_bounds >= result['gradient_bounds'][cut_num]['lb'], 
                                    name=f"Grad_cut_lb_{cut_num}")
                m_bounds.update()

        
        for cut_num in range(counter, num_grad_cuts + counter):
           
            # Add prior cuts to the relaxation model
            if cut_num > counter:
                # note: grad_y and grad_z should be stored from computation below for cut_num >= 1. 
                gradient_relax = apply_gradient(m_relax._bvap, m_relax._vap, grad_y, grad_z,k)
                
                m_relax.addConstr( gradient_relax <= result['gradient_bounds'][cut_num-1]['ub'], 
                                          name = f"Grad_cut ub_{cut_num-1}"  )
                m_relax.addConstr( gradient_relax >= result['gradient_bounds'][cut_num-1]['lb'], 
                                          name = f"Grad_cut_lb{cut_num-1}"  )  
                m_relax.update()
                
                gradient_bounds = apply_gradient(m_bounds._bvap, m_bounds._vap, grad_y, grad_z, k)

                m_bounds.addConstr(gradient_bounds <= result['gradient_bounds'][cut_num-1]['ub'], 
                                    name=f"Grad_cut_ub_{cut_num-1}")
                m_bounds.addConstr(gradient_bounds >= result['gradient_bounds'][cut_num-1]['lb'], 
                                    name=f"Grad_cut_lb_{cut_num-1}")
                m_bounds.update()

            
            
            inject_bvap_ws(m_relax,k,bvap_ws, vap_ws)
            start = time.time()
            m_relax.optimize()
            #m_relax.optimize(terminate_on_warmstart_fail) # No callback needed
            end = time.time()
            
            
            result['gradient_bounds'][cut_num] = {}
            result['gradient_bounds'][cut_num]['Relaxation computation (s)'] = end - start
            result['gradient_bounds'][cut_num]['Relaxation_bound'] = m_relax.objBound
            
            print(f"Relaxation optimized in {end - start} seconds")
            print(f"Best relaxation bound is {m_relax.objBound}")
            print(f"Error in relaxation bound is {m_relax.objBound - float(obj_value_ws)}")
            if m_relax.objBound - float(obj_value_ws) < tol:
                print("Within desired tolerance from relaxation bound")
                break
            
            # get bvap and vap solns
            vap_relax_soln = [m_relax._vap[i].x for i in range(k)]
            bvap_relax_soln =  [m_relax._bvap[i].x for i in range(k)]
            
            # get the gradient
            grad_y, grad_z = hm.grad_bvap(bvap_relax_soln, vap_relax_soln)
            
            print(grad_y, grad_z)
            grad_y, grad_z = hm.rescale_gradients(grad_y, grad_z)
            
            print(f" \"grad_y\" : {grad_y}, \"grad_z\" : {grad_z}")
            print("Soln val: ", apply_gradient(bvap_relax_soln, vap_relax_soln, grad_y, grad_z,k))
            print("Warm start val: ", apply_gradient(bvap_ws, vap_ws, grad_y, grad_z,k))
            print("ratios", [m_relax._bvap[i].x/m_relax._vap[i].x for i in range(k)])
            
            # store the point and gradient based on these things
            result['gradient_bounds'][cut_num]['y'] = bvap_relax_soln
            result['gradient_bounds'][cut_num]['z'] = vap_relax_soln
            result['gradient_bounds'][cut_num]['grad_y'] = list(grad_y)
            result['gradient_bounds'][cut_num]['grad_z'] = list(grad_z)
            result['gradient_bounds'][cut_num]['soln_value'] = apply_gradient(bvap_relax_soln, vap_relax_soln, grad_y, grad_z,k)
            result['gradient_bounds'][cut_num]['Warm_start_val'] = apply_gradient(bvap_ws, vap_ws, grad_y, grad_z,k)

            
            
            # Run maximization
            gradient_objective = apply_gradient(m_bounds._bvap, m_bounds._vap, grad_y, grad_z,k)
            m_bounds.setObjective(gradient_objective, GRB.MAXIMIZE)
            m_bounds.Params.TimeLimit = config["timeGCuts"]
            inject_X_ws(m_bounds,k,X_ws)
            m_bounds.update()
            start = time.time()
            m_bounds.optimize(separation.lcut_separation_generic) # add callback function and optimize
            end = time.time()
            # Should we do anything with the solution from this optimization?  Find cuts from this point?  Store it as warm start option? 
            ## There is probably something useful to do with this point, but perhaps not a necessary enhancement at this time.
            
            # return results
            
            result['gradient_bounds'][cut_num]['ub'] = m_bounds.objBound + 1 # add 1 due to round off errors (due do scaling of the gradients ahead of time, this + 1 should be sufficient to maintain correctness of the solution.
            result['gradient_bounds'][cut_num]['ub_computation (s)'] = end - start
            result['gradient_bounds'][cut_num]['ub objective value'] = m_bounds.getObjective().getValue()
            if result['gradient_bounds'][cut_num]['ub objective value'] > float(obj_value_ws):
                obj_value_ws = result['gradient_bounds'][cut_num]['ub objective value'].getValue()
                result['gradient_bounds'][cut_num]['ub soln'] = [(i,j,m._X[i,j].x) for i in range(k) for j in G.nodes()]
                intermediate_summarize = results_filename + f'gradient_bounds_max{cut_num}'
                summarize_results(result, m_bounds, G, DG, k, fn, level, state, base, obj, start, end, results_filename=intermediate_summarize, my_fieldnames=my_fieldnames)
            
            # Run minimization
            m_bounds.setObjective(gradient_objective, GRB.MINIMIZE) # change to Minimize
            m_bounds.Params.TimeLimit = config["timeGCuts"] # set run time limit
            inject_X_ws(m_bounds,k,X_ws)
            m_bounds.update()
            start = time.time()
            m_bounds.optimize(separation.lcut_separation_generic)  #add callback function for contiguity
            end  = time.time()
            # return results
            result['gradient_bounds'][cut_num]['lb'] = m_bounds.objBound - 1 # subtract 1 due to round off errors 
            result['gradient_bounds'][cut_num]['lb_computation (s)'] = end - start
            
            print(f"Results of gradient cut {cut_num}")
            print(result['gradient_bounds'][cut_num])
            
            # Catch errors:
            if result['gradient_bounds'][cut_num]['Warm_start_val'] > result['gradient_bounds'][cut_num]['ub']:
                print("main_bvap.py: Warm start val exceeds gradient upper bound")
                stop
            if result['gradient_bounds'][cut_num]['Warm_start_val'] < result['gradient_bounds'][cut_num]['lb']:
                print("main_bvap.py: Warm start val is less than gradient lower bound") 
                stop
                
            # Break loop if bounds are not helpful    
            if result['gradient_bounds'][cut_num]['lb'] < 0 < result['gradient_bounds'][cut_num]['ub']:
                print("Improved bounds failed.  Relaxation solution contained in MIP bounds")
                break

        # Write the JSON object to the file
        with open(f"data/bounds/{state}_bvap_gradient_bounds.json", 'w') as f:
            print(result['gradient_bounds'])
            json.dump(result['gradient_bounds'], f)
        
        
        # Plot graient bounds progress
        import matplotlib.pyplot as plt
        iterations = list(result['gradient_bounds'].keys())
        relax_bounds = [result['gradient_bounds'][cut_num]['Relaxation_bound'] for cut_num in iterations]
        plt.plot(iterations, relax_bounds)
        plt.plot(iterations, [obj_value_ws]*len(iterations), 'g-')
        plt.title("Bound improvement over gradient cut iterations")
        # Set the y-axis limits and ticks
        plt.ylim(min(relax_bounds), max(relax_bounds))
        plt.yticks([min(relax_bounds), max(relax_bounds)])
        plt.show()
        
        # Run one round of gradient cuts from best known solution
        # (Skip this for now)
        
        m_relax.write(f"{state}_relaxation_lp.lp")
        
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


