# gradient_cuts.py
import os
import json
import time
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Correct import for ticker

import warm_start as ws
import helper_methods as hm
import bvap_models
import labeling

def build_relax_model(G, DG, population, U, R, k, bvap_ws, vap_ws, bounds, config_filename_wo_extension, result, config):
    ''' This model does not include the X binary variables.  Instead, this model only acts 
    in the space of the bvap and vap variables.  Since the objective function is only interms
    of these variables, it makes sense to study the projection of the feasible region into this space.
    We use this model to add gradient cuts to and then to find upper bounds on optimal solutions to the 
    original problem.'''


    m = gp.Model()

    bvap_models.add_bvap_vap_continuous(m, G, k, U, bounds, comparison = False, bvap_ordering = True)
    # Adds the LogE Objective (the choice of objective is not incredibly important since this relaxation will solve quickly.)
    bvap_models.add_logE_objective(m, bounds, R, k)

    # Inject heuristic warm start

    ws.inject_bvap_ws(m,k,bvap_ws, vap_ws)

    # Finish setting up parameters for MIP
    m.Params.Method = 3 # use concurrent method for root LP. Useful for degenerate models

    fn = f"results/{config_filename_wo_extension}/{result['state']}-{result['level']}-{config['obj']}-{config['R']}"
    m.params.LogFile = f"{fn}.log"
    return m


def build_bounds_model(G, DG, population, L, U, k, heuristic_districts, bounds, config_filename_wo_extension, result, config, X_ws):
    ''' This model inclues the X binary decision variables.  This is meant to have the same
    feasible region as the original problem, but then we will optimize linear objectives over 
    this space instead of our nonlinear objective that we care about.
    
    Thus, this version of the model will help us find deep cuts.
    '''

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
    ws.inject_X_ws(m,k,X_ws)

    # Finish setting up parameters for MIP
    m.Params.Method = 3 # use concurrent method for root LP. Useful for degenerate models

    fn = f"results/{config_filename_wo_extension}/{result['state']}-{result['level']}-{config['obj']}-{config['R']}"
    m.params.LogFile = f"{fn}.log"
    return m
        
def apply_gradient(bvap, vap, grad_y, grad_z, k):
    """Applies gradient cuts to the given bvap and vap values."""
    return sum(bvap[i] * grad_y[i] + vap[i] * grad_z[i] for i in range(k))

def run_gradient_cuts(config, G, DG, population, R, L, U, k, bvap_ws, vap_ws, X_ws, state, result, tlimit, obj_value_ws, tol, summarize_results, separation, bounds, config_filename_wo_extension):
    """
    Executes the gradient cuts procedure.

    Parameters:
        config (dict): Configuration settings from JSON.
        G, DG (Graph): Graph structures used in the optimization.
        population (dict): Population data for the optimization.
        R, L, U (various types): Various parameters needed for the optimization models.
        k (int): Number of regions or divisions.
        bvap_ws, vap_ws (list): Initial solutions for BVAP and VAP.
        X_ws (list): Warm start solutions.
        state (str): State identifier.
        result (dict): Dictionary to store the results.
        tlimit (int): Time limit for the solver.
        obj_value_ws (float): Objective value for warm start.
        tol (float): Tolerance level for convergence checks.
        build_bounds_model (functions): Functions to build models.
        summarize_results (function): Function to summarize results.
        separation (module): Module handling separations in models.
    """
    # Set the number of gradient cuts
    num_grad_cuts = config["numGCuts"]

    # Build relaxation model and bounds model
    m_relax = build_relax_model(G, DG, population, U, R, k, bvap_ws, vap_ws, bounds, config_filename_wo_extension, result, config)
    m_relax.Params.TimeLimit = tlimit

    m_bounds = build_bounds_model(G, DG, population, L, U, k, X_ws, bounds, config_filename_wo_extension, result, config, X_ws)

    # Define the path to the gradient bounds file
    filename = Path(f"data/bounds/{state}_bvap_gradient_bounds.json")

    # Load or initialize gradient bounds data
    if filename.is_file():
        print("Loading existing gradient cuts!!")
        with filename.open('r') as f:
            existing_data = json.load(f)
        max_key = len(existing_data.keys())
        counter = max_key
        result['gradient_bounds'] = existing_data
        # Backup the existing data with a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_filename = Path(f"data/bounds/{state}_bvap_gradient_bounds_{timestamp}.json")
        with backup_filename.open('w') as f:
            json.dump(existing_data, f)
    else:
        print("Not loading any prior cuts")
        counter = 0
        result['gradient_bounds'] = {}

    # Load prior gradient cuts into the models if any exist
    if counter > 0:
        for cut_num, data in existing_data.items():
            grad_y = data['grad_y']
            grad_z = data['grad_z']
            gradient_relax = apply_gradient(m_relax._bvap, m_relax._vap, grad_y, grad_z, k)

            m_relax.addConstr(gradient_relax <= data['ub'], name=f"Grad_cut_ub_{cut_num}")
            m_relax.addConstr(gradient_relax >= data['lb'], name=f"Grad_cut_lb_{cut_num}")
            m_relax.update()

            gradient_bounds = apply_gradient(m_bounds._bvap, m_bounds._vap, grad_y, grad_z, k)

            m_bounds.addConstr(gradient_bounds <= data['ub'], name=f"Grad_cut_ub_{cut_num}")
            m_bounds.addConstr(gradient_bounds >= data['lb'], name=f"Grad_cut_lb_{cut_num}")
            m_bounds.update()

    # Main loop for adding new gradient cuts
    for cut_num in range(counter, num_grad_cuts + counter):
        # Add previous cuts to the models
        if cut_num > counter:
            gradient_relax = apply_gradient(m_relax._bvap, m_relax._vap, grad_y, grad_z, k)
            m_relax.addConstr(gradient_relax <= result['gradient_bounds'][cut_num - 1]['ub'], name=f"Grad_cut_ub_{cut_num - 1}")
            m_relax.addConstr(gradient_relax >= result['gradient_bounds'][cut_num - 1]['lb'], name=f"Grad_cut_lb_{cut_num - 1}")
            m_relax.update()

            gradient_bounds = apply_gradient(m_bounds._bvap, m_bounds._vap, grad_y, grad_z, k)
            m_bounds.addConstr(gradient_bounds <= result['gradient_bounds'][cut_num - 1]['ub'], name=f"Grad_cut_ub_{cut_num - 1}")
            m_bounds.addConstr(gradient_bounds >= result['gradient_bounds'][cut_num - 1]['lb'], name=f"Grad_cut_lb_{cut_num - 1}")
            m_bounds.update()

        # Inject warm start solutions and optimize the relaxation model
        ws.inject_bvap_ws(m_relax, k, bvap_ws, vap_ws)
        start = time.time()
        m_relax.optimize()
        end = time.time()

        # Update results with the relaxation bound
        result['gradient_bounds'][cut_num] = {
            'Relaxation computation (s)': end - start,
            'Relaxation_bound': m_relax.objBound
        }

        # Check convergence
        if m_relax.objBound - float(obj_value_ws) < tol:
            print("Within desired tolerance from relaxation bound")
            break

        # Get solutions and compute gradients
        vap_relax_soln = [m_relax._vap[i].x for i in range(k)]
        bvap_relax_soln = [m_relax._bvap[i].x for i in range(k)]
        grad_y, grad_z = hm.grad_bvap(bvap_relax_soln, vap_relax_soln)
        grad_y, grad_z = hm.rescale_gradients(grad_y, grad_z)

        # Store results
        result['gradient_bounds'][cut_num].update({
            'y': bvap_relax_soln,
            'z': vap_relax_soln,
            'grad_y': list(grad_y),
            'grad_z': list(grad_z),
            'soln_value': apply_gradient(bvap_relax_soln, vap_relax_soln, grad_y, grad_z, k),
            'Warm_start_val': apply_gradient(bvap_ws, vap_ws, grad_y, grad_z, k)
        })

        # Optimize bounds model for upper and lower bounds
        for direction, goal in [('ub', GRB.MAXIMIZE), ('lb', GRB.MINIMIZE)]:
            m_bounds.setObjective(apply_gradient(m_bounds._bvap, m_bounds._vap, grad_y, grad_z, k), goal)
            m_bounds.Params.TimeLimit = config["timeGCuts"]
            ws.inject_X_ws(m_bounds, k, X_ws)
            start = time.time()
            m_bounds.optimize(separation.lcut_separation_generic)
            end = time.time()

            bound_adjustment = 1 if direction == 'ub' else -1
            result['gradient_bounds'][cut_num][f'{direction}'] = m_bounds.objBound + bound_adjustment
            result['gradient_bounds'][cut_num][f'{direction}_computation (s)'] = end - start
            result['gradient_bounds'][cut_num][f'{direction} objective value'] = m_bounds.getObjective().getValue()

        # Save results to file
        with filename.open('w') as f:
            json.dump(result['gradient_bounds'], f)

    # Plot the progress of bounds improvement
    iterations = list(result['gradient_bounds'].keys())
    relax_bounds = [result['gradient_bounds'][i]['Relaxation_bound'] for i in iterations]
     
    # Define the width factor, e.g., 0.5 inches per iteration (adjust as needed)
    width_factor = 0.25

    # Calculate the width based on the number of iterations
    plot_width = max(10, len(iterations) * width_factor)  # Set a minimum width to avoid extremely narrow plots

    # Create the figure with the calculated width
    plt.figure(figsize=(plot_width, 6))  # 6 is the height; you can adjust it if needed

    # Plot the data
    plt.plot(iterations, relax_bounds, label='Relaxation Bounds')
    plt.plot(iterations, [obj_value_ws] * len(iterations), 'g-', label='Objective Value WS')

    # Set the title and labels
    plt.title("Bound Improvement Over Gradient Cut Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Bounds")

    # Set the y-axis limits
    plt.ylim(min(relax_bounds), max(relax_bounds))

    # Add a grid for better scale visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set y-axis ticks automatically and format them to ensure values are displayed
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Rotate the x-axis ticks
    plt.xticks(rotation=-90)

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()
    return result, m_bounds