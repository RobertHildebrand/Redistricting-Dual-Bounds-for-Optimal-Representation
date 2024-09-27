"""
Data Export and Visualization Functions
Adapted Version of Validi's code (extracted from main.py) 
Date: 2023-04-17

This file contains several utility functions for exporting and visualizing computational results related to districting, including:
- Writing summaries of results to a CSV file
- Exporting districting solutions to a JSON file
- Exporting district maps and max B set maps to PNG files
- Plotting and saving maps based on specified columns
- Plotting Black Voting Age Population (BVAP) ratios for each district and saving them as images

Original Author: Validi
Adapted by: Hildebrand
"""

import matplotlib.pyplot as plt
from csv import DictWriter
import json
import helper_methods as hm
from summarize import *
import networkx as nx
import geopandas as gpd
import time
import pprint

################################################
# Summarize computational results to csv file
################################################ 


def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)
        
        
# ################################################
# # Writes districting solution to json file
# ################################################ 

def export_to_json(G, districts, filename):
    print("Exporting solution to json")
    with open(filename, 'w') as outfile:
        soln = {}
        soln['nodes'] = []
        for j in range(len(districts)):
            for i in districts[j]:
                soln['nodes'].append({
                        'name': G.nodes[i]["NAME20"],
                        'index': i,
                        'GEOID20': G.nodes[i]['GEOID20'],
                        'GEOID': G.nodes[i]['GEOID'],
                        'district': j
                        })
        json.dump(soln, outfile, indent=4)
        

def export_to_png(G, gdf, districts, filename):
    """
    Exports district map to a PNG file.
    
    Args:
    G (networkx.Graph): Graph of the geographic area.
    gdf (geopandas.GeoDataFrame): GeoDataFrame containing geographic data.
    districts (list): List of districts.
    filename (str): Name of the output PNG file.
    """
    assignment = [next((j for j, district in enumerate(districts) if G.nodes[i]["GEOID20"] in [G.nodes[node]["GEOID20"] for node in district]), -1) for i in G.nodes]

    if min(assignment) < 0:
        print("Error: did not assign all nodes in district map png.")
    else:
        gdf['assignment'] = assignment
        plot_and_save_map(gdf, 'assignment', filename)

        dissolved = gdf.dissolve(by='assignment', aggfunc={'P0030004': 'sum', 'P0030001': 'sum'})
        plot_bvap(gdf, filename, 'heatmap', overlay=True, gdf2=dissolved)
        plot_bvap(dissolved, filename)

def export_B_to_png(G, gdf, B, filename):
    """
    Exports max B set map to a PNG file.
    
    Args:
    G (networkx.Graph): Graph of the geographic area.
    gdf (geopandas.GeoDataFrame): GeoDataFrame containing geographic data.
    B (list): List of nodes in the max B set.
    filename (str): Name of the output PNG file.
    """
    gdf['B'] = [1 if G.nodes[u]["GEOID20"] in [G.nodes[i]["GEOID20"] for i in B] else 0 for u in G.nodes]
    plot_and_save_map(gdf, 'B', filename)

def plot_and_save_map(gdf, column, filename):
    """
    Plots and saves a map with the specified column to a PNG file.

    Args:
    gdf (geopandas.GeoDataFrame): GeoDataFrame containing geographic data.
    column (str): Column to use for plotting.
    filename (str): Name of the output PNG file.
    """
    my_fig = gdf.plot(column=column).get_figure()
    my_fig.set_size_inches(my_fig.get_size_inches() * 3)
    plt.axis('off')
    my_fig.savefig(filename)


def plot_bvap(dissolved, filename, additional_file_name='', overlay=False, gdf2=0):
    """
    Plots Black Voting Age Population (BVAP) ratio for each district.
    
    Args:
    dissolved (geopandas.GeoDataFrame): DataFrame with dissolved districts.
    filename (str): Name of the output PNG file.
    additional_file_name (str): Additional name to append to the output file. Default is an empty string.
    overlay (bool): Whether to overlay a second GeoDataFrame (gdf2). Default is False.
    gdf2 (geopandas.GeoDataFrame): Second GeoDataFrame to overlay. Default is 0.
    """
    dissolved['Representative'] = [coords[0] for coords in dissolved['geometry'].apply(lambda x: x.representative_point().coords[:])]
    dissolved = dissolved.rename(columns={'P0010001': 'POP', 'P0010004': 'BPOP', 'P0030001': 'VAP', 'P0030004': 'BVAP', 'P0040002': 'HVAP'})
    dissolved['BVAP_ratio'] = dissolved['BVAP'] / dissolved['VAP']
    colormap = 'BuPu'

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    cbax = fig.add_axes([0.95, 0.3, 0.03, 0.39])
    cbax.set_title("BVAP Ratio")
    vmin, vmax = 0, 0.75
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    fig.colorbar(sm, cax=cbax)
    ax.axis("off")
    
    dissolved.plot(column='BVAP_ratio', edgecolor='black', cmap=colormap, ax=ax, vmin=vmin, vmax=vmax, legend=False)
    dissolved.reset_index(inplace=True)

    dissolved.apply(lambda x: ax.annotate(text=f"{x.name}: {round(hm.calculate_black_rep(x.BVAP_ratio), 2)}", xy=x.Representative, ha='center', color='r', weight='bold', size=14), axis=1)
    
    if not overlay:
        ratios = list(dissolved['BVAP_ratio'])
        score = sum(hm.calculate_black_rep(r) for r in ratios)
        ax.set_title(f"BVAP Score: {score}")

    if overlay:
        gdf2.plot(ax=ax, edgecolor='orange', linewidth=2, facecolor='none')

    plt.savefig(f"{filename.split('.png')[0]}_BVAP{additional_file_name}.png", transparent=True)
    plt.savefig(f"{filename.split('.png')[0]}_BVAP{additional_file_name}.png", transparent=True)


def summarize_results(result, m, G, DG, k, fn, level, state, base, obj, start, end, expErr=None, maxErr=None, results_filename="results.csv", my_fieldnames=None):
    print("Results Summary started")
    result.update({'MIP_time': '{0:.2f}'.format(end-start), 'MIP_status': int(m.status), 'MIP_nodes': int(m.NodeCount), 'MIP_bound': m.objBound, 'callbacks': m._numCallbacks, 'lazy_cuts': m._numLazyCuts})
    
    m.write(fn+'.lp')
    if obj == "step-exp" or obj == "step-max":
        print("Is this the error?")
        #result.update({'expected_error': expErr, 'maximum_error': maxErr})

    # report best solution found
    if m.SolCount > 0:
        print("Recording found solution")
        result['MIP_obj'] = m.objVal

        if base == 'hess':
            labels = [ j for j in DG.nodes if m._X[j,j].x > 0.5 ]
        else: # base == 'labeling'
            labels = [ j for j in range(k) ]

        districts = [ [ i for i in DG.nodes if m._X[i,j].x > 0.5 ] for j in labels]
        print("best solution (found) =",districts)

        # export solution to .json file
        print("print write mip soln")
        print(fn+'.sol')
        m.write(fn+'.sol')
        
        json_fn = fn + ".json"
        print("Exporting to json")
        print(json_fn)
        export_to_json(G, districts, json_fn)

        
        print(f'Summarizing {json_fn}')
        district_totals = summarize_soln(json_fn)

        # export solution to .png file (districting map)
        png_fn = fn + ".png"
        # read shapefile
        gdf = gpd.read_file("data/"+level+"/shape_files/"+state+"_"+level+".shp")
        print("Plotting")
        export_to_png(G, gdf, districts, png_fn)

        # is solution connected?
        connected = True
        for district in districts:
            if not nx.is_connected(G.subgraph(district)):
                connected = False
        result['connected'] = connected

    else:
        result.update({'MIP_obj': 'no_solution_found', 'connected': 'n/a'})

    # Summarize results of this run to csv file
    if my_fieldnames is None:
        my_fieldnames = list(result.keys())

    for name in list(result.keys()):
        if name not in my_fieldnames:
            my_fieldnames.append(name)
    #######################


    def print_formatted_result(result):
        """
        Prints the 'result' dictionary in a nicely formatted way, focusing on the 'gradient_bounds' key.
        Each element of the sub-dictionary under 'gradient_bounds' will be printed on a separate line.
        The rest of the dictionary will be printed as usual.

        Args:
            result (dict): The result dictionary containing 'gradient_bounds' as one of the keys.
        """
        # Check if the key 'gradient_bounds' exists in the result dictionary
        if 'gradient_bounds' in result:
            print("\nFormatted Gradient Bounds:")
            gradient_bounds = result['gradient_bounds']

            # Iterate over each key-value pair in the gradient_bounds sub-dictionary
            for key, sub_dict in gradient_bounds.items():
                print(f"\nGradient Bound {key}:")
                # Check if the sub_dict is also a dictionary
                if isinstance(sub_dict, dict):
                    # Print each item in the sub-dictionary on a new line
                    for sub_key, value in sub_dict.items():
                        print(f"  {sub_key}: {value}")
                else:
                    # If not a dictionary, print directly
                    print(f"  {sub_dict}")
        else:
            print("\nThe 'gradient_bounds' key does not exist in the result dictionary.")

        # Print the rest of the result dictionary, excluding 'gradient_bounds'
        print("\nRest of the Result Dictionary:")
        rest_of_result = {k: v for k, v in result.items() if k != 'gradient_bounds'}

        # Pretty print the rest of the dictionary without 'sort_dicts'
        pprint.pprint(rest_of_result)

    # Example usage
    # Assuming 'result' is your dictionary
    print_formatted_result(result)


    
    
    ########################
    print(f"Adding to csv {results_filename}")
    append_dict_as_row(results_filename,result,my_fieldnames)
    


