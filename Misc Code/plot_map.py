import matplotlib.pyplot as plt
import csv
import json
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib import cm
#from calculate_bvap import calc_bvap
import scipy.stats as stats
#from calculate_compactness import calc_compact


def plot_map_basic(args):
    '''
    Input: shape_file for the region you want to plot
           csv_file - describing the district assignments of the region

    Output: A map with the districts colored.
    '''
    
 
    fig_out_filename = args.csv_file[:-4]+"basic_map."+args.fig_out_type

    print("Creating a basic map plot")
    # Read generic US county level shapefile for mapping
    gdf = gpd.read_file(args.shape_file)

    #demo_csv = f"{state_code}_county_demographcs.csv"
    #csv_file = assignment_file

    # # Create Dictionary from json file
    # with open(json_file, "r") as f:
    #     assignment = {}
    #     json_data = json.load(f)
    #     district_list = []
    #     for key, val in json_data.items():
    #         for k, v in val.items():
    #             if int(v) == 1:
    #                 assignment.update({key: int(k)})
    #                 if int(k) not in district_list:
    #                     district_list.append(int(k))
    district_list=[]
    with open(args.csv_file, "r") as f:
        reader = csv.reader(f)
        # next(reader)
        assignment = {}
        for i, row in enumerate(reader):
            assignment.update({row[0]: int(row[1])})
            if int(row[1]) not in district_list:
                district_list.append(int(row[1]))
    #print(assignment)
    district_list = sorted(district_list)

    # Initialize 'District' field in geodataframe
    gdf['District'] = 0

    # Assign 'District' field to district assignment
    for key, val in assignment.items():
        i = gdf.index[gdf['GEOID10'] == key]
        gdf.loc[i,'District'] = int(val)

    # Delete counties from GDF that aren't assigned to a district
    for i,row in gdf.iterrows():
        if gdf.loc[i,'District'] == 0:
            gdf.drop(i,inplace=True)

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    n = 0
    col_maps = []
    average_bvap = 0
    sum_cdf = 0
    for districts in district_list:
        col_map = mpatches.Patch(color=(1.0, 1.0, 0.6, 1.0), label=f"{districts}")
        col_maps.append(col_map)
 

    # Plot map based on District Assignment
    #gdf.plot(column='District', edgecolor='black', cmap="Accent", ax=ax)
    gdf.plot(column='District', edgecolor=None, cmap='tab20', ax=ax)
    plt.axis("off")
    plt.savefig(fig_out_filename)
    plt.show()
    
def plot_map_data(args):
    '''
    Input: shape_file for the region you want to plot
           csv_file - describing the district assignments of the region

    Output: A map with the districts colored that includes all the calculations and legend.
    
    '''
     # Inputs
    state_code = "la"
    run_type = "bvap"
    num = "0"
    
    
    fig_out_filename = args.csv_file[:-4]+"data_map."+args.fig_out_type

    # Read shapefile for mapping
    gdf = gpd.read_file(args.shape_file)
    
    
    # # Create Dictionary from json file
    # with open(json_file, "r") as f:
    #     assignment = {}
    #     json_data = json.load(f)
    #     district_list = []
    #     for key, val in json_data.items():
    #         for k, v in val.items():
    #             if int(v) == 1:
    #                 assignment.update({key: int(k)})
    #                 if int(k) not in district_list:
    #                     district_list.append(int(k))
    district_list=[]
    with open(args.csv_file, "r") as f:
        reader = csv.reader(f)
        # next(reader)
        assignment = {}
        for i, row in enumerate(reader):
            assignment.update({row[0]: int(row[1])})
            if int(row[1]) not in district_list:
                district_list.append(int(row[1]))
    #print(assignment)
    district_list = sorted(district_list)
    
    
    # Get Bvap Ratios
    ratios = district_list# calc_bvap(demo_csv, assignment, district_list)

    # Get Compactness Scores
    compact_scores = district_list #calc_compact(json_file, assignment, district_list)

    # Initialize 'District' field in geodataframe
    gdf['District'] = 0

    # Assign 'District' field to district assignment
    for key, val in assignment.items():
        i = gdf.index[gdf['GEOID10'] == key]
        gdf.loc[i,'District'] = int(val)

    # Delete counties from GDF that aren't assigned to a district
    for i,row in gdf.iterrows():
        if gdf.loc[i,'District'] == 0:
            gdf.drop(i,inplace=True)

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    n = 0
    col_maps = []
    average_bvap = 0
    sum_cdf = 0
    for districts in district_list:
        col_map = mpatches.Patch(color=(1,0,0), label=f"{districts}: BVAP Ratio: {1}, BVAP: {1}, VAP: {1}, CDF: {1}")
        n = n + 1/(len(district_list) - 1)
        average_bvap = 1
        sum_cdf = 1
        col_maps.append(col_map)
    col_maps.append(mpatches.Patch(color="white", label=f"PP Score:{compact_scores}"))
    col_maps.append(mpatches.Patch(color="white",label=f"Average BVAP Ratio: {round(average_bvap/len(district_list),3)}, Average CDF: {round(sum_cdf/len(district_list), 3)}"))
    plt.legend(handles=[col_maps[i] for i in range(len(col_maps))], loc="lower left", bbox_to_anchor=(0, -0.25))

    # Plot map based on District Assignment
    #gdf.plot(column='District', edgecolor='black', cmap="Accent", ax=ax)
    gdf.plot(column='District', edgecolor=None, cmap="Accent", ax=ax)
    plt.axis("off")
    plt.savefig(fig_out_filename)
    plt.show()
    
    
def plot_map_calculations(args_plot):
    '''
    Input: shape_file for the region you want to plot
           csv_file - describing the district assignments of the region

    Output: A map with the districts colored that includes all the calculations and legend.
    
    '''
     # Inputs
    state_code = "la"
    run_type = "bvap"
    num = "0"

    # Input and Output file names
    fig_out_filename = f'{state_code}//{state_code}_{run_type}_plot_0{num}'
    json_file = f"C://Users//emurpha//Documents//Redistrict//states_county_json_with_pop//{state_code}_county_updated.json"
    demo_csv = f"{state_code}_county_demographcs.csv"
    csv_file = f'{state_code}//{state_code}_{run_type}_1.0k_0{num}.csv'

    # # Create Dictionary from json file
    # with open(json_file, "r") as f:
    #     assignment = {}
    #     json_data = json.load(f)
    #     district_list = []
    #     for key, val in json_data.items():
    #         for k, v in val.items():
    #             if int(v) == 1:
    #                 assignment.update({key: int(k)})
    #                 if int(k) not in district_list:
    #                     district_list.append(int(k))
    district_list=[]
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        # next(reader)
        assignment = {}
        for i, row in enumerate(reader):
            assignment.update({row[0]: int(row[1])})
            if int(row[1]) not in district_list:
                district_list.append(int(row[1]))
    print(assignment)
    district_list = sorted(district_list)

    # Get Bvap Ratios
    ratios = calc_bvap(demo_csv, assignment, district_list)

    # Get Compactness Scores
    compact_scores = calc_compact(json_file, assignment, district_list)

    # Initialize 'District' field in geodataframe
    gdf['District'] = 0

    # Assign 'District' field to district assignment
    for i,row in gdf.iterrows():
        for key, val in assignment.items():
            if key == gdf.at[i,'FIPS']:
                gdf.loc[i,'District'] = int(val)
    # Delete counties from GDF that aren't assigned to a district
    for i,row in gdf.iterrows():
        if gdf.loc[i,'District'] == 0:
            gdf.drop(i,inplace=True)

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    n = 0
    col_maps = []
    average_bvap = 0
    sum_cdf = 0
    for districts in district_list:
        col_map = mpatches.Patch(color=(1,0,0), label=f"{districts}: BVAP Ratio: {ratios[districts][0]}, BVAP: {round(ratios[districts][2])}, VAP: {round(ratios[districts][1])}, CDF: {round(stats.norm.cdf(8.26*ratios[districts][0] - 3.271),3)}")
        n = n + 1/(len(district_list) - 1)
        average_bvap = average_bvap + ratios[districts][0]
        sum_cdf = sum_cdf + stats.norm.cdf(8.26*ratios[districts][0] - 3.271)
        col_maps.append(col_map)
    col_maps.append(mpatches.Patch(color="white", label=f"PP Score:{compact_scores}"))
    col_maps.append(mpatches.Patch(color="white",label=f"Average BVAP Ratio: {round(average_bvap/len(district_list),3)}, Average CDF: {round(sum_cdf/len(district_list), 3)}"))
    plt.legend(handles=[col_maps[i] for i in range(len(col_maps))], loc="lower left", bbox_to_anchor=(0, -0.25))

    # Plot map based on District Assignment
    gdf.plot(column='District', edgecolor='black', cmap="Accent", ax=ax)
    plt.axis("off")
    plt.savefig(fig_out_filename)
    plt.show()
