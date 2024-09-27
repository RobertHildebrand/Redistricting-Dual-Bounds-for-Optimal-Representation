import json
import pandas as pd
import os
import helper_methods as hm


def summarize_soln(json_path):
    json_path = json_path.replace("'", "\"")
    if json_path[-5:] != ".json":
        json_path = json_path + ".json"
    print(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get the state abbreviation from the file name
    state = os.path.basename(json_path)[:2].lower()

    # Construct the paths to the CSV files
    voter_data_2016_path = f'data/county/PartisanData/{state}_cnty_census_2020_voter_data_2016.csv'
    voter_data_2020_path = f'data/county/PartisanData/{state}_cnty_census_2020_voter_data_2020.csv'
    demographics_path = f'data/county/PartisanData/{state}_cnty_census_2020_voter_data_2020_summarized.csv'

    # Define the columns that should be read as integers
    int_cols_2016 = ['R16', 'D16', 'L16']
    int_cols_2020 = ['R20', 'D20', 'L20']

    # Create a dictionary to hold the total demographics and voter data for each district
    district_totals = {}

    # Loop over the nodes in the JSON file
    for node in data['nodes']:
        # Get the district number for this node
        district = node['district']

        # Get the GEOID for this node
        geoid = node['GEOID']

        # If this district is not in the district_totals dictionary yet, add it
        if district not in district_totals:
            district_totals[district] = {
                'POP': 0,
                'BPOP': 0,
                'VAP': 0,
                'BVAP': 0,
                'BVAP_TOT': 0,
                'HVAP': 0,
                'ratio_BVAP': 0,
                'ratio_HVAP': 0,
                'R16': 0,
                'D16': 0,
                'L16': 0,
                'R20': 0,
                'D20': 0,
                'L20': 0,
                'Competiveness':0,
                'Compactness':0
            }

        # Read in the 2016 voter data CSV file and search for the row with the matching GEOID
        voter_data_2016 = pd.read_csv(voter_data_2016_path, dtype={col: float for col in int_cols_2016})
        voter_data_2016 = voter_data_2016.set_index('GEOID')
        if geoid in voter_data_2016.index:
            row = voter_data_2016.loc[geoid]
            district_totals[district]['R16'] += row['R16']
            district_totals[district]['D16'] += row['D16']
            district_totals[district]['L16'] += row['L16']

        # Read in the 2020 voter data CSV file and search for the row with the matching GEOID
        voter_data_2020 = pd.read_csv(voter_data_2020_path, dtype={col: float for col in int_cols_2020})
        voter_data_2020 = voter_data_2020.set_index('GEOID')
        if geoid in voter_data_2020.index:
            row = voter_data_2020.loc[geoid]
            district_totals[district]['R20'] += row['R20']
            district_totals[district]['D20'] += row['D20']
            district_totals[district]['L20'] += row['L20']


        # Read in the demographics CSV file and search for the row with the matching GEOID
        #demographics_path = f'data/county/Demos/{state}_bg_demographics_2020.csv'
        demographics = pd.read_csv(demographics_path)
        demographics = demographics.set_index('GEOID')

        if geoid in demographics.index:
            row = demographics.loc[geoid]
            # Add the demographic data to the total for this district
            district_totals[district]['POP'] += row['POP']
            district_totals[district]['BPOP'] += row['BPOP']
            district_totals[district]['VAP'] += row['VAP']
            district_totals[district]['BVAP'] += row['BVAP']
            district_totals[district]['BVAP_TOT'] += row['BVAP_TOT']
            district_totals[district]['HVAP'] += row['HVAP']

    districts = district_totals.keys()

    for district in districts:
        district_totals[district]['ratio_BVAP'] += district_totals[district]['BVAP'] / district_totals[district]['VAP']
        district_totals[district]['ratio_HVAP'] += district_totals[district]['HVAP'] / district_totals[district]['VAP']

        district_totals[district]['BVAP_reps'] = hm.calculate_black_rep(district_totals[district]['ratio_BVAP'])
        district_totals[district]['Dem_PVI'] = hm.pvi(district_totals[district])
        district_totals[district]['Dem_reps'] = hm.calculate_dem_prob_value(district_totals[district]['Dem_PVI'])
        district_totals[district]['Rep_reps'] = hm.calculate_rep_prob_value(district_totals[district]['Dem_PVI'])

    district_totals['Totals'] = {'BVAP_reps':sum(district_totals[district]['BVAP_reps'] for district in districts),
                                 'Dem_reps':sum(district_totals[district]['Dem_reps'] for district in districts),
                                 'Rep_reps':sum(district_totals[district]['Rep_reps'] for district in districts)}

    # Write the district_totals dictionary to a JSON file
    with open(json_path[:-5] + '_summarized_data.json', 'w') as f:
        json.dump(district_totals, f, indent=4)
        
    return district_totals

#district_totals = summarize_soln(folder_path, json_path)


import os
import fnmatch

def run_summarize():
    folder_path = ".." # specify the parent directory of the current directory
    for root, dirnames, filenames in os.walk(folder_path):
       # if not any('results' in dirname for dirname in dirnames):
           # continue
        for filename in filenames:
            if fnmatch.fnmatch(filename, '*.json') and 'summarize' not in filename:
                json_path = os.path.join(root, filename)
                if 'results-'in json_path or 'results_'in json_path:
                    if 'results\\results' not in json_path:
                        print(json_path)
                        summarize_soln(json_path)
#run_summarize()                

import os
import fnmatch
import json
import csv

def extract_summaries(folder_path = ".."):
    print(f"Extracting summaries from {folder_path}")
    
    # create a set of the unique first two letters
    state_set = set()
    for root, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if fnmatch.fnmatch(filename, '*_summarized_data.json'):
                
                state = filename[:2]
                state_set.add(state)
    print(f"Directies are: {dirnames}")
    # create separate sorted files for each unique state and value (BVAP, Dem, Reps)
    for state in state_set:
        bvap_list = []
        dem_list = []
        rep_list = []
        for root, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if fnmatch.fnmatch(filename, '*_summarized_data.json') and filename[:2] == state:
                    base_filename = filename.replace('_summarized_data.json', '')
                    full_path = os.path.join(root, filename).replace('_summarized_data.json', '.json')
                    with open(os.path.join(root, filename), 'r') as f:
                        data = json.load(f)
                    if 'Totals' in data.keys():
                        bvap = data['Totals']['BVAP_reps']
                        dem = data['Totals']['Dem_reps']
                        rep = data['Totals']['Rep_reps']
                    bvap_list.append((full_path, base_filename, bvap, dem, rep))
                    dem_list.append((full_path, base_filename, bvap, dem, rep))
                    rep_list.append((full_path, base_filename, bvap, dem, rep))

        bvap_list.sort(key=lambda x: x[2], reverse=True)
        dem_list.sort(key=lambda x: x[3], reverse=True)
        rep_list.sort(key=lambda x: x[4], reverse=True)
        
        
        outfile = f'results/objective_sorted/{state}_bvap_sorted.csv'
        with open(outfile, 'w', newline='') as f:
            print(f"Writing {outfile}")
            writer = csv.writer(f)
            writer.writerow(['full_path','filename', 'BVAP_reps', 'Dem_reps', 'Rep_reps'])
            for full_path, base_filename, bvap, dem, rep in bvap_list:
                writer.writerow([full_path, base_filename, bvap, dem, rep])
       
        outfile = f'results/objective_sorted/{state}_dem_sorted.csv'
        with open(outfile, 'w', newline='') as f:
            print(f"Writing {outfile}")
            writer = csv.writer(f)
            writer.writerow(['full_path','filename', 'BVAP_reps', 'Dem_reps', 'Rep_reps'])
            for full_path, base_filename, bvap, dem, rep in dem_list:
                writer.writerow([full_path, base_filename, bvap, dem, rep])
        
        outfile = f'results/objective_sorted/{state}_rep_sorted.csv'
        with open(outfile, 'w', newline='') as f:
            print(f"Writing {outfile}")
            writer = csv.writer(f)
            writer.writerow(['full_path','filename', 'BVAP_reps', 'Dem_reps', 'Rep_reps'])
            for full_path, base_filename, bvap, dem, rep in rep_list:
                writer.writerow([full_path, base_filename, bvap, dem, rep])
    return bvap_list 
