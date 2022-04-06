"""
This file plots a scatter plot of compactness vs black representation

Requirements:
calculate_compactness & calculate_bvap files
pandas library

Inputs:
state_code: state code for state that you want to map.
run_types: a list of the file prefixes (i.e. compact, bvap, bvap_pp, etc.)
block_level: county, block group, etc.
num_runs: range of number of runs

Outputs:
HTML Interactive Plot of distance between maps with a reference map serving as the center point

Author: Emily Murphy, Summer 2020
"""
import matplotlib.pyplot as plt
import csv
from calculate_bvap import calc_bvap
import scipy.stats as stats
from calculate_compactness import calc_compact

# Inputs
state_code = "ms"
run_types = ["bvap", "compact", "bvap_pp", "bvap_pp_bweight", "bvap_pp_pweight", "bvap_pp_bbweight"]
block_level = "county"
num_runs = range(10)

# Input and Output file names
json_file = f"states_county_json_with_pop//{state_code}_county_updated.json"
demo_csv = f"{state_code}_county_demographcs.csv"

# Initialize lists
bvap_avg = []
compact_avg = []
label_list = []

# Iterate through solution library files
for run_type in run_types:
    for num in num_runs:
        bvap_ratios = []
        csv_file = f'Solution Library//{state_code}//{state_code}_{run_type}_1.0k_0{num}.csv'

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            # next(reader)
            assignment = {}
            district_list = []
            # Create district assignment dictionary
            for i, row in enumerate(reader):
                assignment.update({row[0]: int(row[1])})
                # Create list of districts
                if int(row[1]) not in district_list:
                    district_list.append(int(row[1]))
        f.close()
        district_list = sorted(district_list)

        # Get Bvap Ratios
        ratios = calc_bvap(demo_csv, assignment, district_list)
        for list in ratios.values():
            bvap_ratios.append(list[0])

        # Get Compactness Scores
        compact_scores = calc_compact(json_file, assignment, district_list)

        # Get Averages
        mean_bvap = sum(bvap_ratios)/len(bvap_ratios)
        bvap_cdf = stats.norm.cdf(8.26*mean_bvap - 3.271)
        mean_compact = sum(compact_scores.values())/len(compact_scores.values())

        # Add values to lists
        bvap_avg.append(bvap_cdf)
        compact_avg.append(mean_compact)
        label_list.append(f"{run_type}:{num}")

# Plot with matplotlib library
# plt.scatter(bvap_avg, compact_avg, label=label_list)
# plt.annotate(label_list)
# plt.title(f'{state_code.upper()} Mean Compactness vs. Mean BVAP CDF Score')
# plt.xlabel('Mean BVAP CDF Score')
# plt.ylabel('Mean Polsby Popper Score')
# plt.show()

import plotly.express as px
fig = px.scatter(x=bvap_avg, y=compact_avg)
fig.update_layout(title=f'{state_code.upper()} {block_level.capitalize()} Level Compactness vs. Black Representation',
                    xaxis_title="Mean BVAP CDF Score",
                    yaxis_title="Mean Polsby Popper Score")
# Create an interactive plot
fig.write_html(f"{state_code}_{block_level}_comparison.html")

