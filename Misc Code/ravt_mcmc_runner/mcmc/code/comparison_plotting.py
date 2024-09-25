# This just loads the summary file into a pandas dataframe.  
# Note: Sometimes there are errors in the summary file that cause a parser error here

import pandas as pd
import plotly.express as px


def comparison_plot_bvap_pp(summary_file, state_code, block_level):
    # Load the data from the summary file
    df_summary = pd.read_csv(summary_file)

    #bvap_avg = list(df_summary['mean_BVAP'])
    black_reps = list(df_summary['black_reps'])
    compact_avg = list(df_summary['polsby_popper'])



    # Create the plot
    #fig = px.scatter(x=black_reps, y=compact_avg)

    fig = px.scatter(df_summary, x="black_reps", y="polsby_popper", hover_data=['outfile_name','pop_tolerance'])
    fig.update_layout(title=f'{state_code.upper()} {block_level.capitalize()} Level Compactness vs. Black Representation',
                        xaxis_title="Expected Black Representatives",
                        yaxis_title="Mean Polsby Popper Score")
    # Create an interactive plot and save as html
    fig.write_html(f"outputs/{state_code}/{state_code}_{block_level}_comparison.html")
    fig.show()

    return df_summary # display the data (you could also just open the .csv file in excel if you want to look at it)