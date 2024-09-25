#!/usr/bin/env python
# coding: utf-8

# # Virginia Redistricting Optimization
# This code is setup to compute maps for Georgia at the Block Group Level
# 
# 
# The main cells that need to be run are the 
# - Import lots of functions
# - Input settings
# 
# Then you can run your choice of optimizing on a single objective, or varying the weights
# 
# Later you can choose to plot one of the solutions.
# 
# Lastly, we create a plot BVAP/Polsby-Popper for solutions that have been created.
# 

# ## Import lots of functions

# In[1]:


from mcmc_runner_new_notebook_version import *
from acceptance_functions import *
from argparse import Namespace


# ## Create json file from shapefile
# This should only need to be run once per dataset.
# Ideally, these are already prepared and this never needs to be run.
# 
# 

# In[ ]:


### These lines are commented since this should not need to be run. 
### Only run if shapefile has not yet been parsed into a json.

# filename = '../../data/shape_files/51/tl_2010_51_bg10.shp'
# G = parse_shapefile(filename, no_json=False)
# G.to_json("va_tl_2010_51_bg10.json")


# ## Input settings

# In[2]:

    
def run_optimization(acceptance_function, num_runs, start_num):

    args = Namespace(accept='mixed_objective_redefined_later', 
                     #config_file='example_settings.ini', 
                     data=['../../data/sample_data/block_group_demographics.csv GEOID VAP VAP',
                           '../../data/sample_data/block_group_demographics.csv GEOID POP POP',
                           '../../data/sample_data/block_group_demographics.csv GEOID BVAP BVAP',
                          '../../data/sample_data/block_group_demographics.csv GEOID D12 D1',
                          '../../data/sample_data/block_group_demographics.csv GEOID D16 D2', 
                           '../../data/sample_data/block_group_demographics.csv GEOID R12 R1',
                          '../../data/sample_data/block_group_demographics.csv GEOID R16 R2',
                          ], 
                     districts=11, 
                     geometry='../../data/json_files/va_tl_2010_51_bg10.json', 
                     #geometry = 'sample_data/tl_2010_13_bg10/tl_2010_13_bg10.shp',  # Prefer not to use .shp here.
                     input_districts=None, 
                     no_json=False, 
                     number_of_runs=num_runs,
                     outfolder='../outputs/VA', 
                     partition='semi_random_split', 
                     population_tolerance=0.01, 
                     accept=acceptance_function,
                     prefix='VA_bg_', 
                     start=start_num, 
                     steps=25000,
                     summary_file='../outputs/VA/VA_summary_compertitive_compact.csv')
    
    
    
    args.prefix = fixed_prefix + args.accept
    #args.prefix = fixed_prefix+ f"bvap{bvap_weight}_pp{pp_weight}" 
    
    # Runs the code    
    output_data = run_the_code(args, acceptance)
    
    # Restores the original prefix
    args.prefix = fixed_prefix
    
    


# In[4]:



