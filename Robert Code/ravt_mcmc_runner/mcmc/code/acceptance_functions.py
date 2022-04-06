
## Initially created by Matt Pierson
## Modified by Robert Hildebrand - January 2021
##
## This file stores all acceptance functions to be passed to the MCMC_runner.py file to use.



import helper_methods
import math


def compactness(partition):
    '''
    Polsby Popper compactness acceptance function
    '''
    c = float(len(partition))
    pp_current = sum(partition['polsby_popper'].values())/c
    pp_prev = (sum(partition.parent['polsby_popper'].values())/c)
    return pp_current >= pp_prev
    
def black_representatives(partition):
    c = float(len(partition))
    reps_current = helper_methods.calculate_black_reps(partition)/c
    reps_prev = helper_methods.calculate_black_reps(partition.parent)/c
    #reps_current = sum(norm.cdf(8.26*x-3.271) for x in partition['BVAP_ratio'].values())/c
    #reps_prev = sum(norm.cdf(8.26*x-3.271) for x in partition.parent['BVAP_ratio'].values())/c
    return reps_current >= reps_prev
    
def bvap_pp(partition):
    '''
    Equally weighted bvap and Polsby Popper acceptance function
    '''
    c = float(len(partition))
    reps_current = helper_methods.calculate_black_reps(partition)/c
    reps_prev = helper_methods.calculate_black_reps(partition.parent)/c
    #reps_current = sum(norm.cdf(8.26*x-3.271) for x in partition['BVAP_ratio'].values())/c
    #reps_prev = sum(norm.cdf(8.26*x-3.271) for x in partition.parent['BVAP_ratio'].values())/c
    pp_current = sum(partition['polsby_popper'].values())/c
    pp_prev = (sum(partition.parent['polsby_popper'].values())/c)
    return (reps_current+pp_current) >= (reps_prev+pp_prev)

def bvap_pp_weights(partition,u,v):
    '''
    weighted bvap and polsby popper objective function
    weights: u * bvap + v * pp
    '''
    c = float(len(partition))
    reps_current = helper_methods.calculate_black_reps(partition)/c
    reps_prev = helper_methods.calculate_black_reps(partition.parent)/c
    #reps_current = sum(norm.cdf(8.26*x-3.271) for x in partition['BVAP_ratio'].values())/c
    #reps_prev = sum(norm.cdf(8.26*x-3.271) for x in partition.parent['BVAP_ratio'].values())/c
    pp_current = sum(partition['polsby_popper'].values())/c
    pp_prev = (sum(partition.parent['polsby_popper'].values())/c)
    return (u*reps_current+v*pp_current) >= (u*reps_prev+v*pp_prev)
    
def equal_pop(partition):
    ''' 
    Accepts only if solution is within a population tolerance of 0.02
    '''
    population_tolerance = 0.02
    eq_pop = helper_methods.check_equal_population(partition, population_tolerance)
    print(eq_pop)
    return eq_pop

def equal_pop_2(partition):
    ''' 
    Accepts when distance to equal population imroves
    '''
    population_tolerance = 0
    eq_pop_current = helper_methods.calculate_pop_deviation(partition, population_tolerance)
    eq_pop_prev = helper_methods.calculate_pop_deviation(partition.parent, population_tolerance)
    # distance_between = Pair(partition, partition.parent).distance
    print(f"Current Pop Dev: {eq_pop_current}")
    print(f"Previous Pop Dev: {eq_pop_prev}")
    print(f"Current better than Previous: {eq_pop_current <= eq_pop_prev}")
    # print(f"Distance between partitions: {distance_between}")
    return eq_pop_current <= eq_pop_prev

def dem_gerrymander_compact(partition):
    ''' 
    Accepts when number of expected democratic representatives improves
    '''
    dem_reps_current = helper_methods.calculate_dem_prob(partition)
    dem_reps_prev = helper_methods.calculate_dem_prob(partition.parent)
    c = float(len(partition))
    pp_current = sum(partition['polsby_popper'].values())/c
    pp_prev = (sum(partition.parent['polsby_popper'].values())/c)
    print(f"Current Dem Reps Probability: {dem_reps_current}")
    print(f"Previous Dem Reps Probability: {dem_reps_prev}")
    print(f"Current better than Previous: {dem_reps_current >= dem_reps_prev}")
    return 0.9*dem_reps_current+0.1*pp_current >= 0.9*dem_reps_prev + 0.1*pp_prev 

def dem_gerrymander(partition):
    ''' 
    Accepts when number of expected democratic representatives improves
    '''
    dem_reps_current = helper_methods.calculate_dem_prob(partition)
    dem_reps_prev = helper_methods.calculate_dem_prob(partition.parent)
    print(f"Current Dem Reps Probability: {dem_reps_current}")
    print(f"Previous Dem Reps Probability: {dem_reps_prev}")
    print(f"Current better than Previous: {dem_reps_current >= dem_reps_prev}")
    return dem_reps_current >= dem_reps_prev

def rep_gerrymander_compact(partition):
    ''' 
    Accepts when number of expected republican representatives improves
    '''
    rep_reps_current = helper_methods.calculate_rep_prob(partition)
    rep_reps_prev = helper_methods.calculate_rep_prob(partition.parent)
    c = float(len(partition))
    pp_current = sum(partition['polsby_popper'].values())/c
    pp_prev = (sum(partition.parent['polsby_popper'].values())/c)
    #print(f"Current Rep Reps Probability: {rep_reps_current}")
    #print(f"Previous Rep Reps Probability: {rep_reps_prev}")
    #print(f"Current better than Previous: {rep_reps_current >= rep_reps_prev}")
    return 0.9*rep_reps_current+0.1*pp_current >= 0.9*rep_reps_prev+ 0.1*pp_prev

def rep_gerrymander(partition):
    ''' 
    Accepts when number of expected republican representatives improves
    '''
    rep_reps_current = helper_methods.calculate_rep_prob(partition)
    rep_reps_prev = helper_methods.calculate_rep_prob(partition.parent)
    #print(f"Current Rep Reps Probability: {rep_reps_current}")
    #print(f"Previous Rep Reps Probability: {rep_reps_prev}")
    #print(f"Current better than Previous: {rep_reps_current >= rep_reps_prev}")
    return rep_reps_current >= rep_reps_prev

def competitive_reps_compact(partition):
    ''' 
    Accepts when competitiveness score improves
    '''
    comp_reps_current = helper_methods.calculate_competitive(partition)
    comp_reps_prev = helper_methods.calculate_competitive(partition.parent)
    c = float(len(partition))
    pp_current = sum(partition['polsby_popper'].values())/c
    pp_prev = (sum(partition.parent['polsby_popper'].values())/c)
    #print(f"Current Competitiveness: {comp_reps_current}")
    #print(f"Previous Competitiveness: {comp_reps_prev}")
    #print(f"Current better than Previous: {comp_reps_current >= comp_reps_prev}")
    return 0.9*comp_reps_current+0.1*pp_current >= 0.9*comp_reps_prev+ 0.1*pp_prev

def competitive_reps(partition):
    ''' 
    Accepts when competitiveness score improves
    '''
    comp_reps_current = helper_methods.calculate_competitive(partition)
    comp_reps_prev = helper_methods.calculate_competitive(partition.parent)
    #print(f"Current Competitiveness: {comp_reps_current}")
    #print(f"Previous Competitiveness: {comp_reps_prev}")
    #print(f"Current better than Previous: {comp_reps_current >= comp_reps_prev}")
    return comp_reps_current >= comp_reps_prev

def county_splits_objective(partition):
    ''' 
    Accepts when splits score improves
    '''
    comp_splits_current = helper_methods.count_county_splits(partition)
    comp_splits_prev = helper_methods.count_county_splits(partition.parent)

    return comp_splits_prev >= comp_splits_current

def county_splits_compact(partition):
    ''' 
    Accepts when competitiveness score improves
    '''
    comp_reps_current = helper_methods.count_county_splits(partition)
    comp_reps_prev = helper_methods.count_county_splits(partition.parent)
    c = float(len(partition))
    pp_current = sum(partition['polsby_popper'].values())/c
    pp_prev = (sum(partition.parent['polsby_popper'].values())/c)

    comp_splits_current = helper_methods.count_county_splits(partition)
    comp_splits_prev = helper_methods.count_county_splits(partition.parent)

    return 0.9*comp_splits_prev+0.1*pp_current >= 0.9*comp_splits_current+0.1*pp_prev