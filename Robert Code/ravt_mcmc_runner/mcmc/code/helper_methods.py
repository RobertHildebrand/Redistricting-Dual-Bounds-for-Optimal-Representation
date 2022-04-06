import scipy.stats as stats
import math
norm = stats.norm(0,1)


def calculate_black_reps(partition):
    '''
    Return the expected number of black representatives via the formula:
    
    sum(norm.cdf(8.26*x-3.271) for x in partition['BVAP_ratio'].values())
    
    '''
    #return sum(norm.cdf(8.26*x-3.271) for x in partition['BVAP_ratio'].values())
    return sum(norm.cdf(6.826*x-2.827) for x in partition['BVAP_ratio'].values())


def calculate_compactness(partition):
    '''
    Returns the polsby_popper compactness score (average over districts) of a partition.
    '''
    c = float(len(partition))
    return  sum(partition['polsby_popper'].values())/c

def check_equal_population(partition, population_tolerance):
    number_of_districts = len(partition['population'].keys())
    total_population = sum(partition['population'].values())
    ideal_population = total_population / number_of_districts
    dev = []
    is_equal = []
    for val in partition['population'].values():
        deviation = (val - ideal_population) / ideal_population
        if abs(deviation) > population_tolerance:
            is_equal.append(False)
        else:
            is_equal.append(True)
        dev.append(abs(deviation))

    avg_deviation = sum(dev) / number_of_districts
    print(f"Average deviation: {avg_deviation}")
    print(f"Max deviation: {max(dev)}")
    return all(item == True for item in is_equal)


def calculate_pop_deviation(partition, pop_tolerance):
    number_of_districts = len(partition['population'].keys())
    total_population = sum(partition['population'].values())
    avg_population = total_population / number_of_districts

    deviation = []

    for district_population in partition['population'].values():
        diff = max(abs(district_population - avg_population) - pop_tolerance * avg_population, 0)
        deviation.append(diff)

    return sum(deviation)

def count_county_splits(partition):
    cs = partition['county_splits']
    return sum(cs[county][0].value for county in cs.keys())

def count_county_splits_total(partition):
    cs = partition['county_splits']
    return sum(len(cs[county][2]) for county in cs.keys()) - len(cs.keys())


def calculate_dem_prob(partition):
    return sum(norm.cdf(pvi / 4.8) for pvi in partition['Dem_PVI'].values())



def calculate_rep_prob(partition):
    return sum(1 - norm.cdf(pvi / 4.8) for pvi in partition['Dem_PVI'].values())


def calculate_competitive(partition):
    return sum(math.exp(-(pvi / 4.8)**2) for pvi in partition['Dem_PVI'].values())
