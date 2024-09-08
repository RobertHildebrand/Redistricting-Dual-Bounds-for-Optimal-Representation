import scipy.stats as stats
import math

norm = stats.norm(0,1)


# A.) For rim south states (MD, VA, NC, TN):
# Pr(Black representative) = phi (-4.194 + (BVAP*0.0975) + (HVAP*0.0300)

# B.) For deep south states (AL, GA, LA, MS, SC): 
# Pr(Black representative) = phi (-4.729 + (BVAP*0.1044) + (HVAP*0.0300)

def calculate_black_rep(ratio):
    return norm.cdf(6.826*ratio-2.827)

def calculate_black_reps(partition):
    #return sum(norm.cdf(8.26*x-3.271) for x in partition['BVAP_ratio'].values()) # 2010 values
    return sum(calculate_black_rep(x) for x in partition['BVAP_ratio'].values()) # 2020 values
    #return sum(norm.cdf(10.67*x-4.81) for x in partition['BVAP_ratio'].values()) # 2020 Southern states values


def calculate_black_reps_rim_south(partition):
    return sum(norm.cdf(-4.194+bvap*9.75+ hvap*3) for (bvap,hvap) in zip(partition['BVAP_ratio'].values(),partition['HVAP_ratio'].values()) ) # 2020 Southern states values

def calculate_black_reps_deep_south(partition):
    return sum(norm.cdf(-4.729+bvap*10.44+ hvap*3) for (bvap,hvap) in zip(partition['BVAP_ratio'].values(),partition['HVAP_ratio'].values()) ) # 2020 Southern states values



######
def calculate_black_reps_list(partition):
    #return sum(norm.cdf(8.26*x-3.271) for x in partition['BVAP_ratio'].values()) # 2010 values
    #return sum(norm.cdf(6.826*x-2.827) for x in partition['BVAP_ratio'].values()) # 2020 values
    return [norm.cdf(10.67*x-4.81) for x in partition['BVAP_ratio'].values] # 2020 Southern states values


def calculate_black_reps_rim_south_list(partition):
    return [norm.cdf(-4.194+bvap*9.75+ hvap*3) for (bvap,hvap) in zip(partition['BVAP_ratio'].values,partition['HVAP_ratio'].values)] # 2020 Southern states values

def calculate_black_reps_deep_south_list(partition):
    return [norm.cdf(-4.729+bvap*10.44+ hvap*3) for (bvap,hvap) in zip(partition['BVAP_ratio'].values,partition['HVAP_ratio'].values) ] # 2020 Southern states values

def calculate_dem_prob_list(partition):
    return [norm.cdf(pvi / 4.8) for pvi in partition['Dem_PVI'].values]

def calculate_rep_prob_list(partition):
    return [1 - norm.cdf(pvi / 4.8) for pvi in partition['Dem_PVI'].values]

def calculate_competitive_list(partition):
    return [math.exp(-(pvi / 4.8)**2) for pvi in partition['Dem_PVI'].values]


def calculate_num_competitive_list(partition):
    ''' Competitive districts are those with pvi between -5 and 5'''
    return [abs(pvi)<= 5 for pvi in partition['Dem_PVI'].values]

def pvi_list(df, bias = 51.69):
    return 50*df['D20']/(df['D20'] + df['R20']) + 50*df['D16']/(df['D16'] + df['R16']) - bias

def pvi(df, bias = 51.69):
    return 50*df['D20']/(df['D20'] + df['R20']) + 50*df['D16']/(df['D16'] + df['R16']) - bias

#######

def calculate_dem_prob_value(pvi):
    return norm.cdf(pvi / 4.8)

def calculate_rep_prob_value(pvi):
    return 1 - norm.cdf(pvi / 4.8)


def calculate_dem_prob(partition):
    ''' This estimates the expected number of democratic representives '''
    return sum(norm.cdf(pvi / 4.8) for pvi in partition['Dem_PVI'].values())

def calculate_rep_prob(partition):
    '''  This estimates the expected number of reublican representives '''
    return sum(1 - norm.cdf(pvi / 4.8) for pvi in partition['Dem_PVI'].values())

def calculate_competitive(partition):
    ''' This provides a measure of competitiveness in a district '''
    return sum(math.exp(-(pvi / 4.8)**2) for pvi in partition['Dem_PVI'].values())

def calculate_num_competitive(partition):
    ''' Competitive districts are those with pvi between -5 and 5'''
    return sum(abs(pvi)<= 5 for pvi in partition['Dem_PVI'].values())


#Cdf function
def cdf_fun(x):
    return stats.norm.cdf(6.826*x-2.827)

def pdf_fun(x):
    return stats.norm.pdf(6.826*x-2.827)

def f_bvap(x):
    return cdf_fun(x)

def CPVI_fun(x):
    if x < 0.07:
        return 0
    return stats.norm.cdf((50*x-51.69)/4.8)

def Comp_fun(x):
    if x < 0.07:
        return 0
    return math.exp(-((50*x-51.69)/4.8)**2)
    
def rho(r):
    return sum(  cdf_fun(r[i])*(r[i]-r[i-1])  for i in range(1,len(r))  )

def grad(r):
    grad = [  6.826*pdf_fun(r[i])*(r[i]-r[i-1])+cdf_fun(r[i])-cdf_fun(r[i+1])  for i in range(1,len(r)-1)  ]
    grad.append(  6.826*pdf_fun(r[-1])*(r[-1]-r[-2])+cdf_fun(r[-1])  )
    grad.insert(  0, 0-cdf_fun(r[1])  )
    return grad
     
def grad_bvap(y,z):
    grad_y = [pdf_fun(y[i]/z[i])/z[i] for i in range(len(y))]
    grad_z = [-pdf_fun(y[i]/z[i])*y[i]/z[i]**2 for i in range(len(y))]    
    return grad_y, grad_z

def grad_partisan(myVars):
    D20, D16, T20, T16 = myVars
    k = len(D20)
    x = {i: D16[i]/T16[i] + D20[i]/T20[i] for i in range(k)}
    pvi = {i: (50*x[i] - 51.69) for i in range(k)}
    d = {i: pdf_fun(pvi[i]/4.8)*(50/4.8) for i in range(k)}
    
    grad_D20 = [d[i]/T20[i] for i in range(k)]
    grad_D16 = [d[i]/T16[i] for i in range(k)]
    
    grad_T20 = [-d[i]*D20[i]/T20[i]**2 for i in range(k)]
    grad_T16 = [-d[i]*D16[i]/T16[i]**2 for i in range(k)]
    
    return [grad_D20, grad_D16, grad_T20, grad_T16]
    #return {'D20':grad_D20, 'D16':grad_D16, 'T20':grad_T20, 'T16':grad_T16}

# Python3 program to generate n-bit Gray codes
import math as mt

def generateGrayarr(n):
    '''
     This function generates all n bit Gray codes and prints the generated codes
    '''
    # base case
    if (n <= 0):
        return

    # 'arr' will store all generated codes
    arr = list()

    # start with one-bit pattern
    arr.append([0])
    arr.append([1])

    # Every iteration of this loop generates
    # 2*i codes from previously generated i codes.
    i = 2
    j = 0
    while(True):

        if i >= 1 << n:
            break

        # Enter the previously generated codes
        # again in arr[] in reverse order.
        # Nor arr[] has double number of codes.
        for j in range(i - 1, -1, -1):
            arr.append(arr[j])

        # append 0 to the first half
        for j in range(i):
            arr[j] = [0] + arr[j]

        # append 1 to the second half
        for j in range(i, 2 * i):
            arr[j] = [1] + arr[j]
        i = i << 1

    # print contents of arr[]
    return arr
    
def summation_sets(code, nu,L):
    S = {}
    for j in range(nu):
        S[j] = {0:[], 1:[]}
        if code[j][0] == 0:
            S[j][0].append(0)
        if code[j][0] == 1:
            S[j][0].append(1)
        for i in range(min(2**nu-1,L)):
            if code[i][j] == 0 and code[i+1][j] == 0:
                S[j][0].append(i+1)
            if code[i][j] == 1 and code[i+1][j] == 1:
                S[j][1].append(i+1)
    return S

def support(code, val):
    return [i for i, v in enumerate(code) if v == val]

import scipy.optimize as spo
import scipy.integrate as integrate
import numpy as np



def calculate_breakpoints_exp(inf, sup, R, cdf_fun, rho, grad):
    ''' Calcualtes breakpoints that minimizes the expected error given that we want R breakpoints '''
        
    r_initial = [inf+i*(sup-inf)/R  for i in range(R+1)]
    cons = ({ 'type':'eq', 'fun': lambda r: r[-1] - r_initial[-1] }, { 'type':'eq', 'fun': lambda r: r[0] - r_initial[0] })
    result = spo.minimize(rho, r_initial, constraints=cons, jac=grad, tol=.00000001)
    integral = integrate.quad(cdf_fun, r_initial[0], r_initial[-1])[0]
    if result.success:
        print('helper_methods.py: Breakpoints successfully generated!')
        print(f'helper_methods.py: Breakpoints = {result.x}')
        expErr = (result.fun-integral)/(r_initial[-1]-r_initial[0])
        print(f'helper_methods.py: ExpectError = {expErr}')
    else:
        print('helper_methods.py: Breakpoints failed to generate!')
    RATIO_BREAKPOINTS = result.x
    PWL_PARTS = len(RATIO_BREAKPOINTS)-1
    CDF_VALUES = [cdf_fun(r) for r in RATIO_BREAKPOINTS]
    maxErr = max([CDF_VALUES[i+1]-CDF_VALUES[i] for i in range(len(CDF_VALUES)-1)])
    return RATIO_BREAKPOINTS, PWL_PARTS, CDF_VALUES, maxErr

def calculate_breakpoints_max(inf, sup, R, cdf_fun):
    ''' Calcualtes breakpoints that minimizes the maximum error given that we want R breakpoints '''
    vstep = (cdf_fun(sup)-cdf_fun(inf))/R
    CDF_VALUES = [cdf_fun(inf)+vstep*i for i in range(R+1)]
    maxErr = max([CDF_VALUES[i+1]-CDF_VALUES[i] for i in range(len(CDF_VALUES)-1)])
    RATIO_BREAKPOINTS = []
    for v in CDF_VALUES:
        def func(x):
            return cdf_fun(x)-v
        RATIO_BREAKPOINTS += [spo.fsolve(func, 0.5)[0]]
    #integral = integrate.quad(cdf_fun, inf, sup)[0]
    #expErr = (sum(CDF_VALUES[i+1]*(RATIO_BREAKPOINTS[i+1]-RATIO_BREAKPOINTS[i]) for i in range(len(CDF_VALUES)-1))-integral)/(sup-inf)
    PWL_PARTS = len(RATIO_BREAKPOINTS)-1
    return RATIO_BREAKPOINTS, PWL_PARTS, CDF_VALUES, maxErr 

def calculate_breakpoints_max_error(inf, sup, vstep, cdf_fun):
    ''' Calculates breakpoints given a max error tolerance vstep, instead of a number of breakpoints '''
    R = round((cdf_fun(sup)-cdf_fun(inf))/vstep) + 1
    CDF_VALUES = [cdf_fun(inf)+vstep*i for i in range(R+1)]
    maxErr = max([CDF_VALUES[i+1]-CDF_VALUES[i] for i in range(len(CDF_VALUES)-1)])
    RATIO_BREAKPOINTS = []
    for v in CDF_VALUES:
        def func(x):
            return cdf_fun(x)-v
        RATIO_BREAKPOINTS += [spo.fsolve(func, 0.5)[0]]
    #integral = integrate.quad(cdf_fun, inf, sup)[0]
    #expErr = (sum(CDF_VALUES[i+1]*(RATIO_BREAKPOINTS[i+1]-RATIO_BREAKPOINTS[i]) for i in range(len(CDF_VALUES)-1))-integral)/(sup-inf)
    PWL_PARTS = len(RATIO_BREAKPOINTS)-1
    return RATIO_BREAKPOINTS, PWL_PARTS, CDF_VALUES, maxErr 


def calculate_breakpoints_alt(inf, sup,R,cdf_fun, rho, grad):
    r_initial = [inf+i*(sup-inf)/R for i in range(R+1)]
    cons = ({'type': 'eq', 'fun': lambda r: r[-1] - r_initial[-1]},
            {'type': 'eq', 'fun': lambda r: r[0] - r_initial[0]})
    result = spo.minimize(rho, r_initial, constraints=cons, jac=grad, tol=0.00000001)
    integral = integrate.quad(cdf_fun, r_initial[0], r_initial[-1])[0]
    if result.success:
        print('helper_methods.py: Breakpoints successfully generated!')
        print(f'helper_methods.py: Breakpoints = {result.x}')
        expErr = (result.fun-integral)/(r_initial[-1]-r_initial[0])
        print(f'helper_methods.py: ExpectError = {expErr}')
    else:
        print('helper_methods.py: Breakpoints failed to generate!')
    RATIO_BREAKPOINTS = result.x
    PWL_PARTS = len(RATIO_BREAKPOINTS)-1
    CDF_VALUES = [cdf_fun(r) for r in RATIO_BREAKPOINTS]
    CDFMAX = CDF_VALUES[-1]
    CDF_DELTAS = [CDF_VALUES[i+1]-CDF_VALUES[i] for i in range(len(CDF_VALUES)-1)]
    maxErr = max([CDF_VALUES[i+1]-CDF_VALUES[i] for i in range(len(CDF_VALUES)-1)])
    return RATIO_BREAKPOINTS, PWL_PARTS, CDF_VALUES, CDFMAX, CDF_DELTAS, maxErr

def liang_barsky(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    ''' Calculates the ray through (x0,y0) (x1,y1) and find the intersections with the box described by the bounds. '''
    t0, t1 = 0, 1
    dx, dy = x1 - x0, y1 - y0

    for edge in range(4):   # Traverse through left, right, bottom, top edges.
        if edge == 0:
            p, q = -dx, -(xmin - x0)
        elif edge == 1:
            p, q = dx, (xmax - x0)
        elif edge == 2:
            p, q = -dy, -(ymin - y0)
        else:
            p, q = dy, (ymax - y0)

        r = q / p

        if p == 0 and q < 0:
            #print("helper_methods.py: No intersection found")
            return (x0, y0), (x1, y1)   # Don't draw line at all. (parallel line outside)

        if p < 0:
            if r > t1:
                #print("helper_methods.py: No intersection found")
                return (xmax, ymin), (xmax, ymin)
                return (x0, y0), (x1, y1)   # Don't draw line at all.
            elif r > t0:
                t0 = r   # Line is clipped!
        elif p > 0:
            if r < t0:
                #print("helper_methods.py: No intersection found")
                return (xmin, ymax), (xmin, ymax)
                return (x0, y0), (x1, y1)   # Don't draw line at all.
            elif r < t1:
                t1 = r   # Line is clipped!

    return [x0 + t0 * dx, y0 + t0 * dy],[x0 + t1 * dx, y0 + t1 * dy]

import numpy as np

def rescale_gradients(v1, v2) -> tuple:
    """
    Rescales two numpy arrays of gradients so that they have the same order of magnitude.
    This is useful to avoid numerical instability in some optimization algorithms.

    Args:
        v1 : The first array of gradients.
        v2 : The second array of gradients.

    Returns:
        tuple: A tuple with the rescaled versions of `v1` and `v2`, respectively.

    Example:
        v1 = np.array([1e-7, 2e-6, 3e-5])
        v2 = np.array([-1e-7, -2e-6, -3e-5])
        scaled_v1, scaled_v2 = rescale_gradients(v1, v2)
        assert np.allclose(scaled_v1, np.array([1, 20, 300]))
        assert np.allclose(scaled_v2, np.array([-1, -20, -300]))
    """
    # Compute the median value of the concatenation of both arrays
    median = np.median(np.concatenate((np.abs(v1), np.abs(v2))))

    # Compute the order of magnitude of the median
    power = 10 ** (-np.floor(np.log10(median)))

    # Scale both gradients by the same power of 10
    return np.array(v1) * power, np.array(v2) * power

import numpy as np

def rescale_gradients_general(list_of_lists) -> list:

    """
    Rescales a list of lists of gradients so that they have the same order of magnitude.
    This is useful to avoid numerical instability in some optimization algorithms.
    Values in absolute value less than 10^-12 are rounded to 0 and ignored in the median computation.

    Args:
        list_of_lists : A list of lists of gradients.

    Returns:
        list: A list with the rescaled versions of the input lists.

    Example:
        list_of_lists = [[1e-7, 2e-6, 3e-5], [4e-7, 5e-6, 6e-5]]
        scaled_list_of_lists = rescale_gradients(list_of_lists)
        assert np.allclose(scaled_list_of_lists[0], np.array([1, 20, 300]))
        assert np.allclose(scaled_list_of_lists[1], np.array([4, 50, 600]))
    """
    # Convert all lists to numpy arrays
    array_list = [np.array(l) for l in list_of_lists]

    # Flatten all arrays into a single array
    flat_array = np.concatenate(array_list)

    # Filter out values less than 10^-12 in absolute value for median calculation
    filtered_array = flat_array[np.abs(flat_array) >= 1e-12]

    # Compute the median value of the filtered array
    median = np.median(filtered_array)

    # Compute the order of magnitude of the median
    power = 10 ** (-np.floor(np.log10(median)))

    # Scale all arrays by the same power of 10
    scaled_arrays = [arr * power for arr in array_list]

    # Round values less than 10^-12 in absolute value to 0 in the scaled arrays
    return [np.where(np.abs(arr) < 1e-12, 0, arr) for arr in scaled_arrays]



# def rescale_gradients_general(list_of_lists) -> list:
#     """
#     Rescales a list of lists of gradients so that they have the same order of magnitude.
#     This is useful to avoid numerical instability in some optimization algorithms.

#     Args:
#         list_of_lists : A list of lists of gradients.

#     Returns:
#         list: A list with the rescaled versions of the input lists.

#     Example:
#         list_of_lists = [[1e-7, 2e-6, 3e-5], [4e-7, 5e-6, 6e-5]]
#         scaled_list_of_lists = rescale_gradients(list_of_lists)
#         assert np.allclose(scaled_list_of_lists[0], np.array([1, 20, 300]))
#         assert np.allclose(scaled_list_of_lists[1], np.array([4, 50, 600]))
#     """
#     # Convert all lists to numpy arrays and flatten into a single array
#     array_list = [np.array(l) for l in list_of_lists]
#     flat_array = np.concatenate(array_list)

#     # Compute the median value of the flattened array
#     median = np.median(np.abs(flat_array))

#     # Compute the order of magnitude of the median
#     power = 10 ** (-np.floor(np.log10(median)))

#     # Scale all arrays by the same power of 10
#     return [arr * power for arr in array_list]


def convert_keys_to_int(obj):
                if isinstance(obj, dict):
                    new_obj = {}
                    for key, value in obj.items():
                        if isinstance(value, dict):
                            new_obj[key] = convert_keys_to_int(value)
                        elif isinstance(value, list):
                            new_obj[key] = [convert_keys_to_int(item) for item in value]
                        else:
                            try:
                                new_obj[int(key)] = value
                            except ValueError:
                                new_obj[key] = value
                    return new_obj
                elif isinstance(obj, list):
                    return [convert_keys_to_int(item) for item in obj]
                else:
                    return obj
