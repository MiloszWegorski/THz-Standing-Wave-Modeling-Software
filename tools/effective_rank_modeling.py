from tools.dependencies import *
from Signal_source.Measurement_schemes import RandomizedUniformMeasurement
import sys
from tools.effective_rank import *

import numpy as np
from time import sleep
import copy

from tools.dependencies import printmodel

from tqdm import tqdm

bool2=False

def build_scheme(*, components, freqs, dist_limit, point_limit, rank_target = 0.5):


    #get best distance for plot
    delta_rank = 1

    wavelength = get_wavelength(freqs[0])
    scheme_range = wavelength

    best_dist = scheme_range

    rank_new = 0

    while rank_new < len(components)-rank_target and scheme_range < dist_limit:

        measure_scheme_prev = RandomizedUniformMeasurement(start_position=-scheme_range, 
                                                    end_position=scheme_range,
                                                    num_points=1000,
                                                    randomization=0)

        rank_prev = get_lowest_rank_by_freq(components=components, freqs=freqs,
                                        scheme=measure_scheme_prev, normalize=True)

        new_range = scheme_range + 0.1
        
        measure_scheme_new = RandomizedUniformMeasurement(start_position=-new_range, 
                                                            end_position=new_range,
                                                            num_points=1000,
                                                            randomization=0)
        
        rank_new = get_lowest_rank_by_freq(components=components, freqs=freqs,
                                                scheme=measure_scheme_new, normalize=True)

        delta_rank = rank_new - rank_prev


        print(f'{scheme_range=}  ', end='\r')

        scheme_range = new_range
        best_dist = scheme_range


    best_dist = np.round(best_dist, 2)
    #starting number of components is the number + 1
    num_points = len(components) + 1

    measure_scheme = RandomizedUniformMeasurement(start_position=-best_dist, 
                                                  end_position=best_dist,
                                                  num_points=num_points,
                                                  randomization=0)

    erank = get_lowest_rank_by_freq(components=components, freqs=freqs,
                                    scheme=measure_scheme, normalize=True)

    itter = 0

    while erank < len(components)-rank_target and num_points < point_limit:

        num_points_increased = num_points + 1


        # Schemes to be analyzed
        measure_scheme_more_points = RandomizedUniformMeasurement(start_position=-best_dist, 
                                                      end_position=best_dist,
                                                      num_points=num_points_increased,
                                                      randomization=0)

        # eff ranks of the schemes
        erank_num_points_increase = get_lowest_rank_by_freq(components=components, freqs=freqs,
                                        scheme=measure_scheme_more_points, normalize=True)
        
        delta_points = erank_num_points_increase - erank

        erank = erank_num_points_increase
        num_points = num_points_increased
        measure_scheme = measure_scheme_more_points

        print(f'{erank=} || {num_points=} || {best_dist=} || {itter=} || {delta_points=}      ', end='\r')
        
        itter += 1


    return measure_scheme
        

def build_model_erank(*, components, base_components, freq, scheme, N_limit, M_limit, normalize= True):

    max_model = copy.copy(components)

    for i in range(N_limit):
        max_model = add_N_component(comps=max_model)

    for i in range(M_limit):
        max_model = add_M_component(comps=max_model, M_num=i)


    components = max_model

    U, S, Vh = compute_SVD(freq=freq, measurement_scheme=scheme, components=max_model,
                           normalize=normalize)

    erank = get_effective_rank(S=S)
    print(f'{erank=}')

    #truncate N worst components

    #erank here is used as a max number of components that are supported by the 
    #scheme
    num_comps_to_truncate = int(len(max_model) - np.ceil(erank))


    U, S, Vh = compute_SVD(freq=freq, measurement_scheme=scheme, components=max_model,
                        normalize=normalize)

    S_threshhold = len(S) - num_comps_to_truncate

    rows = np.empty(shape=num_comps_to_truncate*len(max_model))
    maximum_indexes = np.empty(shape=num_comps_to_truncate*len(max_model), dtype=int)

    indexes = np.linspace(0, len(max_model)-1, len(max_model), dtype=int)


    counter = 0

    for i in Vh[S_threshhold:]:
        for j, max_index in zip(i, indexes):
            rows[counter] = j
            maximum_indexes[counter] = max_index
            counter +=1

    protected_indexes = np.empty(len(base_components), dtype=int)

    row_temp = rows

    for i, comp in enumerate(max_model):
        for j, base_comp in enumerate(base_components):
            if comp == base_comp:
                protected_indexes[j] = i

    
    #array of indexes to be sorted similtaneously with the row

    # bubble sort for array and the array of the corresponding indexes
    for i in range(len(row_temp)):
        for j in range(len(row_temp)-1 - i):
            if row_temp[j] > row_temp[j+1]:
                temp = row_temp[j]
                row_temp[j] = row_temp[j+1]
                row_temp[j+1] = temp

                temp = maximum_indexes[j]
                maximum_indexes[j] = maximum_indexes[j+1]
                maximum_indexes[j+1] = temp

    counter = 0

    print(f'{num_comps_to_truncate=}')

    deleted_comp_indexes = []

    for i, index_max_component in enumerate(maximum_indexes):
        if index_max_component not in protected_indexes:
            if counter == num_comps_to_truncate:
                break
            print(f'{index_max_component=}')
            if index_max_component not in deleted_comp_indexes:
                deleted_comp_indexes.append(index_max_component)
                max_model.pop(index_max_component)
                counter +=1

    U, S, Vh = compute_SVD(freq=freq, measurement_scheme=scheme, components=max_model,
                           normalize=normalize)

    erank = get_effective_rank(S=S)
    print(f'final rank = {erank}')
    print(len(max_model))
    print('\n')
    printmodel(model=max_model)
    print('\n')

    return max_model

        
        


def  build_model(*, components, freq, scheme, limit, N_limit, M_limit, threshold, normalize= True):

    base_comps = components

    if type(freq) in (list, np.ndarray):

        total = (N_limit + M_limit) * len(freq)
        pbar = tqdm(total=total)

        for i in range(N_limit):
            #adding N components
            new_comps = add_N_component(comps=components)

            for f in freq:
                U, S, Vh = compute_SVD(freq=f, measurement_scheme=scheme, components=new_comps, normalize=True)

                pbar.update(1)

                match threshold:
                    case 'flat':
                        mask = get_s_filter_threshold(S, limit)
                    case 'sum':
                        mask = get_s_filter_sum(S, limit)
                    case 'erank':
                        mask = get_s_filter_eff_rank(S=S)

                if not all(mask):
                    components, _, _ = truncate_components(S=S, Vh=Vh, U=U,
                                                           components=new_comps, 
                                                           base_components=base_comps,
                                                           freq=f, measurement_scheme=scheme,
                                                           limit=limit, threshold=threshold)
                else:
                    components = new_comps
            #adding M components
        if M_limit != 0:
            for i in range(M_limit):
    
                new_comps = add_M_component(comps=components, M_num=(i+1))

                for f in freq:        
                    U, S, Vh = compute_SVD(freq=f, measurement_scheme=scheme, components=new_comps, normalize=True)
                    pbar.update(1)


                    match threshold:
                        case 'flat':
                            mask = get_s_filter_threshold(S, limit)
                        case 'sum':
                            mask = get_s_filter_sum(S, limit)
                        case 'erank':
                            mask = get_s_filter_eff_rank(S=S)

                    if not all(mask):
                        components, _, _ = truncate_components(S=S, Vh=Vh, U=U,
                                                            components=new_comps, 
                                                            base_components=base_comps,
                                                            freq=f, measurement_scheme=scheme,
                                                            limit=limit, threshold=threshold)
                    else:
                        components = new_comps
    else:
        #adding N components
        total = N_limit + M_limit
        pbar = tqdm(total=total)
        for i in range(N_limit):
            new_comps = add_N_component(comps=components)

            U, S, Vh = compute_SVD(freq=freq, measurement_scheme=scheme, components=new_comps, normalize=True)

            pbar.update(1)

            match threshold:
                case 'flat':
                    mask = get_s_filter_threshold(S, limit)
                case 'sum':
                    mask = get_s_filter_sum(S, limit)
                case 'erank':
                    mask = get_s_filter_eff_rank(S=S)

            if not all(mask):
                components, _, _ = truncate_components(S=S, Vh=Vh, U=U,
                                                    components=new_comps, 
                                                    base_components=base_comps,
                                                    freq=f, measurement_scheme=scheme,
                                                    limit=limit, threshold=threshold)
            else:
                components = new_comps


        #adding M components
        if M_limit != 0:
            for i in range(M_limit):
                
                new_comps = add_M_component(comps=components, M_num=(i+1))

                U, S, Vh = compute_SVD(freq=freq, measurement_scheme=scheme, components=new_comps, normalize=True)

                pbar.update(1)

                match threshold:
                    case 'flat':
                        mask = get_s_filter_threshold(S, limit)
                    case 'sum':
                        mask = get_s_filter_sum(S, limit)
                    case 'erank':
                        mask = get_s_filter_eff_rank(S=S)

                if not all(mask):
                    components, _, _ = truncate_components(S=S, Vh=Vh, U=U,
                                                        components=new_comps, 
                                                        base_components=base_comps,
                                                        freq=f, measurement_scheme=scheme,
                                                        limit=limit, threshold=threshold)
                else:
                    components = new_comps

    return components

def truncate_measurement(*, comps, freq, scheme, limit, threshold, base_components):

    #check if frequencies is a list or a float
    if type(freq) in (list, np.ndarray):
        for i, f in enumerate(freq):
            U, S, Vh = compute_SVD(freq=f, measurement_scheme=scheme, 
                                               components=comps, normalize=True)
            
            comps, (U, S, Vh), bool1 = truncate_components(S=S, Vh=Vh, 
                                                components=comps,
                                                freq=f,
                                                measurement_scheme=scheme,
                                                limit=limit,
                                                U=U, threshold=threshold, 
                                                base_components=base_components)
            
    else:
        U, S, Vh = compute_SVD(freq=freq, 
                                           measurement_scheme=scheme, 
                                           components=comps, normalize=True)

        comps, (U, S, Vh), bool1 = truncate_components(S=S,
                                            Vh=Vh,
                                            components=comps,
                                            freq=freq,
                                            measurement_scheme=scheme,
                                            limit=limit,
                                            U=U, threshold=threshold,
                                            base_components=base_components)

    if not bool1 and not bool2:
        return comps, scheme
    else:
        return truncate_measurement(comps=comps, freq=freq, scheme=scheme, limit=limit, threshold=threshold, base_components=base_components)

def truncate_components(*, U, S, Vh, components, base_components, freq, measurement_scheme, limit, improved_bool = False, threshold):
    
    if threshold == 'flat':
        mask = get_s_filter_threshold(S, limit)
    elif threshold == 'sum':
        mask = get_s_filter_sum(S, limit)
    elif threshold == 'erank':
        mask = get_s_filter_eff_rank(S=S)


    masked_components=[]

    if not all(mask):
        improved_bool = True

        indexes_rows_Vh = []

        for i, (comp, mask_component) in enumerate(zip(components, mask)):
            
            # make function to search through the Vh decomposition
            # in the row where mask component is true
            if not mask_component:
                indexes_rows_Vh.append(i)
                
        remove_comps = get_problematic_Vh(Vh=Vh, row_nums=indexes_rows_Vh,comps=components,
                                            base_comps=base_components)

        for i, comp in enumerate(components):
            if not i in remove_comps:
                masked_components.append(comp)


        U, S, Vh = compute_SVD(freq=freq, 
                                        measurement_scheme=measurement_scheme,
                                        components=masked_components, normalize=True)

        return truncate_components(S=S, Vh=Vh, components=masked_components, freq=freq, base_components=base_components,
                measurement_scheme=measurement_scheme, limit=limit, U = U, improved_bool=improved_bool, threshold=threshold)
    else:
        return components, (U, S, Vh), improved_bool

def get_problematic_Vh(*, Vh, row_nums, comps, base_comps):
     
    for row_num in row_nums:
        # get absolute balue of row
        row = np.abs(Vh[row_num, :])


        #the idea of the below code is to sort the array by the value of each cell
        # in the order of biggest to smallest so that later the component to be removed
        # can be picked as the one which is the largest contributor however is not 
        # included in the base components which are to be kept

        #temporary value needed for bubble sort
        # bubble sort used here since two arrays are being sorted similtaneously
        # and efficiency is not incredibly important as array is small
        row_temp = row

        #array of indexes to be sorted similtaneously with the row
        maximum_indexes = np.linspace(0, len(row)-1, len(row), dtype=int)

        # bubble sort for array and the array of the corresponding indexes
        for i in range(len(row_temp)):
            for j in range(len(row_temp)-1 - i):
                if row_temp[j] > row_temp[j+1]:
                    temp = row_temp[j]
                    row_temp[j] = row_temp[j+1]
                    row_temp[j+1] = temp

                    temp = maximum_indexes[j]
                    maximum_indexes[j] = maximum_indexes[j+1]
                    maximum_indexes[j+1] = temp



        #get indexes which are to be protected
        protected_indexes = np.empty(len(base_comps), dtype=int)

        for i, comp in enumerate(comps):
            for j, base_comp in enumerate(base_comps):
                if comp == base_comp:
                    protected_indexes[j] = i


        # remove n components of largest contribution

        #here the plan of action is to set n = erank and then sort all the rows
        #together and remove the n largest non protected components from the 
        #modes that are to be discarded
        n = 1

        #another idea is to also have some form of bias for components which are
        # only largely contributing to these problematic modes and avoid ones which
        # contribute to other non problematic modes 

        problematic_components = []#np.zeros(n, dtype=int)# array to store problematic component

        counter = 0

        for i, index_comp in enumerate(maximum_indexes):
            if index_comp not in protected_indexes:
                #problematic_components[i] = index_comp
                problematic_components.append(index_comp)

                
                
                counter += 1
                if counter == n:
                    break
    
    return problematic_components

def randomize_scheme(scheme, U_decomposition, freq, comps, rand_0 = 0):
    
    
    start_point = scheme.start_position
    end_point = scheme.end_position
    num_point = scheme.num_points

    randomizarion = rand_0
    

    S_sum_new = 0
    S_sum_old = 1
    
    while (S_sum_new < S_sum_old) and (randomizarion < 0.1):


        new_scheme = RandomizedUniformMeasurement(start_position=start_point, 
                                                end_position=end_point, 
                                                num_points=num_point,
                                                randomization=randomizarion)

        U, S, Vh = compute_SVD(freq=freq, measurement_scheme=new_scheme, components=comps, normalize=True)

        S_sum_new = np.sum(U)
        S_sum_new = np.sum(U_decomposition)
        if (S_sum_new < S_sum_old) and (randomizarion == rand_0):
            return  RandomizedUniformMeasurement(start_position=start_point, 
                                                end_position=end_point, 
                                                num_points=num_point,
                                                randomization=rand_0), (U, S, Vh),  False

        randomizarion += 0.01

    return new_scheme, (U, S, Vh), True