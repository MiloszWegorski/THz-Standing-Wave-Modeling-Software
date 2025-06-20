import numpy as np
import re
import pandas as pd
import zipfile

from analysis_tools.dependencies import complex_to_mag_and_phase, Mag_and_phase_to_complex

def format_number_with_pn_for_sign(x, num_decimals = 3):
    
    sign_character = 'p' if x >= 0.0 else 'n'
    absolute_value = abs(x)
    absolute_integer_part = int(absolute_value)
    fractional_integer = int((absolute_value - absolute_integer_part) * 10**num_decimals)

    # fractional part is formatted as an integer of fixed length
    # padded with zero at the front to the correct length
    formatted_value = f'{absolute_integer_part}{sign_character}{fractional_integer:0{num_decimals}d}'
    return formatted_value

def deformat_number_with_pn_for_sign(number):

    for char in number:
        if char == 'n':
            number = number.replace('n', '.')

            return -float(number)
        

        elif char == 'p':
            number = number.replace('p', '.')

            return float(number)


class Save_to_file():

    def __init__(self, folder, test_name, x_axis_pos, y_axis_pos = None, z_axis_pos = None):

        self.folder = folder
        self.test_name = test_name

        pos_columns = []
        for pos in [x_axis_pos, y_axis_pos, z_axis_pos]:
            if pos is not None:
                pos_columns.append(pos)
        
        self.positions  = np.column_stack(pos_columns)



    def write_data_to_file(self, data_sets):
        
        #add XML file creation
        
        # try not to delete old files, just in case

        with zipfile.ZipFile(self.folder, 'x') as data_file:

            for seq_num, (data, positions) in enumerate(zip(data_sets, self.positions)):
                
                column_names = ['#f[GHz]']
                parameter_names = list(data[0]['measurements'].keys())

                for parameter_name in parameter_names:
                    column_names.append(f'{parameter_name}Mag[dB]')
                    column_names.append(f'{parameter_name}Phase[deg]')                    

                name = f'{self.test_name}_{seq_num:04d}_'
                name += "_".join([format_number_with_pn_for_sign(x) for x in positions])
                
                num_frequencies = len(data)
                output_data = np.empty((num_frequencies, len(column_names)))
                for freq_idx, measurement_point in enumerate(data):
                    # frequency in GHz (data uses Hz)
                    output_data[freq_idx, 0] = measurement_point['frequency'] / 1e9
                    column_index = 1
                    measurements = measurement_point['measurements']
                    for parameter_name in parameter_names:

                        magnitude, phase = complex_to_mag_and_phase(measurements[parameter_name])

                        output_data[freq_idx, column_index] = magnitude
                        output_data[freq_idx, column_index + 1] = phase
                        column_index += 2

                dataframe = pd.DataFrame(output_data, columns = column_names)
                with data_file.open(name + '.dat', mode='w') as f:
                    dataframe.to_csv(f, sep='\t', index=False)
    

class Read_File():

    def __init__(self, folder_name, index_measurements = False,):
        
        self.freqs = None
        self.dist = None

        self.folder_name = folder_name

        self.index_measurements = index_measurements

        self.S11 = None
        self.S12 = None
        self.S21 = None
        self.S22 = None


        self.load_data(folder_name)

    def load_data(self, folder_name):

        frequencies = []
        component_measurements = []
        comp_order = []

        with zipfile.ZipFile(folder_name) as zip_file:
            

            file_names = zip_file.namelist()

            for i, names in enumerate(file_names):
                file_names[i] = names.replace(f'{folder_name.replace('zip', '')}/', '')


            self.dist = np.empty(len(zip_file.namelist()), dtype=list)

            for i in range(len(zip_file.namelist())):
                self.dist[i] = []


            for i, name in enumerate(np.sort(zip_file.namelist())):
                
                if '.xml' in name:
                    continue

                #remove all instances of folder name in file name since sometimes
                #this happens up to twice (im not sure why it happens)
                name_components = re.findall(r"(.*)_(\d{4})((_\d+[np]\d+)+)", name)[0]


                #get index of measurement number
                index_of_measurement_num = int(name_components[1])
                
                axis_positions = [deformat_number_with_pn_for_sign(i) for i in name_components[2].split('_')[1:]]

                
                #axis num refers to x, y z axis as 1, 2, 3 respectively
                for pos in axis_positions:
                    self.dist[i].append(pos)

                if self.index_measurements:
                    self.dist[i].append(index_of_measurement_num)
                
                with zip_file.open(name) as data_file:
                    measurement_data = pd.read_csv(data_file, sep='\t')
                    
                    keys = measurement_data.keys()
                    
                    # check if the array to store results is big enough 
                    # take one away from keys for frequency and magnitude 
                    # and phase are merged so we divide by 2
                    while len(component_measurements) < (len(keys)-1)/2:
                        component_measurements.append([])

                    # appending the frequencies
                    frequencies = measurement_data[keys[0]]

                    for j, comp in enumerate(range(1, len(keys)-1, 2)):
                        comp_name = keys[comp].replace('Mag[dB]', '')
                        if i == 0:
                            comp_order.append(comp_name)

                        complex_comp = Mag_and_phase_to_complex(measurement_data[keys[comp]],
                                                                measurement_data[keys[comp+1]])
                        
                        component_measurements[j].append(complex_comp)

        self.freqs = np.array(frequencies)

        for name, values in zip(comp_order, component_measurements):

            match name:
                case 'S11':
                    self.S11 = np.rot90(np.array(values))
                case 'S21':
                    self.S21 = np.rot90(np.array(values))
                case 'S22':
                    self.S22 = np.rot90(np.array(values))
                case 'S12':
                    self.S12 = np.rot90(np.array(values))
                case _:
                    print('couldnt match case')


    def get_data(self, component, frequencies = None, distances = None):
        
        return_scalar_freq = False
        return_scalar_dist = False

        if not np.iterable(frequencies) and type(frequencies) != type(None):
            frequencies = [frequencies]
            return_scalar_freq = True

        if not np.iterable(distances) and type(distances) != type(None):
            distances = [distances]
            return_scalar_dist = True

        match component:
            case 'S11':
                if type(self.S11) == type(None):
                    raise ValueError('Component not measured')
                else:
                    comp = self.S11
            case 'S21':
                if type(self.S21) == type(None):
                    raise ValueError('Component not measured')
                else:
                    comp = self.S21
            case 'S22':

                if type(self.S22) == type(None):
                    raise ValueError('Component not measured')
                comp = self.S22
            case 'S12':
                if type(self.S12) == type(None):
                    raise ValueError('Component not measured')
                else:
                    comp = self.S12


        if type(frequencies) != type(None):
            
            #freq_indexes = [np.where(self.freqs == i,) for i in frequencies]
            
            freq_indexes = np.empty(len(frequencies), dtype=int)

            for i, freq in enumerate(self.freqs):
                for j, freq_search in enumerate(frequencies):
                    if freq == freq_search:
                        freq_indexes[j] = i
                        #stop the loop when the first index is found
                        break

            
            return_array = np.empty(len(freq_indexes))
            
            return_array = np.array(comp)[freq_indexes]

        else:
            return_array = comp            

        if type(distances) != type(None):

            distance_indexes = np.empty(len(distances), dtype=int)

            for i, distance in enumerate(self.dist):
                for j, dist_search in enumerate(distances):
                    if distance == dist_search:
                        distance_indexes[j] = i
                        #stop this loop when the first index is found
                        break

            return_array = return_array[:, distance_indexes]

        if return_scalar_freq:
            return_array = return_array[0]
            if return_scalar_dist:
                return_array = return_array[0]

        elif return_scalar_dist:
            return_array = return_array[:, 0]

        return return_array
    
    def get_frequencies(self):
        return np.array(self.freqs)
    
    def get_distances(self):
        return np.array(self.dist)


