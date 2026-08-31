import numpy as np
import re
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET
from tools.dependencies import complex_to_mag_and_phase, Mag_and_phase_to_complex

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

def get_s_parameters(parsed_xml):
    
    s_params = {}

    for i in ['S11', 'S22', 'S21', 'S12', 'Magn.', 'Imag', 'Phase', 'Real']:      
        s_params.update({i : bool(parsed_xml['Cluster']['Cluster'][0]['Cluster'][i])})

    return s_params

def get_all_axis_point_nums(parsed_xml):
    
    scan_lengths = []

    for i, data in enumerate(parsed_xml['Cluster']['Cluster'][1]["Array"]):
        if i > 0: 
            if type(data['Cluster']) in (np.ndarray, list):
                for z in data['Cluster']:
                    scan_lengths.append(z["Points"])
            else: 
                scan_lengths.append(data['Cluster']['Points'])

    return np.asanyarray(scan_lengths)

def clean_tag(tag):
    # split the tag removing the link inside it
    return tag.split('}')[-1]


def convert_value(text):
    #remove tabulations or spaces
    text = (text or "").strip()

    #try convert to int then float and if both fail string
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text

def parse_element(element):
    #get tag of branch
    tag = clean_tag(element.tag)
    #get all children of branch
    children = list(element)

    name = None
    value = None

    
    for child in children:
        child_tag = clean_tag(child.tag)
        if child_tag == "Name":
            name = (child.text or "").strip()
        elif child_tag == "Val":
            value = convert_value(child.text)

    if name is not None and value is not None:
        return name, value

    # If the tree branch has children aka. is not the end node  we need to go
    # all the way till a value is found for all children
    if children:
        result = {}

        for child in children:
            # pass child to be parsed this will dig all the way until it 
            # finds a value
            parsed = parse_element(child)

            # in case value returned is None
            if parsed is None:
                continue

            key, val = parsed

            # This handles elements with more than two items aka one which is
            # not just a Name and value
            if key in result:
                if isinstance(result[key], list):
                    result[key].append(val)
                else:
                    result[key] = [result[key], val]
            else:
                result[key] = val

        return tag, result

    # in case there is a leaf element on a branch
    text = (element.text or "").strip()
    if text:
        return tag, convert_value(text)

    return None

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

    def __init__(self, folder_name, repeat_axis, index_measurements = False):

        self.xml = self.load_xml(folder_name)

        self.freqs = None
        self.dist = None
        self.sqashed_positions = None

        self.folder_name = folder_name

        self.index_measurements = index_measurements
        self.num_dimensions = None

        self.S11 = []
        self.S12 = []
        self.S21 = []
        self.S22 = []


        self.load_data(folder_name, repeat_ax_index=repeat_axis)


    def load_xml(self, folder):

        """Loads in XML file contained within the data file

        Returns:
            _type_: _description_
        """

        with zipfile.ZipFile(folder) as zip:
            
            file_names = zip.namelist()
            
            xml_data = None

            #search for xml and parse it
            for file_name in file_names:

                if '.xml' in file_name:
                    with zip.open(file_name) as xml_file:
                        xml_data = ET.parse(xml_file)

                    root = xml_data.getroot()

                    data = parse_element(root)

                    final_result = data[1] if data else {}
                    break
            
            if final_result == None:
                print('Data file does not contain XML file')
            
        return final_result
    

    def load_data(self, folder_name, repeat_ax_index, index_measurement=False):

        frequencies = []
        component_measurements = []
        comp_order = []

        with zipfile.ZipFile(folder_name) as zip_file:
            

            file_names = zip_file.namelist()

            for i, names in enumerate(file_names):
                file_names[i] = names.replace(f'{folder_name.replace('zip', '')}/', '')


            self.dist = np.empty(len(zip_file.namelist())-2, dtype=list)

            for i, name in enumerate(np.sort(zip_file.namelist())[1:]):
                
                if '.xml' in name:
                    continue

                #remove all instances of folder name in file name since sometimes
                #this happens up to twice (im not sure why it happens)
                name_components = re.findall(r"(.*)_(\d{4})((_\d+[np]\d+)+)", name)[0]

                #get index of measurement number
                index_of_measurement_num = int(name_components[1])
                
                axis_positions = tuple([deformat_number_with_pn_for_sign(i) for i in name_components[-2].split('_')[1:]])

                #get number of dimensions in measurement
                self.num_dimensions = len(axis_positions)

                #axis num refers to x, y z axis as 1, 2, 3 respectively
                self.dist[i] = axis_positions

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




                    for comp in keys:
                        
                        # compile S11 component
                        if 'S11Mag' in comp:
                            comp_order.append('S11')

                            self.S11.append(Mag_and_phase_to_complex(measurement_data['S11Mag[dB]'],
                                                                measurement_data['S11Phase[deg]']))
                            
                        if  'S12Mag' in comp:
                            comp_order.append('S12')

                            self.S12.append(Mag_and_phase_to_complex(measurement_data['S12Mag[dB]'],
                                                                measurement_data['S12Phase[deg]']))
                        
                        if  'S21Mag' in comp:
                            comp_order.append('S21')

                            self.S21.append(Mag_and_phase_to_complex(measurement_data['S21Mag[dB]'],
                                                                measurement_data['S21Phase[deg]']))

                        if  'S22Mag' in comp:
                            comp_order.append('S22')

                            self.S22.append(Mag_and_phase_to_complex(measurement_data['S22Mag[dB]'],
                                                                measurement_data['S22Phase[deg]']))



                        # comp_name = keys[comp].replace('Mag[dB]', '')
                        # if i == 0:
                        #     comp_order.append(comp_name)

                        # complex_comp = Mag_and_phase_to_complex(measurement_data[keys[comp]],
                        #                                         measurement_data[keys[comp+1]])
                        
                        # component_measurements[j].append(complex_comp)

        self.freqs = np.asarray(frequencies)

        #if there is more dimensions than one in measurement clean up the array
        # into a cleaner data structure where
        # for a 2d case:
        # [[set of coordinates], [repeat axis]]
        axis_lengths = get_all_axis_point_nums(self.xml)
        
        shape = np.append(np.delete(axis_lengths, repeat_ax_index), axis_lengths[repeat_ax_index])

        shape_distances = np.append(shape, 2)

        shape_measurement = np.append(shape, len(frequencies))

        self.dist = self.reshape_scan(data=np.stack(self.dist), shape=shape_distances)
        

        self.sqashed_positions = self.dist.mean(axis=len(axis_lengths)-1)


        if 'S11' in comp_order:
            self.S11 = self.reshape_scan(data=self.S11, shape=shape_measurement)
        if 'S12' in comp_order:
            self.S12 = self.reshape_scan(data=self.S12, shape=shape_measurement)
        if 'S21' in comp_order:
            self.S21 = self.reshape_scan(data=self.S21, shape=shape_measurement)
        if 'S22' in comp_order:
            self.S22 = self.reshape_scan(data=self.S22, shape=shape_measurement)


    def get_data(self,*, S_parameter, freqs = None, distances = None):
    
        measurement = None

        if freqs == None:
            if distances == None:
                if S_parameter == 'S11':
                    return np.asarray([self.S11, self.get_distances()])
                elif S_parameter == 'S12':
                    return np.asarray([self.S12, self.get_distances()])
                elif S_parameter == 'S21':
                    return np.asarray([self.S21, self.get_distances()])
                elif S_parameter == 'S22':
                    return np.asarray([self.S22, self.get_distances()])
                else: 
                    raise Exception('Please provide a valid S parameter')

            else:


            #idea is to select the distances to be returned and return 
            # the whole scan for that position
                pass

    def center_data_points(self, *, scheme):

        return scheme - np.mean(scheme)

    def reshape_scan(self, *,data, shape):
        return np.asarray(data).reshape(shape)

    def get_frequencies(self):
        return self.freqs
    
    def get_distances(self):
        return np.squeeze(self.dist)


class Save_simulation():

    def __init__(self,  folder, simulation_name):

        self.folder = folder
        self.simulation_name = simulation_name


    def save_data(self, data_table, columns, Simulation_data):

        with zipfile.ZipFile(self.folder + '.zip', 'x') as folder:

            dataframe = pd.DataFrame(data = data_table, columns=columns)

            with folder.open(f'{self.simulation_name}_data' + '.dat', mode='w') as file:

                dataframe.to_csv(file, sep='\t', index=False)

            with folder.open(f'{self.simulation_name}_parameters' + '.npz', mode='w') as file:
                np.savez(file,  num_simulations=Simulation_data[0],
                                coeff_matrix=Simulation_data[1],
                                measure_scheme=Simulation_data[2].get_points(),
                                amplitudes=Simulation_data[3],
                                amplitude_noise=Simulation_data[4],
                                phase_noise=Simulation_data[5])


class Read_Simulation():

    def __init__(self, folder):

        self.frequencies = None
        self.data = None

        self.coeffs = None
        self.num_simulations = None
        self.measure_scheme = None
        self.simulation_amps = None
        self.amp_noise = None
        self.phase_noise = None

        self.read_data(folder)

    def read_data(self, folder):

        with zipfile.ZipFile(folder, 'r') as folder:
            
            files = folder.namelist()

            data_file = next((f for f in files if '_data' in f), None)
            param_file = next((f for f in files if '_parameters' in f), None)
            
            with folder.open(data_file) as datfile:

                data = pd.read_csv(datfile, sep='\t')

                keys = data.keys()
                
                real_components = []
                angle_components = []


                for i, key in enumerate(keys):
                    if key[0] == 'f':
                        self.frequencies = np.array(data[key])

                    if key[0] == 'A':
                        if 'Real' in key:
                            real_components.append(np.array(data[key]))
                        if 'Angle' in key:
                            angle_components.append(np.array(data[key]))

                complex_data = np.empty((len(real_components), len(real_components[0])), dtype=complex)

                for i, (real, angle) in enumerate(zip(real_components, angle_components)):
                    complex_data[i] = [complex(r, a) for r, a in zip(real, angle)]
            
            self.data = complex_data
            
            with folder.open(param_file) as params:
                parameters = np.load(params, allow_pickle=True)

                self.num_simulations = np.array(parameters['num_simulations'])
                self.coeffs = np.array(parameters['coeff_matrix'])
                self.measure_scheme = np.array(parameters['measure_scheme'])
                self.amp_noise = np.array(parameters['amplitude_noise'])
                self.phase_noise = np.array(parameters['phase_noise'])


    def get_amplitudes(self, frequency):

        indexes = np.where(self.frequencies == frequency)

        amps = np.empty((len(self.data), len(self.data[0][indexes])), dtype = complex)

        for i, amp_list in enumerate(self.data):
            amps[i] = amp_list[indexes]

        return amps
    
    def get_measure_scheme(self):

        return self.measure_scheme
    
        
