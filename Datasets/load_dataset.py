import csv
import SYS_VARS
import matplotlib.pyplot as plt
import numpy as np
import math

attacks_map  = {} #attacks_map = {attack_name: attack_index, ...}
feature_names = []
    
_ATTACK_INDEX_KDD  = -1
_ATTACK_INDEX_NSLKDD = -2

_PROTOCOL_INDEX = 1
_SERVICE_INDEX = 2
_FLAG_INDEX = 3

_attack_row  = 0


def load_variables (var_names):
    csvFileArray = []
    
    with open(var_names, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        
        i = 0
        for row in reader:
            csvFileArray.append(row)
            if (i>0):
                for values in row:
                   feature_names.append(values.split(SYS_VARS.key_val_separator)[0])
            else:
                i += 1

        i = 0
        #Process attacks
        for attack in csvFileArray[_attack_row]:
            attacks_map[attack+'.'] = i
            i += 1

        print ('\n ATTACKS')
        print(attacks_map )
        print ('\n FEATURES')
        print(feature_names)
        
"""Write data into a file_name located in file_directory """
def write_file (data, file_name, file_directory):
    variable_file =  file_directory+SYS_VARS.separator+file_name
    with open(variable_file, 'wb') as f_handle:
        #w = csv.writer(f_handle, delimiter=',')
        np.save(f_handle, data)
        #np.savetxt(name, symbols[index], delimiter=',')

"""Open a a file_name located in file_directory and return its content as numpy darray """
def read_file(file_name, file_directory):
    variable_file =  file_directory+SYS_VARS.separator+file_name
    return np.load(variable_file)


##################### DATA PREPROCESSING #####################
"""Convert the given categorigal variables  to numerical variables, and save then in new_file, living in result_directory
    When we know the mapping of catogry to number
"""
def categorical_known_variable_conversion (dataset, symbols_map, variable_index, new_file, result_directory):
    data= []
    i = []
    save_file = result_directory + SYS_VARS.separator + new_file
    #Open dataset file and modified categorical variable
    """with open(dataset, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            i = row
            i [variable_index]= symbols_map[row[variable_index]]
            data.append(i) """
    file_content = read_file(new_file, result_directory)
    for row in file_content:
            i = row
            i [variable_index]= symbols_map[row[variable_index]]
            data.append(i)
    #Save converted dataset to new_file        
    half = int(math.ceil(len(data)/2))
    s1 = data[0:half]
    s2 = data[half:len(data)-1]
    write_file (data, new_file, result_directory)

"""Convert the given categorigal variables  to numerical variables, and save then in new_file, living in result_directory
    _PROTOCOL_INDEX = 1 / _SERVICE_INDEX = 2 / _FLAG_INDEX = 3
"""
def categorical_variables_conversion (dataset, variable_indexes, new_file, result_directory):
    #New dataset
    data= []
    new_row = []
    symbols = [{} for _ in range(len(variable_indexes))]
    #Open dataset file and modified categorical variables
    with open(dataset, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            symbol_val_index = 0
            for var_index in variable_indexes:
                if row[var_index] not in symbols[symbol_val_index]:
                    #Add a new 'symbol name' in the symbols map
                    symbols[symbol_val_index][row[var_index]] = len(symbols[symbol_val_index])
                #Update row value with its number equivalent
                new_row = row
                new_row [var_index] = symbols[symbol_val_index][row[var_index]]
                data.append(new_row)
                symbol_val_index += 1
    #Save converted dataset to new_file        
    half = int(math.ceil(len(data)/2))
    s1 = data[0:half]
    s2 = data[half:len(data)-1]
    write_file (data, new_file, result_directory)

    #Save symbols per variable
    #print(symbols[0])
    #np.savetxt('test', symbols[0], delimiter=',')
    variable_file =  result_directory+SYS_VARS.separator+ "train_var"
    for index in range (0, len(variable_indexes)-1):
        name = variable_file+str(variable_indexes[index])+'.txt'
        with open(name, 'w') as f:
            w = csv.writer(f, delimiter=',')
            for key, val in symbols[index].items():
                w.writerow([key, val])
            #np.savetxt(name, symbols[index], delimiter=',')

def simple_preprocessing_KDD():
    load_variables(SYS_VARS.KDDCup_path_names)
    """ TRY write and read
    write_file([[1, 2, 3],[4, 5, 6]], "trial.npy", SYS_VARS.KDDCup_path_result)
    sample = read_file("trial.npy", SYS_VARS.KDDCup_path_result)
    print (sample) """
    """Divide the samples into attacks of type A and rest
    select_attack('normal.', SYS_VARS.KDDCup_path_train)
    others = []
    attack = select_attack(attacks_map.keys()[22], SYS_VARS.KDDCup_path_train, _ATTACK_INDEX_KDD, others)
    print ('# attacks: ', len(attack))
    print ('# other flows:',len(others))"""

    """Plot all attacks statistics
    plot_attacks( SYS_VARS.KDDCup_path_train, _ATTACK_INDEX_KDD)"""

    """Convert ATTACKS symbol to integer
    symbolic_known_variable_conversion (SYS_VARS.KDDCup_path_train_10, attacks_map, _ATTACK_INDEX_KDD, 'KDD_train_attack_10.npy', SYS_VARS.KDDCup_path_result)"""

    """Convert ALL symbols to integers"""
    saved_preprocess = "KDD_train_num_10.npy"
    #TODO Generate the preprocessed samples
    categorical_variables_conversion (SYS_VARS.KDDCup_path_train_10, [_PROTOCOL_INDEX, _SERVICE_INDEX, _FLAG_INDEX], saved_preprocess, SYS_VARS.KDDCup_path_result)
    categorical_known_variable_conversion (SYS_VARS.KDDCup_path_result+SYS_VARS.separator+saved_preprocess, attacks_map, _ATTACK_INDEX_KDD, saved_preprocess, SYS_VARS.KDDCup_path_result)
    return read_file(saved_preprocess, SYS_VARS.KDDCup_path_result).astype(np.float)
    

                

##################### ATTACKS ################################
def select_attack(attack_name, dataset, a_index, other):
    print ('\n Find '+attack_name)
    csvFileArray = []

    with open(dataset, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        
        for row in reader:
            if row[a_index] == attack_name :
                csvFileArray.append(row)
            else:
                other.append(row)
    return csvFileArray
    
def get_attacks_percent (dataset, a_index):
    percents = {}
    for a, index in attacks_map.items():
        percents[a]= 0
    total = 0
    with open(dataset, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            total += 1
            for a, index in attacks_map.items():
                if row[a_index] == a :
                    percents[a]=percents[a]+1
                    break
    for a, index in attacks_map.items():
        percents[a]= 100*percents[a]/total
    return percents
                

def plot_attacks (dataset, a_index):
    attacks_percents = get_attacks_percent(dataset, a_index)
    
    #other = []
    #for attack, index in attacks_map.items():
        #attack_tot = select_attack(attack, dataset, a_index, other)
        #attacks_percents[attack] = len(attack_tot)/(len(attack_tot)+len(other))
    print ('\n RATIOS ')
    print (attacks_percents)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    #labels = attacks_percents.keys()
    keys = attacks_percents.keys()
    values = attacks_percents.values()
    print (sum(values))
    #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    #labels=keys,
    ax1.pie([float(v) for v in values],  autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

""" Separate dataset into classes 
        First simples version: each attack is a class"""
def separate_classes(train_data, key_index):
    #y = np.zeros((1, train_data.shape[1]))
    y = np.transpose(train_data[key_index, :])
    x = np.delete(train_data, key_index, 0)
    classes_names = attacks_map.keys()
    #Transpose to iterate row by row
    #for value in np.transpose(train_data):
        #y  = np.concatenate([y, [value[int(key_index)]]], axis=0)
    return x, y, classes_names





