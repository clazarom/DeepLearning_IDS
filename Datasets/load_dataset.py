import csv
import SYS_VARS
import matplotlib.pyplot as plt
import numpy as np
import math


attacks_map  = {}
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
        for attack in csvFileArray[_attack_row]:
            attacks_map[attack+'.'] = i
            i += 1

        print ('\n ATTACKS')
        print(attacks_map )
        print ('\n FEATURES')
        print(feature_names)

##################### DATA PREPROCESSING #####################
def symbolic_known_variable_conversion (dataset, symbols, variable_index, new_file, result_directory):
    data= []
    i = []
    save_file = result_directory + SYS_VARS.separator + new_file
    with open(dataset, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            i = row
            i [variable_index]= symbols[row[variable_index]]
            data.append(i)
    with open(save_file, 'wb') as f_handle:
        half = int(math.ceil(len(data)/2))
        s1 = data[0:half]
        s2 = data[half:len(data)-1]
        np.save(f_handle, s1)
        np.save(f_handle, s2)


def symbolic_variables_conversion (dataset, variables, new_file, result_directory):
    #_PROTOCOL_INDEX = 1 / _SERVICE_INDEX = 2 / _FLAG_INDEX = 3
    data= []
    i = []
    symbols = [{} for _ in range(len(variables))]
    save_file = result_directory + SYS_VARS.separator + new_file

    
    with open(dataset, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            index = 0
            for var in variables:
                if row[var] not in symbols[index]:
                    symbols[index][row[var]] = len(symbols[index])
                i = row
                i [var] = symbols[index][row[var]]
                data.append(i)
                index += 1
    #Save converted dataset to new_file
    with open(save_file, 'wb') as f_handle:
        half = int(math.ceil(len(data)/2))
        s1 = data[0:half]
        s2 = data[half:len(data)-1]

        np.save(f_handle, s1)
        np.save(f_handle, s2)

    #Save symbols per variable
    #print(symbols[0])
    #np.savetxt('test', symbols[0], delimiter=',')   
    for index in range (0, len(variables)-1):
        name = 'train_var'+str(variables[index])+'.txt'
        with open(name, 'w') as f:
            w = csv.writer(f, delimiter=',')
            for key, val in symbols[index].items():
                w.writerow([key, val])
            #np.savetxt(name, symbols[index], delimiter=',')

def simple_preprocessing_KDD():
    load_variables(SYS_VARS.KDDCup_path_names)
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
    saved_preprocess = 'KDD_train_num_10.npy'
    #symbolic_variables_conversion (SYS_VARS.KDDCup_path_train_10, [_PROTOCOL_INDEX, _SERVICE_INDEX, _FLAG_INDEX, _ATTACK_INDEX_KDD], saved_preprocess, SYS_VARS.KDDCup_path_result)
    #print ('\n SAMPLE numeric')
    #var = np.load('KDD_train_num.npy')
    #print(var[0])
    return np.transpose(np.load(SYS_VARS.KDDCup_path_result+SYS_VARS.separator+saved_preprocess).astype(np.float))
    

                

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





