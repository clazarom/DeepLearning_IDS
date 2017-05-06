import csv
import SYS_VARS
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
import operator


#Maps for some categorical variables
attacks_map  = {} #attacks_map = {attack_name: attack_index, ...}
feature_names = []

#Known columns/indexes for categorical variables  
_ATTACK_INDEX_KDD  = -1
_ATTACK_INDEX_NSLKDD = -2
_PROTOCOL_INDEX = 1
_SERVICE_INDEX = 2
_FLAG_INDEX = 3

_attack_row  = 0

#Attacks classification: 5 types {Probe, DoS, R2L, U2R, Normal}
_attack_classes ={'ipsweep.': 'probe', 'loadmodule.': 'u2r', 'warezclient.': 'r2l', 'phf.': 'r2l', 'portsweep.': 'probe', 
                  'smurf.': 'dos', 'imap.': 'r2l', 'multihop.': 'r2l', 'rootkit.': 'u2r', 'satan.': 'probe', 'nmap.': 'probe', 
                  'back.': 'dos', 'ftp_write.': 'r2l', 'neptune.': 'dos', 'teardrop.': 'dos', 'perl.': 'u2r', 'guess_passwd.': 'r2l', 
                  'pod.': 'dos', 'normal.': 'normal', 'buffer_overflow.': 'u2r', 'warezmaster.': 'r2l', 'spy.': 'r2l', 'land.': 'dos'}
#List of attacks in NSL - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.413.589&rep=rep1&type=pdf
_attack_classes_NSL ={'ipsweep': 'probe', 'loadmodule': 'u2r', 'warezclient': 'r2l', 'phf': 'r2l', 'portsweep': 'probe', 
                  'smurf': 'dos', 'imap': 'r2l', 'multihop': 'r2l', 'rootkit': 'u2r', 'satan': 'probe', 'nmap': 'probe', 
                  'back': 'dos', 'ftp_write': 'r2l', 'neptune': 'dos', 'teardrop': 'dos', 'perl': 'u2r', 'guess_passwd': 'r2l', 
                  'pod': 'dos', 'normal': 'normal', 'buffer_overflow': 'u2r', 'warezmaster': 'r2l', 'spy': 'r2l', 'land': 'dos', 
                  'snmpguess':'r2l', 'processtable': 'dos', 'saint':'probe', 'mscan':'probe', 'apache2':'dos', 'httptunnel':'r2l',
                  'mailbomb':'dos', 'snmpgetattack':'r2l', 'worm':'dos', 'sendmail':'r2l', 'xlock':'r2l', 'xterm':'u2r', 'xsnoop':'r2l',
                  'ps':'u2r', 'named':'r2l', 'udpstorm':'dos', 'sqlattack':'u2r'}
_attack_classes_num = {'probe':1, 'dos':2, 'r2l':3, 'u2r':4, 'normal':0}


def load_variables (var_names):
    """ Load given dataset file, var_names, and generate attack_map & feature_names """
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

        
def write_file (data, file_name, file_directory):
    """Write data into a file_name located in file_directory """
    variable_file =  file_directory+SYS_VARS.separator+file_name
    with open(variable_file, 'wb') as f_handle:
        #w = csv.writer(f_handle, delimiter=',')
        np.save(f_handle, data)
        #np.savetxt(name, symbols[index], delimiter=',')

def read_file(file_name, file_directory):
    """Open a a file_name located in file_directory and return its content as numpy darray """
    variable_file =  file_directory+SYS_VARS.separator+file_name
    return np.load(variable_file)


##################### DATA PREPROCESSING #####################

def categorical_labels_conversion (dataset, symbols_map, variable_index, classes, new_file, result_directory, processed_file):
    """Convert the given categorigal variables  to numerical variables, and save then in new_file, living in result_directory
    When we know the mapping of catogry to number
    """
    data= []
    i = []
    save_file = result_directory + SYS_VARS.separator + new_file

    #Open dataset file and modified categorical variable:
    #   - Convert to its attack type (according to attack_classes)
    #   - Convert to numerical values
    if (processed_file):
        file_content = read_file(dataset, file_directory)
        for row in file_content:
                i = row
                i [variable_index]= symbols_map[row[variable_index]]
                data.append(i)
    else:
        with open(dataset, 'r') as file:
            reader = csv.reader(file, delimiter = ',')
            for row in reader:
                i = row
                #i [variable_index]= symbols_map[row[variable_index]]
                i [variable_index]= _attack_classes_num[classes[row[variable_index]]]
                data.append(i)

    """ Read from process data
    file_content =read_file(dataset, file_directory)
    for row in file_content:
            i = row
            i [variable_index]= symbols_map[row[variable_index]]
            data.append(i)"""
    #Save converted dataset to new_file        
    half = int(math.ceil(len(data)/2))
    s1 = data[0:half]
    s2 = data[half:len(data)-1]
    write_file (data, new_file, result_directory)
    return data



def categorical_features_conversion (dataset, variable_indexes, new_file, result_directory, processed_file):
    """Convert the given categorigal variables  to numerical variables, and save then in new_file, living in result_directory
    _PROTOCOL_INDEX = 1 / _SERVICE_INDEX = 2 / _FLAG_INDEX = 3
    """

    #New dataset
    data= []
    new_row = []
    symbols = [{} for _ in range(len(variable_indexes))]
    #Open dataset file and modified categorical variables
    if (processed_file):
        file_content = read_file(dataset, result_directory)
        for row in file_content:
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
    else:
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
    data = np.array(data)               
    return data
   

def categorical_features_onehot (dataset, variable_indexes, new_file, result_directory, processed_file):
    """Convert the categorical feature to numerical using onehot bit encoding and save then in new_file, living in result_directory
    _PROTOCOL_INDEX = 1 / _SERVICE_INDEX = 2 / _FLAG_INDEX = 3"""

    #Convert to numerical
    d_numerical = categorical_features_conversion (dataset, variable_indexes, new_file, result_directory, processed_file)

    #OneHot encoding
    enc = OneHotEncoder(categorical_features = variable_indexes)
    enc.fit(d_numerical)

    #Save converted dataset to new_file        
    half = int(math.ceil(len(d_numerical)/2))
    s1 = d_numerical[0:half]
    s2 = d_numerical[half:len(d_numerical)-1]
    write_file (d_numerical, new_file, result_directory)



def simple_preprocessing_KDD(attack_index):
    """Load and convert train and test datasets """

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


    """Convert ATTACKS symbol to integer
    symbolic_known_variable_conversion (SYS_VARS.KDDCup_path_train_10, attacks_map, _ATTACK_INDEX_KDD, _attack_classes, 'KDD_train_attack_10.npy', SYS_VARS.KDDCup_path_result)"""

    t_data = []
    test_data = []
    if (attack_index == _ATTACK_INDEX_KDD):
        """KDD Convert ALL symbols to integers"""
        #1. TRAIN DATA
        saved_preprocess = "KDD_train_num_10.npy"
        # Generate the preprocessed samples
        categorical_labels_conversion (SYS_VARS.KDDCup_path_train_10 , attacks_map, _ATTACK_INDEX_KDD, _attack_classes, saved_preprocess, SYS_VARS.KDDCup_path_result, False)
        categorical_features_onehot (saved_preprocess, [_PROTOCOL_INDEX, _SERVICE_INDEX, _FLAG_INDEX], saved_preprocess, SYS_VARS.KDDCup_path_result, True)
        t_data = read_file(saved_preprocess, SYS_VARS.KDDCup_path_result).astype(np.float)
        #2. TEST DATA
        saved_preprocess = "KDD_test_num_10.npy"
        # Generate the preprocessed samples
        categorical_features_onehot (SYS_VARS.KDDCup_path_test_10 , [_PROTOCOL_INDEX, _SERVICE_INDEX, _FLAG_INDEX], saved_preprocess, SYS_VARS.KDDCup_path_result, False)
        test_data = read_file(saved_preprocess, SYS_VARS.KDDCup_path_result).astype(np.float)
        
    elif (attack_index == _ATTACK_INDEX_NSLKDD):
        """NSL-KDD Convert ALL symbols to integers"""
        #1. TRAIN DATA
        saved_preprocess = "NSL_train_num_10.npy"
        # Generate the preprocessed samples
        categorical_labels_conversion (SYS_VARS.NSLKDD_path_train20 , attacks_map, _ATTACK_INDEX_NSLKDD, _attack_classes_NSL, saved_preprocess, SYS_VARS.NSL_path_result, False)
        categorical_features_onehot (saved_preprocess, [_PROTOCOL_INDEX, _SERVICE_INDEX, _FLAG_INDEX], saved_preprocess, SYS_VARS.NSL_path_result, True)
        t_data = read_file(saved_preprocess, SYS_VARS.NSL_path_result).astype(np.float)
        #2. TEST DATA
        saved_preprocess = "NSL_test_num_10.npy"
        # Generate the preprocessed samples
        categorical_labels_conversion (SYS_VARS.NSLKDD_path_test20 , attacks_map, _ATTACK_INDEX_NSLKDD, _attack_classes_NSL, saved_preprocess, SYS_VARS.NSL_path_result, False)
        categorical_features_onehot (saved_preprocess , [_PROTOCOL_INDEX, _SERVICE_INDEX, _FLAG_INDEX], saved_preprocess, SYS_VARS.NSL_path_result, True)
        test_data = read_file(saved_preprocess, SYS_VARS.NSL_path_result).astype(np.float)

    return t_data, test_data
    

                

##################### ATTACKS ################################
def select_attack(attack_name, dataset, a_index, other):
    """Get one attack by its name"""
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
    
def get_attacks_percent (a_data, a_index, dataset= None):
    """ Return the % for each attack in a_index of the dataset """
    percents = {}
    attacks = {}
    #for a, index in attacks_map.items():
        #percents[a]= 0
    for a, index in _attack_classes.items():
        percents[a] = 0
    total = 0

    if dataset is None:
        print('Dont open file here')
        for row in np.transpose(a_data)[a_index]:
            total += 1
            for a, index in _attack_classes_num.items():
                #print(str(row))
                if row[a_index] == index :
                    percents[a]=percents[a]+1
                    break           
    else:   
        with open(dataset, 'r') as file:
            reader = csv.reader(file, delimiter = ',')
            for row in reader:
                total += 1
                for a, index in attacks_map.items():
                    if row[a_index] == a :
                        #percents[a]=percents[a]+1
                        four_attack = _attack_classes[row[variable_index]]
                        percents[four_attack] = percents[four_attack]+1
                        break
                    
    for a, index in attacks_map.items():
        percents[a]= 100*percents[a]/total
    return percents
                

def plot_attacks (a_index, dataset= None,  attacks_data = None):
    """ Plot the % for each attack in a_index of the dataset """
    if attacks_data is None:
        attacks_percents = get_attacks_percent(a_index=a_index, dataset=dataset)
        
        #other = []
        #for attack, index in attacks_map.items():
            #attack_tot = select_attack(attack, dataset, a_index, other)
            #attacks_percents[attack] = len(attack_tot)/(len(attack_tot)+len(other))
        print ('\n RATIOS ')
        print (attacks_percents)
    else:
         attacks_percents = get_attacks_percent(a_index=a_index, a_data=attacks_data)
        

    #labels = attacks_percents.keys()
    keys = attacks_percents.keys()
    values = attacks_percents.values()
    print (sum(values))
    #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    y = [float(v) for v in values]
    plot_percentages(y, list(keys), 'ATTACKS')



def separate_classes(train_data, key_index):
    """ Separate dataset into classes 
        First simples version: each attack is a class"""

    #y = np.zeros((1, train_data.shape[1]))
    y = np.transpose(train_data[key_index, :])
    x = np.delete(train_data, key_index, 0)
    #ALL ATTACKS
    #classes_names = attacks_map.keys()
    
    #ATTACK CLASSES
    sorted_attacks = sorted(_attack_classes_num.items(), key=operator.itemgetter(1))
    #classes_names = _attack_classes_num.keys()
    #classes_values =   _attack_classes_num.values()
    classes_values=[]
    classes_names= []
    for c, v in sorted_attacks:
        classes_names.append(c)
        classes_values.append(v)
    #Transpose to iterate row by row
    #for value in np.transpose(train_data):
        #y  = np.concatenate([y, [value[int(key_index)]]], axis=0)
    return x, y, classes_names, classes_values

def plot_various():
    """Plot the attacks of train 10% dataset """
    load_variables(SYS_VARS.KDDCup_path_names)
    saved_preprocess = "KDD_train_num_10.npy"
    data =  categorical_labels_conversion (SYS_VARS.KDDCup_path_train_10 , attacks_map, _ATTACK_INDEX_KDD, saved_preprocess, SYS_VARS.KDDCup_path_result, False)
    plot_attacks( attacks_data = data, a_index = _ATTACK_INDEX_KDD)
    #plot_attacks( dataset = SYS_VARS.KDDCup_path_train_10, a_index = _ATTACK_INDEX_KDD)

def plot_percentages(outputs, o_names, title, list_values= None):
    """Compute the % for each variable in the list and plot as: 
        Pie chart & Bar graph """
    #Plotting parameters
    N = len(outputs)
    x = range(N)
    matplotlib.rc('xtick', labelsize=5) 
    width = 1/1.5
    
    plotting = [0]*N
    if list_values is None:
        print('percentages in inputs')
        plotting = outputs
       
    else:
        #Compute the percentages of the list
        total = 0
        for v in list_values:
            total += 1
            plotting[int(v)] += 1
        plotting =[ 100*i/total for i in plotting]
        

    #PLOT PIE CHART
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig1, ax1 = plt.subplots()
    #labels=keys,
    ax1.pie(plotting,  autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('KDD 10% - '+title)
    plt.show()

    #PLOT BAR DIAGRAM
    plt.xticks(x, o_names)
    plt.bar(x, plotting, width, color="blue", alpha=0.5)
    #plt.bar(y_pos, performance, align='center', alpha=0.5)
    #plt.xticks(y_pos, objects)
    plt.ylabel('%')
    plt.title('KDD 10% - '+title)
    fig = plt.gcf()
    plt.show()










