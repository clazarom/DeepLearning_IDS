import os

separator = os.path.sep
key_val_separator = ':'


#Results
result_folder = 'preprocessing'

#KDD-1999 Dataset Directory
KDDCup_dataset = 'KDDCup'
KDDCup_train = 'kddcup.data.corrected'
KDDCup_train_10 = 'kddcup.data_10_percent_corrected'
KDDCup_test_10 = 'kddcup.testdata.unlabeled_10_percent'
KDDCup_test = 'kddcup.testdata.unlabeled'
KDDCup_var_names = 'kddcup.names.txt'

KDDCup_path_train = KDDCup_dataset + separator + KDDCup_train
KDDCup_path_test = KDDCup_dataset + separator + KDDCup_test
KDDCup_path_names = KDDCup_dataset + separator + KDDCup_var_names
#10% samples
KDDCup_path_train_10 = KDDCup_dataset + separator + KDDCup_train_10
KDDCup_path_test_10 = KDDCup_dataset + separator + KDDCup_test_10
KDDCup_path_result = KDDCup_dataset + separator + result_folder

#NSL-KDD Dataset Directory
NSLKDD_dataset = 'NSL-KDD'
NSLKDD_train = 'KDDTrain+.txt'
NSLKDD_test = 'KDDTest+.txt'
NSLKDD_train20 = 'KDDTrain+_20Percent.txt'
NSLKDD_test20 = 'KDDTest-21.txt'

NSLKDD_path_train = NSLKDD_dataset + separator + NSLKDD_train
NSLKDD_path_test = NSLKDD_dataset + separator + NSLKDD_test
NSLKDD_path_train20 = NSLKDD_dataset + separator + NSLKDD_train20
NSLKDD_path_test20 = NSLKDD_dataset + separator + NSLKDD_test20

NSL_path_result = KDDCup_dataset + separator + result_folder



