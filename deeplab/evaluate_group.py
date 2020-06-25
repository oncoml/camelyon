# Evaluate group of patients

import csv
from evaluate_patient import *

group_list = ["024"]
main_result = []

def evaluate_group(group_list):
    for num in group_list:
        patient_result, macro, micro, itc = evaluate_patient(num)
        for i in range(5):
            string = "patient_{}_node_00{}.tif, ".format(num, i)
            if macro[i]:
                main_result.append(string + "macro")
            elif micro[i]:
                main_result.append(string + "micro")
            elif itc[i]:
                main_result.append(string + "itc")
            else:
                main_result.append(string + "negative")
        main_result.append("patient_{}, {}".format(num, patient_result))

    with open('output_group.csv','w') as f:
        for l in main_result:
            f.write(l + '\n')

