# Evaluates all slides of nodes from a single patient

import os

from evaluate_slide import *


def evaluate_patient(patient_num):
    root_dir = '/home/steveyang/Disk/Camelyon17/Train/'
    folder_name = 'patient_' + patient_num
    folder_dir = root_dir + folder_name + '/'
    nodes_files = os.listdir(folder_dir)

    macro = [[] for _ in range(5)]
    micro = [[] for _ in range(5)] 
    itc = [[] for _ in range(5)]
    nodes = [[] for _ in range(5)]

    for i in range(len(nodes_files)):
        try:
            dims = evaluate(folder_dir, nodes_files[i])
        except:
            continue

        for j in dims:
            if j > 2000: macro[i].append(1)
            elif i > 200: micro[i].append(1)
            else: itc[i].append(1)

        if macro[i] or micro[i] or itc[i]:
            if macro[i]: print("Macro detected in", nodes_files[i])
            if micro[i]: print("Micro detected in", nodes_files[i])
            if itc[i]: print("ITC detected in", nodes_files[i])
            nodes[i].append(1)
    
    print("Nodes 1 - 5 positivity: ", nodes)
    sum_nodes = sum(sum(x) for x in nodes) 

    if sum_nodes == 0: 
        patient_result = "pN0"
    elif (not any(macro) and not any(micro) and itc):
        patient_result = "pN0(i+)"
    elif (not any(macro) and (micro or itc)):
        patient_result = "pN1mi"
    elif (any([1 for x in macro if any(x)]) and (sum_nodes > 1 and sum_nodes < 4)):
        patient_result = "pN1" 
    elif (any([1 for x in macro if any(x)]) and (sum_nodes == 4)):
        patient_result = "pN2"
    else:
        patient_result = "N/A"

    return patient_result, macro, micro, itc

    
