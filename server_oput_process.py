import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.neural_network as nn
import sklearn.pipeline as pl
import sklearn.preprocessing as pp
import subprocess


DEBUG = 0
num_sub = 5

def loadpart(terminate_str, temp):
    temp = temp.split('\', ')[1].replace(')', '')
    temp = temp.replace("\n", "")
    temp = temp.replace('array(', '')
    strinput = str()
    while terminate_str not in temp:
        if temp == '':
            break
        temp = temp.replace("\n", "")
        temp = temp.replace("array(", "")
        temp = temp.replace(")", "")
        strinput += temp.strip()
        temp = f.readline()
    return strinput.replace("))", ""), temp

# load content to file
startline = 1623692
endline = 1719248
filename = '~/FC_log_lin'
c = 'sed -n \''+str(startline)+', '+str(endline)+'p\' '+filename+' > ~/temp'
print c
a = subprocess.check_output(c, shell=True)

f = open("/home/lme/temp", 'r')
temp = f.readline()
roundcnt = int(temp.split('\', ')[1].split(')')[0])
temp = f.readline()
size = int(temp.split('\', ')[1].split(')')[0])
if DEBUG:
    print roundcnt

'''subskill coefficient and subskill training set'''
# subskill No.0
temp = f.readline()
if "subskill No." not in temp and "0" not in temp:
    print "error on finding subskill No.0."
    exit(1)
temp = f.readline()
sub_0_mod_coef, temp = loadpart("model intercept", temp)
sub_0_intercept, temp = loadpart("model featured input", temp)
sub_0_mod_input, temp = loadpart("subskill No.", temp)
if DEBUG:
    print sub_0_mod_coef

# subskill No.1
if "subskill No." not in temp and "1" not in temp:
    print "error on finding subskill No.1."
    exit(1)
temp = f.readline()
sub_1_mod_coef, temp = loadpart("model intercept", temp)
sub_1_intercept, temp = loadpart("model featured input", temp)
sub_1_mod_input, temp = loadpart("subskill No.", temp)

# subskill No.2
if "subskill No." not in temp and "2" not in temp:
    print "error on finding subskill No.2."
    exit(1)
temp = f.readline()
sub_2_mod_coef, temp = loadpart("model intercept", temp)
sub_2_intercept, temp = loadpart("model featured input", temp)
sub_2_mod_input, temp = loadpart("subskill No.", temp)


# subskill No.3
if "subskill No." not in temp and "3" not in temp:
    print "error on finding subskill No.3."
    exit(1)
temp = f.readline()
sub_3_mod_coef, temp = loadpart("model intercept", temp)
sub_3_intercept, temp = loadpart("model featured input", temp)
sub_3_mod_input, temp = loadpart("subskill No.", temp)

# subskill No.4
if "subskill No." not in temp and "4" not in temp:
    print "error on finding subskill No.4."
    exit(1)
temp = f.readline()
sub_4_mod_coef, temp = loadpart("model intercept", temp)
sub_4_intercept, temp = loadpart("model featured input", temp)
sub_4_mod_input, temp = loadpart("data input", temp)


# data input
data_inputs, temp = loadpart("data output", temp)
if DEBUG:
    print data_inputs

# data output
data_outputs, temp = loadpart("predict subskill", temp)

if DEBUG:
    print data_outputs

# predict subskill
pre_subskill, temp = loadpart("data subargument", temp)


# data subargument
data_subarguments, temp = loadpart("predict subargument", temp)

if DEBUG:
    print data_subarguments

# predict subargument
pre_subarguments, temp = loadpart("sub_corr", temp)
if DEBUG:
    print pre_subarguments

# sub_corr
sub_corr = eval(temp.split('\', ')[1].split(')')[0])

# arg_rmse
temp = f.readline()
arg_rmse, temp = loadpart("sth", temp)
if DEBUG:
    print arg_rmse


# tmp = []
# step_idxs = [[] for _ in range(num_sub)]
# for step_idx, sub in enumerate(pre_subskill):
#     step_idxs[sub].append(step_idx)
# for step_idx in step_idxs[0]:
#     tmp.append(data_inputs[step_idx])
# test = []
# for input in tmp:
#     temp = []
#     for oput_idx in range(len(sub_0_mod_coef)):
#         x = 0
#         for iput_idx in range(len(input)):
#             x += sub_0_mod_coef[oput_idx][iput_idx]*input[iput_idx]
#         temp.append(x)
# print(temp)

sub_0_mod_coef= eval(sub_0_mod_coef)
sub_0_intercept = eval(sub_0_intercept)
sub_0_mod_input = eval(sub_0_mod_input)

sub_1_mod_coef= eval(sub_1_mod_coef)
sub_1_intercept = eval(sub_1_intercept)
sub_1_mod_input = eval(sub_1_mod_input)

sub_2_mod_coef= eval(sub_2_mod_coef)
sub_2_intercept = eval(sub_2_intercept)
sub_2_mod_input = eval(sub_2_mod_input)

sub_3_mod_coef= eval(sub_3_mod_coef)
sub_3_intercept = eval(sub_3_intercept)
sub_3_mod_input = eval(sub_3_mod_input)

data_inputs = eval(data_inputs)
data_outputs = eval(data_outputs)
data_subarguments = eval(data_subarguments)
pre_subarguments = eval(pre_subarguments)
pre_subskill = eval(pre_subskill)
arg_rmse = eval(arg_rmse)

if not all((len(data_inputs)==len(data_outputs), len(data_inputs)==len(data_outputs), len(data_outputs)==len(data_subarguments), len(data_subarguments)==len(pre_subarguments))):
    print("error: did not load data correctly")
    exit(1)


def showcoeffmodel(sub_name, sub_mod_coef, data_inputs, poly=2):
    if poly == 2:
        model = pp.PolynomialFeatures(2)
        a =  model.fit_transform(data_inputs)
        feature = model.get_feature_names()
    else:
        # linear feature
        feature = []
        for i in range(len(data_inputs[0])):
            feature.append('x_'+str(i))
    for sub_arg_idx in range(len(sub_mod_coef)):
        di = zip(feature, sub_mod_coef[sub_arg_idx])
        large_val = []
        for feature_name, val in di:
            if abs(val) >1e-3:
                large_val.append((val, feature_name))
        print("coefficient with large value in subskill "+str(sub_name)+" model, subargument No.", str(sub_arg_idx), ":", sorted(large_val, reverse=True))

def showerrorstep(pre_subarguments, data_subarguments, data_outputs, data_inputs):
    for i in range(0, len(pre_subarguments)):
        arg_check = [1.0e-03 < abs(data_subarguments[i][argidx] - pre_subarguments[i][argidx]) for argidx in
                     range(0, len(pre_subarguments[i]))]
        # print(arg_check)
        if any(arg_check):
            print(arg_check)
            print('subskill: ', data_outputs[i])
            print('input: ', data_inputs[i])
            print('data subarg: ', data_subarguments[i])
            print('predicted subargument: ', pre_subarguments[i])
            print('\n')


def showrankpoly(sub_name, sub_mod_input, data_inputs, pre_subskill):
    '''rank check for training set'''
    rank = np.linalg.matrix_rank(np.array(sub_mod_input))
    print("rank for training input of sub skill "+ str(sub_name)+":", rank)
    rank = np.linalg.matrix_rank(np.array(sub_mod_input)[0:, 0:len(data_inputs[0]) + 1])
    print("Before poly feature, rank for training input of sub skill "+ str(sub_name)+":", rank)

    '''create feature input matrix of both training and validation set'''
    model = pp.PolynomialFeatures(degree=2)
    tmp = []
    step_idxs = [[] for _ in range(num_sub)]
    for step_idx, sub in enumerate(pre_subskill):
        step_idxs[sub].append(step_idx)
    for step_idx in step_idxs[sub_name]:
        tmp.append(data_inputs[step_idx])
    print()
    # subskill No.0
    valid_input = model.fit_transform(np.array(tmp))
    valid_input = np.concatenate((valid_input, np.array(sub_mod_input)), axis=0)
    rank = np.linalg.matrix_rank(valid_input)
    print("rank for training and validation input of sub skill " + str(sub_name)+":", rank)
    rank = np.linalg.matrix_rank(valid_input[0:, 0:len(data_inputs[0]) + 1])
    print("before Poly feature, rank for training and validation input of sub skill "+ str(sub_name)+":", rank)

def showranklin(sub_name, sub_mod_input, data_inputs, pre_subskill, poly=2):
    '''rank check for training set'''
    rank = np.linalg.matrix_rank(np.array(sub_mod_input))
    print("rank for training input of sub skill "+ str(sub_name)+":", rank)
    rank = np.linalg.matrix_rank(np.array(sub_mod_input)[0:, 0:len(data_inputs[0]) + 1])
    print("Before poly feature, rank for training input of sub skill "+ str(sub_name)+":", rank)

    '''create feature input matrix of both training and validation set'''
    tmp = []
    step_idxs = [[] for _ in range(num_sub)]
    for step_idx, sub in enumerate(pre_subskill):
        step_idxs[sub].append(step_idx)
    for step_idx in step_idxs[sub_name]:
        tmp.append(data_inputs[step_idx])
    print()
    # subskill No.0
    valid_input = np.array(tmp)
    if poly == 2:
        model = pp.PolynomialFeatures(2)
        valid_input = model.fit_transform(valid_input)
    valid_input = np.concatenate((valid_input, np.array(sub_mod_input)), axis=0)
    rank = np.linalg.matrix_rank(valid_input)
    print("rank for training and validation input of sub skill " + str(sub_name)+":", rank)
    rank = np.linalg.matrix_rank(valid_input[0:, 0:len(data_inputs[0]) + 1])
    print("before Poly feature, rank for training and validation input of sub skill "+ str(sub_name)+":", rank)

def getorthonormalset(sub_name, iputmatrix, data_inputs, poly=2):
    x = np.array(iputmatrix[0])
    x = x/np.linalg.norm(x)
    orthoset = [x]
    for iputvec in iputmatrix[1:]:
        iputvec = np.array(iputvec)
        span = False
        for orthvec in orthoset:
            iputvec = iputvec - np.dot(iputvec, orthvec)*orthvec
            if np.linalg.norm( iputvec - np.zeros(len(iputvec))) < 1e-10:
                span = True
                break
        if not span:
            iputvec = iputvec/np.linalg.norm(iputvec)
            orthoset.append(iputvec)
    # if DEBUG:
    print "size of span: ", len(orthoset)

    # check validation set dimension
    '''create feature input matrix of both training and validation set'''
    model = pp.PolynomialFeatures(degree=2)
    tmp2 = []
    tmp = []
    step_idxs = [[] for _ in range(num_sub)]
    for step_idx, sub in enumerate(pre_subskill):
        step_idxs[sub].append(step_idx)
    for step_idx in step_idxs[sub_name]:
        tmp.append(data_inputs[step_idx])
    if poly == 2:
        tmp2 = model.fit_transform(tmp)
    elif poly == 1:
        tmp2 = tmp
    print()
    for idx in range(len(tmp2)):
        iputvec = np.array(tmp2[idx])
        span = False
        for orthvec in orthoset:
            iputvec = iputvec - np.dot(iputvec, orthvec)*orthvec
            if np.linalg.norm( iputvec - np.zeros(len(iputvec))) < 1e-10:
                span = True
                break
        if not span:
            print("Not spanned", tmp[idx])

def checksubname(data_outputs, pred_subskill, data_inputs):
    wrong_pre = []
    for idx in range(len(data_outputs)):
        if data_outputs[idx] != pred_subskill[idx]:
            wrong_pre.append((data_outputs[idx], pred_subskill[idx]))
            print('Check subname: error pair:', (data_outputs[idx], pred_subskill[idx]))
            print('Check subname: data input:', data_inputs[idx])


# showranklin(0, sub_0_mod_input, data_inputs, pre_subskill)
# getorthonormalset(0, sub_0_mod_input, data_inputs, 1)
# showranklin(1, sub_1_mod_input, data_inputs, pre_subskill)
# getorthonormalset(1, sub_1_mod_input, data_inputs, 1)
# showranklin(2, sub_2_mod_input, data_inputs, pre_subskill)
# getorthonormalset(2, sub_2_mod_input, data_inputs, 1)
# showerrorstep(pre_subarguments, data_subarguments, data_outputs, data_inputs)
# showcoeffmodel(0, sub_0_mod_coef, data_inputs, 1)
# showcoeffmodel(1, sub_1_mod_coef, data_inputs, 1)
# showcoeffmodel(2, sub_2_mod_coef, data_inputs, 1)
# showcoeffmodel(3, sub_3_mod_coef, data_inputs, 1)
checksubname(data_outputs, pre_subskill, data_inputs)
