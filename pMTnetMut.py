import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import csv
import random
import os
from io import StringIO
import keras
from keras.layers import Input,Dense,concatenate,Dropout,LSTM
from keras.models import Model,load_model                                                      
from keras.optimizers import Adam, Adamax, RMSprop, Adagrad, Adadelta, Nadam
from keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint
import argparse
random.seed(54321)
##Customer Input
#python pMTnet.py -input input.csv -library library_dir 

#added argumentParser; added option to name output file -ZZ
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=os.path.join('test', 'input', 'test_input.csv'),
                    help='input protein seq file')
parser.add_argument('--library', type=str, default='library',
                    help='diretory to the downloaded library')
parser.add_argument('--output', type=str, default=os.path.join('test','output'),
                    help='diretory to hold encoding and prediction output')
parser.add_argument('--output_log', type=str, default=os.path.join('test','output','mutTestOutput.log'),
                    help='diretory for logs')
parser.add_argument('--outputFilename', type=str, default='mutTest_prediction.csv',
                    help='filename of result csv of computed ranks')
parser.add_argument('--mutateSeq', type=str, default='',
                    help='select whether to mutate antigen or TCRb CDR3. Valid inputs: ''a'' = antigen, ''c'' = TCRbeta CDR3, "''" = none')
parser.add_argument('--mutationType', type=str, default='',
                    help='simulate mutants of specified sequence; leave blank for "gentle" mutant. Valid inputs: ''a'' = alaline scan, ''x'' = X scan')
parser.add_argument('--gentleMut', type=float, default=0,
                    help='factor to modify encoded values for sequences undergoing "gentle" mutation\n\toverwrites mutationType input')
args = parser.parse_args()
file_dir = args.input
library_dir = args.library
model_dir = os.path.join(library_dir, 'h5_file')
aa_dict_dir = os.path.join(library_dir, 'Atchley_factors.csv')
hla_db_dir = os.path.join(library_dir, 'hla_library')
output_dir = args.output
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
output_log_dir = args.output_log
outputFn = args.outputFilename
if not '.csv' in outputFn:
    outputFns = outputFn + '.csv'
else:
    outputFns = outputFn
outputFilePath = os.path.join(output_dir, outputFns)

mutSeq = args.mutateSeq
if mutSeq == 'a':
    mutAntigen = True
    mutCDR = False
elif mutSeq == 'c':
    mutCDR = True
    mutAntigen = False
else:
    mutAntigen = False
    mutCDR = False
mutType = args.mutationType
mutFactor = args.gentleMut
if mutType == 'a':
    alaScan = True
    xScan = False
    gentleMut = False
elif mutType == 'x':
    xScan = True
    gentleMut = False
    alaScan = False
elif mutFactor != float(0):
    gentleMut = True
    alaScan = False
    xScan = False
else:
    gentleMut = False
    alaScan = False
    xScan = False
if (not mutAntigen) and (not mutCDR):
    if alaScan:
        raise UnboundLocalError('no seqence specified for alanine scan')
    if xScan:
        raise UnboundLocalError('no seqence specified for X scan')
    if gentleMut:
        raise UnboundLocalError('no seqence specified for gentle mutation scan')
#if mutCDR and xScan:
    

'''
args = sys.argv
file_dir=args[args.index('-input')+1] #input protein seq file
library_dir=args[args.index('-library')+1] #directory to downloaded library

model_dir=library_dir+'/h5_file'
aa_dict_dir=library_dir+'/Atchley_factors.csv' #embedding vector for tcr encoding
hla_db_dir=library_dir+'/hla_library/' #hla sequence
output_dir=args[args.index('-output')+1] #diretory to hold encoding and prediction output
output_log_dir=args[args.index('-output_log')+1] #standard output
'''
################################
# Reading Encoding Matrix #
################################
########################### Atchley's factors#######################
aa_dict_atchley=dict()
with open(aa_dict_dir,'r') as aa:
    aa_reader=csv.reader(aa)
    next(aa_reader, None)
    for rows in aa_reader:
        aa_name=rows[0]
        aa_factor=rows[1:len(rows)]
        aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float')
########################### One Hot ##########################   
aa_dict_one_hot = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,
           'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,
           'W': 18,'Y': 19,'X': 20}  # 'X' is a padding variable
########################### Blosum ########################## 
BLOSUM50_MATRIX = pd.read_table(StringIO(u"""                                                                                      
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *                                                           
A  5 -2 -1 -2 -1 -1 -1  0 -2 -1 -2 -1 -1 -3 -1  1  0 -3 -2  0 -2 -2 -1 -1 -5                                                           
R -2  7 -1 -2 -4  1  0 -3  0 -4 -3  3 -2 -3 -3 -1 -1 -3 -1 -3 -1 -3  0 -1 -5                                                           
N -1 -1  7  2 -2  0  0  0  1 -3 -4  0 -2 -4 -2  1  0 -4 -2 -3  5 -4  0 -1 -5                                                           
D -2 -2  2  8 -4  0  2 -1 -1 -4 -4 -1 -4 -5 -1  0 -1 -5 -3 -4  6 -4  1 -1 -5                                                           
C -1 -4 -2 -4 13 -3 -3 -3 -3 -2 -2 -3 -2 -2 -4 -1 -1 -5 -3 -1 -3 -2 -3 -1 -5                                                           
Q -1  1  0  0 -3  7  2 -2  1 -3 -2  2  0 -4 -1  0 -1 -1 -1 -3  0 -3  4 -1 -5                                                           
E -1  0  0  2 -3  2  6 -3  0 -4 -3  1 -2 -3 -1 -1 -1 -3 -2 -3  1 -3  5 -1 -5                                                           
G  0 -3  0 -1 -3 -2 -3  8 -2 -4 -4 -2 -3 -4 -2  0 -2 -3 -3 -4 -1 -4 -2 -1 -5                                                           
H -2  0  1 -1 -3  1  0 -2 10 -4 -3  0 -1 -1 -2 -1 -2 -3  2 -4  0 -3  0 -1 -5                                                          
I -1 -4 -3 -4 -2 -3 -4 -4 -4  5  2 -3  2  0 -3 -3 -1 -3 -1  4 -4  4 -3 -1 -5                                                           
L -2 -3 -4 -4 -2 -2 -3 -4 -3  2  5 -3  3  1 -4 -3 -1 -2 -1  1 -4  4 -3 -1 -5                                                           
K -1  3  0 -1 -3  2  1 -2  0 -3 -3  6 -2 -4 -1  0 -1 -3 -2 -3  0 -3  1 -1 -5                                                           
M -1 -2 -2 -4 -2  0 -2 -3 -1  2  3 -2  7  0 -3 -2 -1 -1  0  1 -3  2 -1 -1 -5                                                           
F -3 -3 -4 -5 -2 -4 -3 -4 -1  0  1 -4  0  8 -4 -3 -2  1  4 -1 -4  1 -4 -1 -5                                                           
P -1 -3 -2 -1 -4 -1 -1 -2 -2 -3 -4 -1 -3 -4 10 -1 -1 -4 -3 -3 -2 -3 -1 -1 -5                                                           
S  1 -1  1  0 -1  0 -1  0 -1 -3 -3  0 -2 -3 -1  5  2 -4 -2 -2  0 -3  0 -1 -5                                                           
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  2  5 -3 -2  0  0 -1 -1 -1 -5                                                           
W -3 -3 -4 -5 -5 -1 -3 -3 -3 -3 -2 -3 -1  1 -4 -4 -3 15  2 -3 -5 -2 -2 -1 -5                                                           
Y -2 -1 -2 -3 -3 -1 -2 -3  2 -1 -1 -2  0  4 -3 -2 -2  2  8 -1 -3 -1 -2 -1 -5                                                           
V  0 -3 -3 -4 -1 -3 -3 -4 -4  4  1 -3  1 -1 -3 -2  0 -3 -1  5 -3  2 -3 -1 -5                                                           
B -2 -1  5  6 -3  0  1 -1  0 -4 -4  0 -3 -4 -2  0  0 -5 -3 -3  6 -4  1 -1 -5                                                           
J -2 -3 -4 -4 -2 -3 -3 -4 -3  4  4 -3  2  1 -3 -3 -1 -2 -1  2 -4  4 -3 -1 -5                                                           
Z -1  0  0  1 -3  4  5 -2  0 -3 -3  1 -1 -4 -1  0 -1 -2 -2 -3  1 -3  5 -1 -5                                                           
X -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -5                                                           
* -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5  1                                                           
"""), sep='\s+').loc[list(aa_dict_one_hot.keys()), list(aa_dict_one_hot.keys())]
assert (BLOSUM50_MATRIX == BLOSUM50_MATRIX.T).all().all()

ENCODING_DATA_FRAMES = {
    "BLOSUM50": BLOSUM50_MATRIX,
    "one-hot": pd.DataFrame([
        [1 if i == j else 0 for i in range(len(aa_dict_one_hot.keys()))]
        for j in range(len(aa_dict_one_hot.keys()))
    ], index=aa_dict_one_hot.keys(), columns=aa_dict_one_hot.keys())
}

########################### HLA pseudo-sequence ##########################
#pMHCpan 
HLA_ABC=[os.path.join(hla_db_dir,'A_prot.fasta'), 
         os.path.join(hla_db_dir,'B_prot.fasta'),
         os.path.join(hla_db_dir,'C_prot.fasta'), 
         os.path.join(hla_db_dir,'E_prot.fasta')]
HLA_seq_lib={}
for one_class in HLA_ABC:
    prot=open(one_class)
    #pseudo_seq from netMHCpan:https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000796
    pseudo_seq_pos=[7,9,24,45,59,62,63,66,67,79,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171]
    #write HLA sequences into a library
    #class I alles
    name=''
    sequence=''                                                                                                                        
    for line in prot:
        if len(name)!=0:
            if line.startswith('>HLA'):
                pseudo=''
                for i in range(0,33):
                    if len(sequence)>pseudo_seq_pos[i]:
                        pseudo=pseudo+sequence[pseudo_seq_pos[i]]
                HLA_seq_lib[name]=pseudo
                name=line.split(' ')[1]
                sequence=''
            else:
                sequence=sequence+line.strip()
        else:
            name=line.split(' ')[1]
########################################
# Input data encoding helper functions #
########################################
#################functions for TCR encoding####################
def preprocess(filedir):
    #Preprocess TCR files                                                                                                                 
    print('Processing: '+filedir)
    if not os.path.exists(filedir):
        print('Invalid file path: ' + filedir)
        return 0
    dataset = pd.read_csv(filedir, header=0)
    #Preprocess HLA_antigen files
    #remove HLA which is not in HLA_seq_lib; if the input hla allele is not in HLA_seq_lib; then the first HLA startswith the input HLA allele will be given     
    #Remove antigen that is longer than 15aa
    dataset=dataset.dropna()
    HLA_list=list(dataset['HLA'])
    ind=0
    index_list=[]
    for i in HLA_list:
        if len([hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(i))])==0:
            index_list.append(ind)
            print('drop '+i)
        ind=ind+1
    dataset=dataset.drop(dataset.iloc[index_list].index)
    dataset=dataset[dataset.Antigen.str.len()<16]
    print(str(max(dataset.index)-dataset.shape[0]+1)+' antigens longer than 15aa are dropped!')
    TCR_list=dataset['CDR3'].tolist()
    antigen_list=dataset['Antigen'].tolist()
    HLA_list=dataset['HLA'].tolist()
    return TCR_list,antigen_list,HLA_list

def aamapping_TCR(peptideSeq,aa_dict, mutInd=None, mutFactor=None):
    #Transform aa seqs to Atchley's factors.                                                                                              
    peptideArray = []
    if len(peptideSeq)>80:
        print('Length: '+str(len(peptideSeq))+' over bound!')
        peptideSeq=peptideSeq[0:80]
    if mutFactor:
        i = 0
        while i < len(peptideSeq):
            try:
                if i == mutInd:
                    peptideArray.append([x * mutFactor for x in aa_dict[peptideSeq[i]]])
                else:
                    peptideArray.append(aa_dict[peptideSeq[i]])
            except KeyError:
                print('Not proper aaSeqs: '+peptideSeq)
                peptideArray.append(np.zeros(5,dtype='float64'))
            i += 1
    else:
        for aa_single in peptideSeq:
            try:
                peptideArray.append(aa_dict[aa_single])
            except KeyError:
                print('Not proper aaSeqs: '+peptideSeq)
                print(f'invalid aa {aa_single}; accepted aa:')
                for key in aa_dict:
                    print(key)
                peptideArray.append(np.zeros(5,dtype='float64'))
    for i in range(0,80-len(peptideSeq)):
        peptideArray.append(np.zeros(5,dtype='float64'))
    return np.asarray(peptideArray)

def hla_encode(HLA_name,encoding_method):
    #Convert the a HLA allele to a zero-padded numeric representation.
    if HLA_name not in HLA_seq_lib.keys():
        if len([hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(HLA_name))])==0:
            print('cannot find'+HLA_name)
        HLA_name=[hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(HLA_name))][0]
    if HLA_name not in HLA_seq_lib.keys():
        print('Not proper HLA allele:'+HLA_name)
    HLA_sequence=HLA_seq_lib[HLA_name]
    HLA_int=[aa_dict_one_hot[char] for char in HLA_sequence]
    while len(HLA_int)!=34:
        #if the pseudo sequence length is not 34, use X for padding
        HLA_int.append(20)
    result=ENCODING_DATA_FRAMES[encoding_method].iloc[HLA_int]
    # Get a numpy array of 34 rows and 21 columns
    return np.asarray(result)

def peptide_encode_HLA(peptide, maxlen,encoding_method, mutIndex=None, mutFactor=None):
    #Convert peptide amino acid sequence to numeric encoding
    if len(peptide) > maxlen:
        msg = 'Peptide %s has length %d > maxlen = %d.'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    peptide= peptide.replace(u'\xa0', u'')    #remove non-breaking space  
    o = list(map(lambda x: aa_dict_one_hot[x.upper()] if x.upper() in aa_dict_one_hot.keys() else 20 , peptide)) 
    #if the amino acid is not valid, replace it with padding aa 'X':20       
    k = len(o)
    #use 'X'(20) for padding
    o = o[:k // 2] + [20] * (int(maxlen) - k) + o[k // 2:]
    if len(o) != maxlen:
        msg = 'Peptide %s has length %d < maxlen = %d, but pad is "none".'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    result=ENCODING_DATA_FRAMES[encoding_method].iloc[o]
    if mutFactor and (mutIndex != 'n/a'):
        indList = []
        paddedMutIndex = -1
        i = 0
        while i < k:
            indList.append(i)
            i += 1
        paddedIndList = indList[:k // 2] + [90] * (int(maxlen) - k) + indList[k // 2:]
        i = 0
        while i < maxlen:
            if paddedIndList[i] == mutIndex:
                paddedMutIndex = i
                break
            else:
                i += 1
        if paddedMutIndex == -1:
            raise UnboundLocalError(f'***********mutation index {mutIndex} not found mutFactor {mutFactor}******\n\n')
        resultF = np.asarray(result, dtype='float64')
        resultF[paddedMutIndex] *= mutFactor
    else:
        resultF = np.asarray(result)
    return resultF

def TCRMap(dataset,aa_dict, mutInds=None, mutFactor=None):
    #Wrapper of aamapping    
    if mutInds:
        dataset = [tuple(x) for x in zip(dataset, mutInds)]
        for i in range(0,len(dataset)):
            if i==0:
                TCR_array=aamapping_TCR(dataset[i][0], aa_dict, mutInd=dataset[i][1], mutFactor=mutFactor).reshape(1,80,5,1)
            else:
                TCR_array=np.append(TCR_array,aamapping_TCR(dataset[i][0], aa_dict, mutInd=dataset[i][1], mutFactor=mutFactor).reshape(1,80,5,1),axis=0)
    else:                                                                                                         
        for i in range(0,len(dataset)):
            if i==0:
                TCR_array=aamapping_TCR(dataset[i],aa_dict).reshape(1,80,5,1)
            else:
                TCR_array=np.append(TCR_array,aamapping_TCR(dataset[i],aa_dict).reshape(1,80,5,1),axis=0)
    print('TCRMap done!')
    return TCR_array

def HLAMap(dataset,encoding_method):
    #Input a list of HLA and get a three dimentional array
    m=0
    for each_HLA in dataset:
        if m==0:
            HLA_array=hla_encode(each_HLA,encoding_method).reshape(1,34,21)
        else:
            HLA_array=np.append(HLA_array,hla_encode(each_HLA,encoding_method).reshape(1,34,21),axis=0)
        m=m+1
    print('HLAMap done!')
    return HLA_array

def antigenMap(dataset,maxlen,encoding_method, mutIndList=None, mutFactor=None):
    #Input a list of antigens and get a three dimentional array  
    m=0
    if mutFactor:
        dataset = [tuple(x) for x in zip(dataset, mutIndList)]
        for each_antigen in dataset:
            if m==0:
                antigen_array=peptide_encode_HLA(each_antigen[0],maxlen,encoding_method, each_antigen[1], mutFactor).reshape(1,maxlen,21)
            else:
                antigen_array=np.append(antigen_array,peptide_encode_HLA(each_antigen[0],maxlen,encoding_method, each_antigen[1], mutFactor).reshape(1,maxlen,21),axis=0)
            m=m+1
    else:
        for each_antigen in dataset:
            if m==0:
                antigen_array=peptide_encode_HLA(each_antigen,maxlen,encoding_method).reshape(1,maxlen,21)
            else:
                antigen_array=np.append(antigen_array,peptide_encode_HLA(each_antigen, maxlen,encoding_method).reshape(1,maxlen,21),axis=0)
            m=m+1
    print('antigenMap done!')
    return antigen_array

def pearson_correlation_f(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred) #being K.mean a scalar here, it will be automatically subtracted from all elements in y_pred                
    fst = y_true - K.mean(y_true)
    devP = K.std(y_pred)
    devT = K.std(y_true)
    return K.mean(fsp*fst)/(devP*devT)

def pos_neg_acc(y_true,y_pred):
    #self-defined prediction accuracy metric
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.cast(negative_pred<positive_pred,"float16"))
    return diff

def pos_neg_loss(y_true,y_pred):
    #self-defined prediction loss function 
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.relu(1+negative_pred-positive_pred))+0.2*K.mean(K.square(negative_pred)+K.square(positive_pred))
    return diff

#########################################
#   added functions for mutation simulation
def mutSeq(seq, alaScan, xScan, gentleMut):
    '''
    simulate mutations in a protein sequence: 
        alanine scan, X scan, or "gentle" mutation scan
    '''
    mutList = [seq]
    mutIndexList = ['n/a']
    mutIndex = 0
    nMutants = len(seq)
    while mutIndex < nMutants:
        if gentleMut:
            mutList.append(seq)
        elif alaScan:
            mutSeqi = seq[:mutIndex]
            mutSeqi += 'A'
            mutSeqi += seq[(mutIndex+1):]
            mutList.append(mutSeqi)
        elif xScan:
            mutSeqi = seq[:mutIndex]
            mutSeqi += 'X'
            mutSeqi += seq[(mutIndex+1):]
            mutList.append(mutSeqi)
        mutIndexList.append(mutIndex)
        mutIndex += 1
    return mutList, mutIndexList
    
    

#########################################                                                                                                      
# preprocess input data and do encoding #                                                                                                      
#########################################
#Read data
#TCR Data preprocess                                                                                                                      
log_file=open(output_log_dir,'a')
sys.stdout=log_file
print('Mission loading.')

TCR_list,antigen_list,HLA_list=preprocess(file_dir)
mutTCRList = []
mutAntigenList = []
mutHLAList = []
mutIndList = []
lenList = len(TCR_list)
if mutAntigen:
    i = 0
    while i < lenList:
        antigen = antigen_list[i]
        TCR = TCR_list[i]
        HLA = HLA_list[i]
        mutAntigenBlock, mutIndexBlock = mutSeq(antigen, alaScan, xScan, gentleMut)
        mutAntigenList += mutAntigenBlock
        mutTCRList += [TCR for x in mutAntigenBlock]
        mutHLAList += [HLA for x in mutAntigenBlock]
        mutIndList += mutIndexBlock
        i += 1
if mutCDR:
    i = 0
    while i < lenList:
        antigen = antigen_list[i]
        TCR = TCR_list[i]
        HLA = HLA_list[i]
        mutTCRblock, mutIndexBlock = mutSeq(TCR, alaScan, xScan, gentleMut)
        mutTCRList += mutTCRblock
        mutAntigenList += [antigen for x in mutTCRblock]
        mutHLAList += [HLA for x in mutTCRblock]
        mutIndList += mutIndexBlock
        i += 1
TCR_list = mutTCRList
antigen_list = mutAntigenList
HLA_list = mutHLAList
if gentleMut:
    if mutCDR:
        TCR_array=TCRMap(TCR_list,aa_dict_atchley, mutIndList, mutFactor)
        antigen_array=antigenMap(antigen_list,15,'BLOSUM50')
    elif mutAntigen:
        TCR_array=TCRMap(TCR_list,aa_dict_atchley)
        antigen_array=antigenMap(antigen_list,15,'BLOSUM50', mutIndList, mutFactor)
    HLA_array=HLAMap(HLA_list,'BLOSUM50')
else:
    TCR_array=TCRMap(TCR_list,aa_dict_atchley)
    antigen_array=antigenMap(antigen_list,15,'BLOSUM50')
    HLA_array=HLAMap(HLA_list,'BLOSUM50')

#Model prediction                                                                                                                         
TCR_encoder=load_model(os.path.join(model_dir,'TCR_encoder_30.h5'))
TCR_encoder=Model(TCR_encoder.input,TCR_encoder.layers[-12].output)
TCR_encoded_result=TCR_encoder.predict(TCR_array)

HLA_antigen_encoder=load_model(model_dir+'/HLA_antigen_encoder_60.h5',custom_objects={'pearson_correlation_f': pearson_correlation_f})
HLA_antigen_encoder=Model(HLA_antigen_encoder.input,HLA_antigen_encoder.layers[-2].output)
HLA_antigen_encoded_result=HLA_antigen_encoder.predict([antigen_array,HLA_array])

TCR_encoded_matrix=pd.DataFrame(data=TCR_encoded_result,index=range(1,len(TCR_list)+1))
HLA_antigen_encoded_matrix=pd.DataFrame(data=HLA_antigen_encoded_result,index=range(1,len(HLA_list)+1))
allele_matrix=pd.DataFrame({'CDR3':TCR_list,'Antigen':antigen_list,'HLA':HLA_list},index=range(1,len(TCR_list)+1))
TCR_encoded_matrix.to_csv(os.path.join(output_dir, f'{outputFn}_TCR_output.csv'),sep=',')
HLA_antigen_encoded_matrix.to_csv(os.path.join(output_dir, f'{outputFn}_MHC_antigen_output.csv'),sep=',')
print('Encoding Accomplished.\n')
#########################################                                                                                                                       
# make prediction based on encoding     #                                                                                                                     
#########################################   
############## Load Prediction Model ################                                                                                                           
#set up model                                                                                                                                             
hla_antigen_in=Input(shape=(60,),name='hla_antigen_in')
pos_in=Input(shape=(30,),name='pos_in')
ternary_layer1_pos=concatenate([pos_in,hla_antigen_in])
ternary_dense1=Dense(300,activation='relu')(ternary_layer1_pos)
ternary_do1=Dropout(0.2)(ternary_dense1)
ternary_dense2=Dense(200,activation='relu')(ternary_do1)
ternary_dense3=Dense(100,activation='relu')(ternary_dense2)
ternary_output=Dense(1,activation='linear')(ternary_dense3)
ternary_prediction=Model(inputs=[pos_in,hla_antigen_in],outputs=ternary_output)
#load weights                                                                                                                                                    
ternary_prediction.load_weights(model_dir+'/weights.h5')
################ read dataset #################                                                                                                                  
#read background negative TCRs
TCR_neg_df_1k=pd.read_csv(os.path.join(library_dir, 'bg_tcr_library', 'TCR_output_1k.csv'),index_col=0)
TCR_neg_df_10k=pd.read_csv(os.path.join(library_dir, 'bg_tcr_library','TCR_output_10k.csv'),index_col=0)
TCR_pos_df=pd.read_csv(os.path.join(output_dir, f'{outputFn}_TCR_output.csv'),index_col=0)
MHC_antigen_df=pd.read_csv(os.path.join(output_dir, f'{outputFn}_MHC_antigen_output.csv'),index_col=0)
################ make prediction ################# 
rank_output=[]
for each_data_index in range(TCR_pos_df.shape[0]):
    tcr_pos=TCR_pos_df.iloc[[each_data_index,]]
    pmhc=MHC_antigen_df.iloc[[each_data_index,]]
    #used the positive pair with 1k negative tcr to form a 1001 data frame for prediction                                                                      

    TCR_input_df=pd.concat([tcr_pos,TCR_neg_df_1k],axis=0)
    MHC_antigen_input_df= pd.DataFrame(np.repeat(pmhc.values,1001,axis=0))
    prediction=ternary_prediction.predict({'pos_in':TCR_input_df,'hla_antigen_in':MHC_antigen_input_df})

    rank=1-(sorted(prediction.tolist()).index(prediction.tolist()[0])+1)/1000
    #if rank is higher than top 2% use 10k background TCR                                                                                                         
    if rank<0.02:
        TCR_input_df=pd.concat([tcr_pos,TCR_neg_df_10k],axis=0)
        MHC_antigen_input_df= pd.DataFrame(np.repeat(pmhc.values,10001,axis=0))
        prediction=ternary_prediction.predict({'pos_in':TCR_input_df,'hla_antigen_in':MHC_antigen_input_df})

        rank=1-(sorted(prediction.tolist()).index(prediction.tolist()[0])+1)/10000
    rank_output.append(rank)

#label gentle mutations as lowercase
if gentleMut:
    n = len(HLA_list)
    if mutAntigen:
        newAntList = []
        i = 0
        while i < n:
            ant = antigen_list[i]
            mutInd = mutIndList[i]
            if mutInd == 'n/a':
                newAnt = ant
            else:
                newAnt = ant[:mutInd]
                newAnt += ant[mutInd].lower()
                newAnt += ant[(mutInd + 1):]
            newAntList.append(newAnt)
            i += 1
        antigen_list = newAntList
    elif mutCDR:
        newCDRlist = []
        i = 0
        while i < n:
            TCR = TCR_list[i]
            mutInd = mutIndList[i]
            if mutInd == 'n/a':
                newTCR = TCR
            else:
                newTCR = TCR[:mutInd]
                newTCR += TCR[mutInd].lower()
                newTCR += TCR[(mutInd + 1):]
            newCDRlist.append(newTCR)
            i += 1
        TCR_list = newCDRlist
#output
rank_output_matrix=pd.DataFrame({'CDR3':TCR_list,'Antigen':antigen_list,'HLA':HLA_list,'Rank':rank_output, 'MutationIndex': mutIndList},index=range(1,len(TCR_list)+1))
#rank_output_matrix.to_csv(output_dir+'/prediction.csv',sep=',')
rank_output_matrix.to_csv(outputFilePath,sep=',')
print('Prediction Accomplished.\n')
log_file.close()
#delete encoding files
os.remove(os.path.join(output_dir,f'{outputFn}_MHC_antigen_output.csv'))
os.remove(os.path.join(output_dir,f'{outputFn}_TCR_output.csv'))
