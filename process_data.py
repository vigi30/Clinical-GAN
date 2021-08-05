import sys
import pickle
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from collections import Counter
import argparse
parser = argparse.ArgumentParser()

def get_ICDs_from_mimic_file(fileName, hadmToMap,isdiagnosis=True):
    mimicFile = open(fileName, 'r')  # row_id,subject_id,hadm_id,seq_num,ICD9_code
    mimicFile.readline()
    codes = []
    
    number_of_null_ICD9_codes = 0
    number_of_null_ICD10_codes = 0
    
    for line in mimicFile: #   0  ,     1    ,    2   ,   3  ,    4
        tokens = line.strip().split(',')
        #print(tokens)
        hadm_id = int(tokens[1])
        if (len(tokens[3]) == 0):
            if isdiagnosis:
                if (tokens[4] =='9'):
                    # ignore diagnoses where ICD9_code is null
                    number_of_null_ICD9_codes += 1
                else:
                    number_of_null_ICD10_codes += 1

                continue;
            else:
                if (tokens[5] =='9'):
                    # ignore diagnoses where ICD9_code is null
                    number_of_null_ICD9_codes += 1
                else:
                    number_of_null_ICD10_codes += 1

                continue;
                
        if isdiagnosis:
            ICD_code = tokens[3]
        else:
            ICD_code = tokens[4] 
            
            
        if ICD_code.find("\"") != -1:
            #print("ICD_Code before",ICD_code )
            ICD_code = ICD_code[1:-1].strip()  # toss off quotes and proceed
            #print("ICD_Code after",ICD_code )
        # since diagnosis and procedure ICD9 codes have intersections, a prefix is necessary for disambiguation
       
        if isdiagnosis:
            ICD_code = 'D'+tokens[4]+'_'+ICD_code
        else:
            ICD_code = 'P'+tokens[5]+'_'+ICD_code

        # To understand the line below, check https://mimic.physionet.org/mimictables/diagnoses_icd/
        # "The code field for the ICD-9-CM Principal and Other Diagnosis Codes is six characters in length (not really!),
        # with the decimal point implied between the third and fourth digit for all diagnosis codes other than the V codes.
        # The decimal is implied for V codes between the second and third digit."
        # Actually, if you look at the codes (https://raw.githubusercontent.com/drobbins/ICD9/master/ICD9.txt), simply take the three first characters
        #if not map_ICD9_to_CCS:
          #  ICD_code = ICD_code[:4]  # No CCS mapping, get the first alphanumeric four letters only


        if hadm_id in hadmToMap:
            hadmToMap[hadm_id].append(ICD_code.strip())
        else:
            hadmToMap[hadm_id]= [ICD_code.strip()]  
    mimicFile.close()
    print ('-Number of null ICD9 codes in file ' + fileName + ': ' + str(number_of_null_ICD9_codes))
    print ('-Number of null ICD10 codes in file ' + fileName + ': ' + str(number_of_null_ICD10_codes))
    #print ('-Number of diagnosis codes in file ' + fileName + ': ' + str(len(codes)))
    
    
    
def get_drugs_from_mimic_file(fileName, hadmToMap, choice ='ndc'):
    mimicFile = open(fileName, 'r',encoding="utf8")  # subject_id,hadm_id,gsn,ndc,drug
    mimicFile.readline()
    number_of_null_NDC_codes = 0
    drugDescription = {}
    try:
        for line in mimicFile:
            #print(line)#   0  ,     1    ,    2   ,   3  ,    4
            #break
            tokens = line.strip().split(',')
            #print(tokens)
            hadm_id = int(tokens[1])
            if choice =='ndc':                        #code : Total Number of NDC code 5912
                drug_code = tokens[3]   
            else:    
                drug_code = tokens[2]                    #code : Total Number of gsn code 3081

            drug_code = drug_code.strip()  

            drug_code = 'DR'+'_'+drug_code
            if hadm_id in hadmToMap:
                hadmToMap[hadm_id].append(drug_code.strip())
            else:
                #hadmToMap[hadm_id]=set()           #use set to avoid repetitions
                #hadmToMap[hadm_id].add(drug_code.strip())
                hadmToMap[hadm_id]=[drug_code.strip()]
                
            if drug_code not in drugDescription:
                drugDescription[drug_code] = tokens[4]
                
    except Exception as e:
        print(line)
        print(e)
    #for hadm_id in hadmToMap.keys():
        #hadmToMap[hadm_id] = list(hadmToMap[hadm_id])   #convert to list, as the rest of the codes expects
    mimicFile.close()
    return drugDescription





def ListAvgVisit(dic):
    a =[len(intList) for k,intList in dic.items()]
    return sum(a)/len(a)

def countCodes(dic):
    countCode = {}
    for codes in dic.values():
        for code in codes:
            
            if code  in countCode:
                countCode[code] = countCode[code] +1
            else:
                countCode[code] = 1
                
    return len(countCode)

def minMaxCodes(dic):
    countCode = []
    for codes in dic.values():
        countCode.append(len(codes))    
                
    return min(countCode),max(countCode)

def display(pidAdmMap,admDxMap,admPxMap,admDrugMap):
    print(f" Total Number of patients {len(pidAdmMap)}")
    print(f" Total Number of admissions {len(admDxMap)}")
    print(f" Average number of admissions per patient {ListAvgVisit(pidAdmMap)}")
    print(f" Total Number of diagnosis code {countCodes(admDxMap)}")
    print(f" Total Number of procedure code {countCodes(admPxMap)}")
    print(f" Total Number of drug code {countCodes(admDrugMap)}")
    print(f" Total Number of codes {countCodes(admPxMap) +countCodes(admDxMap)+countCodes(admDrugMap) }")
    print(f" average Number of procedure code per visit {ListAvgVisit(admPxMap)}")
    print(f" average Number of diagnosis code per visit {ListAvgVisit(admDxMap)}")
    print(f" average Number of Drug code per visit {ListAvgVisit(admDrugMap)}")

def load_mimic_data(mimic3_path,CCSRDX_file,CCSRPCS_file,choice ='ndc'):
    mimic3_path = 'data'
    CCSRDX_file = 'DXCCSR_v2021-2/DXCCSR_v2021-2.csv'
    CCSRPCS_file = 'PRCCSR_v2021-1/PRCCSR_v2021-1.csv'
    #os.path.join(mimic3_path, 'ADMISSIONS.csv')
    admissionFile = os.path.join(mimic3_path, 'ADMISSIONS.csv')
    diagnosisFile = os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv')
    procedureFile = os.path.join(mimic3_path, 'PROCEDURES_ICD.csv')
    patientsAge = os.path.join(mimic3_path, 'patientsAge.csv')
    prescriptionFile = os.path.join(mimic3_path, 'PRESCRIPTIONS.csv')
    diagnosisFrequencyFile = os.path.join(mimic3_path, 'WITHOUT_IF_CODE_COUNT.csv')
    #outFile = 'data'
    print ('Building pid-admission mapping, admission-date mapping')
    previous_subject = 0
    previous_admission = 0
    pidAdmMap = {}
    admDateMap = {}
    pidStatic = {}   # adm type, Insurance , ethnicity , marital status

    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0])
        admId = int(tokens[1])
        #admTime = datetime.strptime(tokens[2], '%Y-%m-%d %H:%M:%S')
        #admTime = tokens[3]
        #admDateMap[admId] = admTime

        #pidStatic[pid] = [convert_binary_to_Int(tokens)]
        if pid in pidAdmMap: 
            pidAdmMap[pid].add(admId)
        else: 
            pidAdmMap[pid] = set()
            pidAdmMap[pid].add(admId)
    for pid in pidAdmMap.keys():
        pidAdmMap[pid] = list(pidAdmMap[pid])  
    infd.close()

    print ('Building admission-diagnosis mapping')
    admDxMap = {}
    admPxMap = {}
    admDrugMap = {}
    get_ICDs_from_mimic_file(diagnosisFile, admDxMap)

    print ('Building admission-procedure mapping')
    get_ICDs_from_mimic_file(procedureFile, admPxMap,isdiagnosis=False)

    print ('Building admission-drug mapping')
    drugDescription = get_drugs_from_mimic_file(prescriptionFile,admDrugMap,choice)
    return pidAdmMap,admDxMap,admPxMap,admDrugMap,drugDescription



def clean_data(pidAdmMap,admDxMap,admPxMap,admDrugMap):
    # removing the subject_id which are not present in diagnostic code but present in procedure and vice versa
    print("Cleaning data...")
    subDelList = []

    print("Removing patient records who does not have all three medical codes for an admission")
    for subject_id,hadm_ids in  pidAdmMap.items():
        for hadm_id in hadm_ids:
            if (hadm_id not in admDxMap.keys()):
                subDelList.append(subject_id)
            if (hadm_id not in admPxMap.keys()):
                subDelList.append(subject_id)
            if (hadm_id not in admDrugMap.keys()):
                subDelList.append(subject_id)

    subDelList = list(set(subDelList))       
    #print(f"Number of subject_ids to be deleted :{len(subDelList)} ")

    for i in subDelList:
        del pidAdmMap[i]

    #print(f"Number of subject_ids aftr cleaning :{len(pidAdmMap)} ")  
            
    adDx,adPx,adDrug=updateAdmCodeList(pidAdmMap,admDxMap,admPxMap,admDrugMap)

    display(pidAdmMap,adDx,adPx,adDrug)
    # removing patient who made less than 2 admissions

    print("Removing patient who made less than 2 admissions")
    pidMap = {}
    adm = []
    subDelList=[]
    pidAdmMap1 = pidAdmMap
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 2:
            subDelList.append(pid)
            continue

    for i in subDelList:
        del pidAdmMap[i]  

    adDx,adPx,adDrug=updateAdmCodeList(pidAdmMap,adDx,adPx,adDrug)   
    display(pidAdmMap,adDx,adPx,adDrug)  
    return pidAdmMap,adDx,adPx,adDrug


def updateAdmCodeList(sub_AdmID,admDxMap,admPxMap,admDrugMap):
    adDx = {}
    adPx = {}
    adDrug={}
    for pid, admIdList in sub_AdmID.items():
        for admId in admIdList:

            adDx[admId] = admDxMap[admId]
            adPx[admId] =admPxMap[admId]
            #adAge[admId] = admAgeMap[admId]
            adDrug[admId] =admDrugMap[admId]
            
    return adDx,adPx,adDrug


def map_ICD9_to_CCSR(mape):
    icdTOCCS_Map = pickle.load(open('icd10DP_to_cssr_dictionary','rb'))
    procCODEstoInternalID_map = {}
    missingCodes = []
    set_of_used_codes = set()
    number_of_codes_missing = 0
    countICD9=0
    countICD10 =0
    for (hadm_id, ICDs_List) in mape.items():
        for ICD in ICDs_List:
            #print(ICD,type(ICD),len(ICD))
            #while (len(ICD9) < 6): ICD9 += ' '  #pad right white spaces because the CCS mapping uses this pattern
            if ICD.startswith('D10_'):
                padStr = 'D10_'
            elif ICD.startswith('D9_'):
                padStr = 'D9_'
            elif ICD.startswith('P10_'):
                padStr = 'P10_'    
            elif ICD.startswith('P9_'):
                padStr = 'P9_'  
            else:
                print("Wrong coding format")

            try:

                CCS_code = icdTOCCS_Map[ICD]

                if hadm_id in procCODEstoInternalID_map:
                    if(isinstance(CCS_code, str)): 
                        procCODEstoInternalID_map[hadm_id].append(CCS_code)
                    else:
                        for code in CCS_code:
                            procCODEstoInternalID_map[hadm_id].append(code)
                        
                else:
                    if(isinstance(CCS_code, str)): 
                        procCODEstoInternalID_map[hadm_id] = [CCS_code]
                    else:
                        for i in range(len(CCS_code)):
                            if i==0:
                                procCODEstoInternalID_map[hadm_id] = [CCS_code[i]]
                            else:
                                procCODEstoInternalID_map[hadm_id].append(CCS_code[i])
                                
                            
                set_of_used_codes.add(ICD)

            except KeyError:
                print(f"the mapping of {ICD} {hadm_id}")
                missingCodes.append(ICD)
                #print(f"the mapping of  is : {icdTOCCS_Map[ICD]}")
                number_of_codes_missing +=1
                #print (str(sys.exc_info()[0]) + '  ' + str(ICD) + ". ICD9 code not found, please verify your ICD9 to CCS mapping before proceeding.")


            
    print(f"total number of ICD9 codes used {countICD9} and ICD10 codes: {countICD10}")  
    print ('-Total number (complete set) of ICD9+ICD10 codes (diag + proc): ' + str(len(set(icdTOCCS_Map.keys()))))
    #print ('-Total number (complete set) of CCS codes (diag + proc): ' + str(len(set(icd9TOCCS_Map.values()))))
    print ('-Total number of ICD codes actually used: ' + str(len(set_of_used_codes)))
    print ('-Total number of ICD codes missing in the admissions list: ' , number_of_codes_missing)
    #print(icd9TOCCS_Map)
    
    return procCODEstoInternalID_map,missingCodes,set_of_used_codes


def map_ccsr_description(filename,type='Diag'):
    if type == 'Diag':
        padStr = 'D10_'
    else:
        padStr = 'P10_'
    df = pd.read_excel(filename, sheet_name="CCSR_Categories",skiprows=1)
    if type!='Diag':
        df = df[:-1]
    codeDescription = df[["CCSR Category", "CCSR Category Description"]]
    codeDescription = codeDescription.applymap(lambda x: padStr+str(x))
    codeDescription = codeDescription.set_index("CCSR Category").T.to_dict('list')
    for key,value in codeDescription.items():
        newValue = value[0][4:]
        codeDescription[key] = newValue

    return codeDescription

def convValuestoList(codeDic):
    for key, value in codeDic.items():
        codeDic[key] =  [value]
    return codeDic


def create_CCS_CCSR_mapping(CCSRDX_file,CCSRPCS_file,CCSDX_file,CCSPX_file):
    df = pd.read_csv(CCSRDX_file)
    a = df[["\'ICD-10-CM CODE\'", "\'CCSR CATEGORY 1\'", "\'CCSR CATEGORY 2\'", "\'CCSR CATEGORY 3\'", "\'CCSR CATEGORY 4\'", "\'CCSR CATEGORY 5\'", "\'CCSR CATEGORY 6\'"]]

    a = a.applymap(lambda x: str(x)[1:-1])

    a = a.set_index("\'ICD-10-CM CODE\'").T.to_dict('list')
    # remove null values
    for key, value in a.items():
        newValue = []
        value = list(filter(lambda x: x.strip(),value))
        for value in value:
            newValue.append('D10_'+value)
        a[key] =  newValue
        
    b={}
    for key in a.keys():
        new_key = 'D10_'+key 
        b[new_key] = a[key]

    df = pd.read_csv(CCSRPCS_file)
    df = df[["\'ICD-10-PCS\'", "\'PRCCSR\'"]]
    df = df.applymap(lambda x: str(x)[1:-1])
    df = df.set_index("\'ICD-10-PCS\'").T.to_dict('list')

    for key, value in df.items():
        newValue = []
        value = list(filter(lambda x: x.strip(), value))
        for value in value:
            newValue.append('P10_'+value)
        df[key] =  newValue
        
    for key in df.keys():
        new_key = 'P10_'+key 
        b[new_key] = df[key]
        
    # ICD -9 diagnosis code and prescription to CCS
    ccsTOdescription_Map = {}
    #'ICD-9-CM CODE','CCS CATEGORY','CCS CATEGORY DESCRIPTION','ICD-9-CM CODE DESCRIPTION','OPTIONAL CCS CATEGORY','OPTIONAL CCS CATEGORY DESCRIPTION'
    #dxref_ccs_file = open('Single_Level_CCS_2015/$dxref 2015.csv', 'r')
    dxref_ccs_file = open(CCSDX_file, 'r')
    dxref_ccs_file.readline() #note
    dxref_ccs_file.readline() #header
    dxref_ccs_file.readline() #null
    for line in dxref_ccs_file:
        tokens = line.strip().split(',')
        # since diagnosis and procedure ICD9 codes have intersections, a prefix is necessary for disambiguation
        b['D9_'+str(tokens[0][1:-1]).strip()] = 'D9_'+str(tokens[1][1:-1]).strip() #[1:-1] retira aspas
        ccsTOdescription_Map['D9_'+str(tokens[1][1:-1]).strip()] = str(tokens[2][1:-1]).strip() #[1:-1] retira aspas
    dxref_ccs_file.close()

    dxprref_ccs_file = open(CCSPX_file, 'r')
    dxprref_ccs_file.readline() #note
    dxprref_ccs_file.readline() #header
    dxprref_ccs_file.readline() #null
    for line in dxprref_ccs_file:
        tokens = line.strip().split(',')
        #since diagnosis and procedure ICD9 codes have intersections, a prefix is necessary for disambiguation
        b['P9_'+str(tokens[0][1:-1]).strip()] = 'P9_'+str(tokens[1][1:-1]).strip() #[1:-1] retira aspas
        ccsTOdescription_Map['P9_'+str(tokens[1][1:-1]).strip()] = str(tokens[2][1:-1]).strip() #[1:-1] retira aspas
    dxprref_ccs_file.close()

    pickle.dump(b, open('icd10DP_to_cssr_dictionary', 'wb'), -1)
    pickle.dump(ccsTOdescription_Map, open('ccs_to_description_dictionary', 'wb'), -1)
    print ('Total ICD to ccs entries: ' + str(len(b)))
    print( 'Total ccs codes/descriptions: ' + str(len(ccsTOdescription_Map)))

    v1= []
    for v in b.values():
        for val in v:
            
            v1.append(val)
    v1 = list(set(v1))
    print("total number of unqiue codes(DIag + proc):", len(v1))

    return ccsTOdescription_Map


def icd_mapping(CCSRDX_file,CCSRPCS_file,CCSDX_file,CCSPX_file,D_CCSR_Ref_file,P_CCSR_Ref_file,adDx,adPx,adDrug,drugDescription):
    # creating mappint between all ICD codes to CCS and CCSR mapping
    ccsTOdescription_Map = create_CCS_CCSR_mapping(CCSRDX_file,CCSRPCS_file,CCSDX_file,CCSPX_file)
    # getting the description of all codes
    DxcodeDescription = map_ccsr_description(D_CCSR_Ref_file)
    PxcodeDescription = map_ccsr_description(P_CCSR_Ref_file,type='Proc')
    codeDescription ={**DxcodeDescription ,**PxcodeDescription }
    codeDescription ={**codeDescription ,**convValuestoList(ccsTOdescription_Map),**drugDescription}
    # mapping diagnois codes
    adDx,missingDxCodes,set_of_used_codes1 = map_ICD9_to_CCSR(adDx)
    # mapping procedure codes
    adPx,missingPxCodes,set_of_used_codes2 = map_ICD9_to_CCSR(adPx)
    codeDescription['SOH'] = 'Start of history'
    codeDescription['EOH'] = 'End of history'
    codeDescription['EOV'] = 'End of visit'
    displayCodeStats(adDx,adPx,adDrug)
    return adDx,adPx,codeDescription


def countCodes2(dic1,dic2):
    a = list(dic1.values())
    b = list(dic2.values())
    c= a+b
    countCode = {}
    for code in c:
        for code in code:
            if code  in countCode:
                countCode[code] = countCode[code] +1
            else:
                countCode[code] = 1
                
    return len(countCode)

def countCodes1(dic1,dic2,dic3):
    a = list(dic1.values())
    b = list(dic2.values())
    c = list(dic3.values())
    d= a+b+c
    countCode = {}
    for code in d:
        for code in code:
            if code  in countCode:
                countCode[code] = countCode[code] +1
            else:
                countCode[code] = 1
                
    return len(countCode)

def displayCodeStats(adDx,adPx,adDrug):
    print(f" Total Number of diagnosis code {countCodes(adDx)}")
    print(f" Total Number of procedure code {countCodes(adPx)}")
    print(f" Total Number of drug code {countCodes(adDrug)}")
    print(f" Total Number of unqiue D,P codes {countCodes2(adDx,adPx) }")
    print(f" Total Number of all codes {countCodes1(adDx,adPx,adDrug) }")


    print(f" average Number of procedure code per visit {ListAvgVisit(adPx)}")
    print(f" average Number of diagnosis code per visit {ListAvgVisit(adDx)}")
    print(f" average Number of drug code per visit {ListAvgVisit(adDrug)}")

    print(f" Min. and max. Number of diagnosis code  {minMaxCodes(adDx)}")
    print(f" Min. and max. Number of procedure code  {minMaxCodes(adPx)}")
    print(f" Min. and max. Number of drug code  {minMaxCodes(adDrug)}")

def trim(min_dx,min_px,min_drg,adDx,adPx,adDrug):
    #min_dx = 80
    #min_px = 80
    #min_drg = 80
    print("Trimmming the diagnosis,procedure and medication code for each visit")
    adDx1 = {}
    for codes in adDx.items():
        adDx1[codes[0]] = codes[1][:min_dx]
        
    adPx1 = {}
    for codes in adPx.items():
        adPx1[codes[0]] = codes[1][:min_px]
        
    adDrug1 = {}
    for codes in adDrug.items():
        adDrug1[codes[0]] = codes[1][:min_drg]
        
        
    adDx=adDx1
    adPx=adPx1
    adDrug = adDrug1

    displayCodeStats(adDx,adPx,adDrug)
    return adDx,adPx,adDrug

def buildData(idList,adDx,adPx,adDrug):
    
    print ('Building admission-Visits mapping')
    pidSeqMap = {}
    for pid, admIdList in idList.items():
        if len(admIdList) < 2: continue

        sortedList = [( adDx[admId], adPx[admId],adDrug[admId]) for admId in admIdList]
        pidSeqMap[pid] = sortedList



    print ('Building subject-id, diagnosis,procedure,drugs mapping')
    subject_ids = []
    dates = []
    seqs =[]
    ages = []
    for subject_id,visits in pidSeqMap.items():
        subject_ids.append(subject_id)
        diagnose = []
        procedure = []
        drugs = []
        date = []
        seq=[]
        age = []
        for visit in visits:
            #date.append(visit[0])
            #age.append(visit[4])
            #joined = [visit[4]] + visit[1] +visit[2]+visit[3]
            joined =  list(dict.fromkeys(list(dict.fromkeys(visit[0])) +list(dict.fromkeys(visit[1])))) + list(dict.fromkeys(visit[2]))
            seq.append(joined)

        #dates.append(date)
        seqs.append(seq)
        #ages.append(age)

    print ('Converting Strings Codes into unique integer, and making types')
    #types = {"PAD":0 , "SOH":1,  "EOV":2,  "EOH":3}
    types={}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        #print("patient",patient)
        for visit in patient:
            #print("vsit",visit)
            newVisit = []
            for code in visit:
                #print("code",code)
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)+4
                    newVisit.append(types[code])
                    #print("newVisit",newVisit)
            newPatient.append(newVisit)
        newSeqs.append(newPatient)
    return newSeqs,types

def ListAvgVisitForRemoveCode(dic):
    a =[len(intList) for intList in dic]
    return sum(a)/len(a)

def removeCode(newSeqs,types,threshold=5):
    currentSeqs = []
    print(ListAvgVisitForRemoveCode(newSeqs))
    countCode = {}
    for visits in newSeqs:
        for visit in visits:
            for code in visit:
                if code in countCode:
                    countCode[code] = countCode[code] +1
                else:
                    countCode[code] = 1
                    
    #x2 = list(countCode.keys())      
    #y2 = list(countCode.values())  
    a  = [(value,key) for (key,value) in countCode.items()]

    codes = []
    for i in a:
        if i[0] <=threshold:
            codes.append(i[1])
    print(f" Total number of codes removed: {len(codes)}  ")
    print(f" Total number of  unique codes : {len(a)}  ")
    for visits in newSeqs:
        newPatient = []
        for visit in visits:
            newVisit = []
            for code in visit:
                if code not in codes:
                    newVisit.append(code)
            newPatient.append(newVisit)
        currentSeqs.append(newPatient) 


    countCode = {}
    for visits in currentSeqs:
        for visit in visits:
            for code in visit:
                if code in countCode:
                    countCode[code] = countCode[code] +1
                else:
                    countCode[code] = 1
                    
    #x2 = list(countCode.keys())      
    #y2 = list(countCode.values())  
    a  = [(value,key) for (key,value) in countCode.items()]

    print(f" Total number of  unique codes after cleaning : {len(a)}  ")

    ## mapping from integer to codes
    reverseTypes = {v:k for k,v in types.items()}
    newSeqs = []
    for visits in currentSeqs:
        newPatient = []
        #print("patient",patient)
        for visit in visits:
            #print("vsit",visit)
            newVisit = []
            for code in visit:
                #print("code",code)
                newVisit.append(reverseTypes[code])
                    #print("newVisit",newVisit)
            newPatient.append(newVisit)
        newSeqs.append(newPatient)     
        
    ## mapping from codes to integer    
    #types = {"PAD":0 , "SOH":1,  "EOV":2,  "EOH":3}
    #types = {"PAD":0 , "SOH":1,  "EOV":2}
    types= {}
    updatedSeqs = []
    for patient in newSeqs:
        newPatient = []
        #print("patient",patient)
        for visit in patient:
            #print("vsit",visit)
            newVisit = []
            for code in visit:
                #print("code",code)
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)+4
                    newVisit.append(types[code])
                    #print("newVisit",newVisit)
            newPatient.append(newVisit)
        updatedSeqs.append(newPatient)
    types.update({"PAD":0 , "SOH":1,  "EOV":2, "EOH":3})
    reverseTypes = {v:k for k,v in types.items()}


    return updatedSeqs,types,reverseTypes

def saveFiles(updatedSeqs,types,codeDescription):
    outFile = os.path.join('outputData','originalData')
    #save =True
    #if save:
    pickle.dump(updatedSeqs, open(outFile+'.seqs', 'wb'), -1)
    pickle.dump(types, open(outFile+'.types', 'wb'), -1)
    pickle.dump(codeDescription, open(outFile+'.description', 'wb'), -1)

def generateCodeTypes(outFile,reverseTypes):
    icdTOCCS_Map = pickle.load(open('icd10DP_to_cssr_dictionary','rb'))
    codeType = {}
    countD = 0
    countP=0
    countDr =0
    countT =0
    for keys,values in reverseTypes.items():
        #print(keys,values)
        found =0
        if keys not in codeType:
            if values.startswith('DR_'):
                found =1
            
                codeType[keys] ='DR'
                countDr= countDr+1
            elif values == 'SOH' or values=='PAD' or values=='EOV' or values=='EOH':
                found =1
                
                codeType[keys] ='T'
                countT= countT+1
            else:
                for k,v in icdTOCCS_Map.items():
                    #print(k,v)
                    #break
                    if values in v:
                        found =1
                        if keys not in codeType:
                            if k.startswith('D'):
                                codeType[keys] = 'D'
                                countD= countD+1
                            elif k.startswith('P'):
                                codeType[keys] = 'P'
                                countP= countP+1
                #else:
                    #print(f"Thes value {values} is not present in icdTOCCS_Map map ")
            if found ==0:
                print(keys,values)
    # store the data as it will be useful
    print(countD,countP,countDr,countT)        
    pickle.dump(codeType, open(outFile+'.codeType', 'wb'), -1)
    
    return codeType

def load_data(outFile):
    # load the data again
    seqs = pickle.load(open(outFile +'.seqs','rb'))
    types = pickle.load(open(outFile + '.types','rb'))
    codeType = pickle.load(open(outFile + '.codeType','rb'))
    reverseTypes = {v:k for k,v in types.items()}
    return seqs,types,codeType,reverseTypes





def pairing1(x,y):
    pairs =[]
    for i,a in enumerate(zip(x,y)):
        pairs.append(a)
    
    return pairs

def pairing2(x):
    inp,trg,pairs =[],[],[]
    for x in x:
        inp.append(x[:-1])
        trg.append(x[1:])
    for i,a in enumerate(zip(inp,trg)):
        pairs.append(a)
    return pairs

def pairing3(x):
    inp,trg,pairs =[],[],[]
    for x in x:
        inp.append(x[:-1])
        trg.append([x[-1]])
    for i,a in enumerate(zip(inp,trg)):
        pairs.append(a)
    return pairs

def stats(newPairs,mn =600):
    x,y = [],[]
    count,county,counts =0,0,0
    for pair in newPairs:
        if len(pair[0]) > mn:
            count =count +1

        if len(pair[1]) > mn:
            county =county +1 

        if len(pair[0])>mn and len(pair[1])  >mn:
            counts = counts +1
        x.append(len(pair[0]))
        y.append(len(pair[1]))
    #print(f"\n Statistics of the input and output data")
    #print(f"\n Avg seq len y: {sum(y)/len(y)} ,  Avg seq len x: {sum(x)/len(x)}")  
    #print(f"\n Total no of pairs > seq of len({mn}): \n X: {count},\n Y : {county},\n X,Y : {counts},\n total pairs :{len(newPairs)} \n max value X :{max(torch.tensor(x)) }\n max value Y :{max(torch.tensor(y))}")
    if count > 0 or county > 0 or counts > 0:
        run = True
    else:
        run = False
    return run
## removing the pairs which are less than mn    
def removePairs(newPairs,mn =600):
    print(f"\n  Total no of pairs before removing :{len(newPairs)}")
    b= len(newPairs)
    x,y,curPair = [],[],[]
    count,county,counts =0,0,0
    for pair in newPairs:
        if len(pair[0]) > mn and len(pair[1]) > mn:
            counts =counts +1
            #newPairs.remove(pair)   
        elif len(pair[0]) > mn or len(pair[1]) > mn:
            count =count +1
            #newPairs.remove(pair)
        else:
            curPair.append(pair)
            
    print(f"\n  Total no of pairs after removing :{len(curPair)}")
    print(f"\n  Total no of pairs removed :{b-len(curPair)}")
    return curPair

def updateOutput(newPairs,codeType,diagnosis=0,procedure=0,drugs =0,all =0):
    updSeqs = []
    if procedure ==1 and drugs == 1:
        print("\n Removing drug and procedure codes from output for forecasting diagnosis code only")
        for i,pair in enumerate(newPairs):
            newOutput = []
            for code in pair[1]:
                if (codeType[code] =='D' or codeType[code] =='T'):
                    newOutput.append(code)
                        
            if len(newOutput)>=4:
            #print(f"{newOutput} \n")
                updSeqs.append((pair[0],newOutput))
    if drugs == 1 and procedure ==0:
        print("\n Removing only drug codes from output for forecasting diagnosis and procedure code only")
        for i,pair in enumerate(newPairs):
            newOutput = []
            for code in pair[1]:
                if not (codeType[code] == 'DR'):
                    newOutput.append(code)
            if len(newOutput)>=4:
                updSeqs.append((pair[0],newOutput))
    if all:
        print("\n keeping all codes")
        updSeqs = newPairs.copy()
        
    return updSeqs


def resetIntegerOutput(updSeqs,isall =1):
    # updating the output codes to reduce hypothesis space as some of the medical codes have been removed.

    # outTypes = {prev-codes : new-codes} ,  token codes remain same 
    updPair = []
    outTypes = {}
    outTypes.update({0:0 , 1:1,  2:2, 3:3})
    for i,pair in enumerate(updSeqs):
        newVisit = []
        for code in pair[1]:
            if code in outTypes:
                newVisit.append(outTypes[code])
            else:
                outTypes[code] = len(outTypes)
                newVisit.append(outTypes[code])
        updPair.append((pair[0],newVisit))
    return updPair,outTypes



# record = t1,t2,t3 
# example inp = t1 , output = t2,t3
#         inp = t1,t2 , output = t3 -- trajevtory forecasting
def split_sequence3(sequence):
    X, y,pairs = list(), list(),list()
    for i in range(len(sequence)):
    # find the end of this pattern
        #print(f"i:{i}, seq: {len(sequence)}")
        # check if we are beyond the sequence
        if i+1 >= len(sequence):
            break
        seq_x, seq_y = sequence[:i+1], sequence[i+1:]
        #print(f"{seq_x} ---- {seq_y}")
        #print("in")
        #print(sequence[:i+1],sequence[i+1:])
        X.append(seq_x)
        y.append(seq_y)
        #print(f"X: {X}, Y: {y}")
    pairs=pairing1(X,y)
    return pairs


# input = t-1,t-2  , output = t-2,t 
# example input = t1,t2  , output = t2,t3  -- doctor ai # only once for each patient
def split_sequence4(sequence, n_steps):
    X,pairs = list(),list()
    for i in range(1):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input  parts of the pattern
        
        seq_x = sequence[i:]
        X.append(seq_x)
    pairs=pairing2(X)
    return pairs

def split_sequence0(sequence):
    X, y,pairs = list(), list(),list()
    for i in range(len(sequence)):
    # find the end of this pattern
        #print(f"i:{i}, seq: {len(sequence)} \n {sequence}")
        # check if we are beyond the sequence
        if i+1 >= len(sequence):
            break
        seq_x, seq_y = sequence[:i+1], [sequence[i+1]]
        #print(f"X: {seq_x} ----\n Y: {seq_y}")
       # print("in")
        #print(sequence[:i+1],sequence[i+1])
        X.append(seq_x)
        y.append(seq_y)
        #print(f"X: {X}, Y: {y}")
    pairs=pairing1(X,y)
    
    return pairs

def formatData(originalSeqs,dataFormat = 'TF', mn = 400):
    pairs = []
    for i in range(len(originalSeqs)):
        if dataFormat =='TF':
            pairs.extend(split_sequence3(originalSeqs[i]))
        elif dataFormat =='SDP':
            #pairs.extend(split_sequence2(originalSeqs[i],1))
            pairs.extend(split_sequence0(originalSeqs[i]))
        elif dataFormat =='DAI':
            pairs.extend(split_sequence4(originalSeqs[i],1))
            
        else:
            print("Wrong Format")
            
    newPairs,p = [],[]

    for pair in pairs:
        #print("paiot",pair)
        input,output,p =[],[],[]
        for i in pair[0]:
            #print("i",i)
            i = i +[2]
            input.extend(i)
        p.append([1]+ input + [3])
        for o in pair[1]:
            o = o +[2]
            #print("o",o)
            output.extend(o)
        p.append([1]+ output+ [3])

        newPairs.append(tuple(p))
    ## sample
    n =2
    #print(f" Orginal: {pairs[:10]}  \n\n\n After formating : {newPairs[:10]} \n ----------------------------------------\n\n\n")
    if(stats(newPairs,mn=mn)):
        print(f"\n\n\nRemoving pairs greater than  {mn} seq length")
        newPairs = removePairs(newPairs,mn=mn)
        stats(newPairs)
    return newPairs
    #pairs.extend(split_sequence1(seqs[i],1,1))
    #pairs.extend(split_sequence2(seqs[i],1))
    #pairs.extend(split_sequence3(seqs[i]))
    #pairs.extend(split_sequence4(seqs[i],1))
    #print(f"pairs :{pairs}")

    
def storeFiles(pair,outTypes,codeType,types,reverseTypes,outFile):
    pickle.dump(pair, open(outFile+'.seqs', 'wb'), -1)
    pickle.dump(outTypes, open(outFile+'.outTypes', 'wb'), -1)
    pickle.dump(codeType, open(outFile+'.codeType', 'wb'), -1)
    pickle.dump(types, open(outFile+'.types', 'wb'), -1)
    pickle.dump(reverseTypes, open(outFile+'.reverseTypes', 'wb'), -1)
    reverseOutTypes = {v:k for k,v in outTypes.items()}
    pickle.dump(reverseOutTypes, open(outFile+'.reverseTypes', 'wb'), -1)


def main(args):
    mimic3_path = args.mimic3_path
    CCSRDX_file = args.CCSRDX_file
    CCSRPCS_file = args.CCSRPCS_file
    D_CCSR_Ref_file = args.D_CCSR_Ref_file
    P_CCSR_Ref_file = args.P_CCSR_Ref_file
    CCSDX_file = args.CCSDX_file
    CCSPX_file = args.CCSPX_file
    min_dx = args.min_dx
    min_px = args.min_px
    min_drg = args.min_drg
    seqLength = args.seqLength
    threshold = args.threshold
    print("Loading the data...")
    pidAdmMap,admDxMap,admPxMap,admDrugMap,drugDescription=load_mimic_data(mimic3_path,CCSRDX_file,CCSRPCS_file,choice ='ndc')
    print("\n Completed...")
    #stage 2 and 3
    print("\n Cleaning data...")
    pidAdmMap,adDx,adPx,adDrug = clean_data(pidAdmMap,admDxMap,admPxMap,admDrugMap)
    print("\n Completed...")
    #stage 4
    print("\n Mapping ICD data to CCS and CCSR...")
    adDx,adPx,codeDescription = icd_mapping(CCSRDX_file,CCSRPCS_file,CCSDX_file,CCSPX_file,D_CCSR_Ref_file,P_CCSR_Ref_file,adDx,adPx,adDrug,drugDescription)
    print("\n Completed...")
    #stage 5
    print("\n Trimming the codes assigned per visit based on a threshold...")
    adDx,adPx,adDrug= trim(min_dx,min_px,min_drg,adDx,adPx,adDrug)
    print("\n Completed...")
    
    print("\n Building the data..")
    newSeqs,types=buildData(pidAdmMap,adDx,adPx,adDrug)
    #stage 6
    print(f"\n removing the code whose occurence is less than a certain {threshold}")
    updatedSeqs,types,reverseTypes = removeCode(newSeqs,types,threshold=threshold)
    # outFile - is a folder path in the working directory where the data is going to get stored
    outFile = os.path.join('outputData','originalData')
    print("\n Save the data before formmating based on the task")
    saveFiles(updatedSeqs,types,codeDescription)
    codeType = generateCodeTypes(outFile,reverseTypes)
    seqs,types,codeType,reverseTypes = load_data(outFile)
    print("\n Completed...")
    print("\n Preparing data for Trajectory Forecasting....")
    # sequence length threshold  -mn
    newPairs = formatData(seqs,dataFormat = 'TF',mn = seqLength)
    diagnosisOutputFile = os.path.join('outputData','TF','Inp_d_p_dr_out_d')
    diagnosisProcedureOutputFile = os.path.join('outputData','TF','Inp_d_p_dr_out_d_p')
    AllOutputFile = os.path.join('outputData','TF','Inp_d_p_dr_out_d_p_dr')

    print(f"\n Remove certain codes from output for different data formats")
    AllUpdPair,AllOutTypes= resetIntegerOutput(updateOutput(newPairs.copy(),codeType,diagnosis=0,procedure=0,drugs =0,all =1))
    diagnosisUpdPair,diagnosisOutTypes= resetIntegerOutput(updateOutput(newPairs.copy(),codeType,diagnosis=0,procedure=1,drugs =1,all =0))
    diagnosisProcedureUpdPair,diagnosisProcedureOutTypes= resetIntegerOutput(updateOutput(newPairs.copy(),codeType,diagnosis=0,procedure=0,drugs =1,all =0))

    print(f"\n total # S1 records : {len(diagnosisUpdPair)}\n total # S2 records :{len(diagnosisProcedureUpdPair)}\n total # S3 records :{len(AllUpdPair)}")
    print(f"\n total Dx codes:{len(diagnosisOutTypes)} \n  total Dx,Px codes:{len(diagnosisProcedureOutTypes)} \n total Dx,Px,Rx codes:{len(AllOutTypes)}")
    print("\n Storing all the information related to Trajectory Forecasting...")

        
    storeFiles(diagnosisUpdPair,diagnosisOutTypes,codeType,types,reverseTypes,diagnosisOutputFile)
    storeFiles(diagnosisProcedureUpdPair,diagnosisProcedureOutTypes,codeType,types,reverseTypes,diagnosisProcedureOutputFile)
    storeFiles(AllUpdPair,AllOutTypes,codeType,types,reverseTypes,AllOutputFile)
    print("\n Completed...")

    print("\nPreparing data for Sequential disease prediction....")
    newPairs = formatData(seqs,dataFormat = 'SDP',mn =500)
    diagnosisOutputFile = os.path.join('outputData','SDP','Inp_d_p_dr_out_d')

    print(f"\n\n Remove certain codes from output for different data formats")
    diagnosisUpdPair,diagnosisOutTypes= resetIntegerOutput(updateOutput(newPairs.copy(),codeType,diagnosis=0,procedure=1,drugs =1,all =0))

    print(f"\n total # records: {len(diagnosisUpdPair)} \n total # of codes: {len(diagnosisOutTypes)}")
   
    print("\n Storing all the information related to TSequential disease prediction...")
    storeFiles(diagnosisUpdPair,diagnosisOutTypes,codeType,types,reverseTypes,diagnosisOutputFile)
    print("\n Completed...")
    print("\n All the preprocessing step has been completed, Now use the data in the outputData folder to build the model...")


parser.add_argument('--mimic3_path',default='data', type=str,help="Path of mimic IV CSV files where the queried data is stored")

parser.add_argument('--CCSRDX_file',default='DXCCSR_v2021-2/DXCCSR_v2021-2.csv', type=str,help="Path of diagnosis based CCSR files")
parser.add_argument('--CCSRPCS_file',default='PRCCSR_v2021-1/PRCCSR_v2021-1.csv', type=str,help="Path of procedure based CCSR files")

parser.add_argument('--D_CCSR_Ref_file',default='DXCCSR_v2021-2/DXCCSR-Reference-File-v2021-2.xlsx', type=str,help="Path of diagnosis based CCSR Reference file ")
parser.add_argument('--P_CCSR_Ref_file',default='PRCCSR_v2021-1/PRCCSR-Reference-File-v2021-1.xlsx', type=str,help="Path of procedure based CCSR Reference file")


parser.add_argument('--CCSDX_file',default='Single_Level_CCS_2015/$dxref 2015.csv', type=str,help="Path of diagnosis based CCS files")
parser.add_argument('--CCSPX_file',default='Single_Level_CCS_2015/$prref 2015.csv', type=str,help="Path of procedure based CCS files")

parser.add_argument('--min_dx',default=80, type=int,help="Minimum diagnosis code assigned per visit")
parser.add_argument('--min_px',default=80, type=int,help="Minimum procedure code assigned per visit")
parser.add_argument('--min_drg',default=80, type=int,help="Minimum drug/medication code assigned per visit")

parser.add_argument('--threshold',default=5, type=int,help="Remove the code whose frequency  is less than the threshold")
parser.add_argument('--seqLength',default=500, type=int,help="maximum sequence length of each sequence in input and output")




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)