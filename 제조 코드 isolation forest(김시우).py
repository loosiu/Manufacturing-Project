import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def interval(SPINDLE_SPEED,SPINDLE_LOAD, model):
    LOAD_DATA, ERROR_DATA, Seq_Index = [], [], []
    if model == "A1":                                                       #A1 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=500
        SPINDLE_SPEED[SPINDLE_SPEED>=400]=500
        SPINDLE_SPEED = SPINDLE_SPEED-500

    elif model == "A2":                                                     #A2 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=600
        SPINDLE_SPEED[SPINDLE_SPEED>=600]=600
        SPINDLE_SPEED = SPINDLE_SPEED-600
        
    SPINDLE_SPEED = SPINDLE_SPEED*(-1)
    SPINDLE_SPEED[SPINDLE_SPEED>0]=300
    SPINDLE_SPEED_FF  = SPINDLE_SPEED.diff()

    for i, ff in enumerate(SPINDLE_SPEED_FF[:-200]):                        #diff 처리 후 고점, 저점 탬색
        if ff>0:
            length = SPINDLE_SPEED_FF[i:i+200]
            if min(length)<0:
                Seq_Index.append([i,length.index[length == min(length)][0]])
                    
    indexs = []                                     
    for x, y in Seq_Index:                                                  #치리된 고점, 저점을 하나의 섹터로 하고 그 섹터의 최저값을 시작 및 끝 지점으로 사이클 정의
        data = SPINDLE_LOAD[x:y]
        index = data.index[data ==min(data)]
        indexs.append(index[0])
        
    for i,j in enumerate(indexs[:-2]):                                      # 사이클에서 정상적인 범주와 비정상적인 범주 확인 후 처리
        a = j
        b = indexs[i+1]
        data = SPINDLE_LOAD[a:b]
        if (j+300 > indexs[i+1])&(data.isna().sum()==0)&((b-a)<3000)&(len(data.index[data==0])/len(data)<0.5):
            LOAD_DATA.append(data)
            # print(LOAD_DATA)
            
        else:
            ERROR_DATA.append(data)
    print(len(LOAD_DATA),"\t",len(ERROR_DATA))
    # print(LOAD_DATA)
    return LOAD_DATA, ERROR_DATA

def interval_err(SPINDLE_SPEED,SPINDLE_LOAD, model):
    LOAD_DATA, ERROR_DATA, Seq_Index = [], [], []
    if model == "A1":                                                       #A1 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=500
        SPINDLE_SPEED[SPINDLE_SPEED>=400]=500
        SPINDLE_SPEED = SPINDLE_SPEED-500

    elif model == "A2":                                                     #A2 기기에 대한 전처리
        SPINDLE_SPEED[SPINDLE_SPEED<=100]=600
        SPINDLE_SPEED[SPINDLE_SPEED>=600]=600
        SPINDLE_SPEED = SPINDLE_SPEED-600
        
    SPINDLE_SPEED = SPINDLE_SPEED*(-1)
    SPINDLE_SPEED[SPINDLE_SPEED>0]=300
    SPINDLE_SPEED_FF  = SPINDLE_SPEED.diff()

    for i, ff in enumerate(SPINDLE_SPEED_FF[:-200]):                        #diff 처리 후 고점, 저점 탬색
        if ff>0:
            length = SPINDLE_SPEED_FF[i:i+200]
            if min(length)<0:
                Seq_Index.append([i,length.index[length == min(length)][0]])
                    
    indexs = []                                     
    for x, y in Seq_Index:                                                  #치리된 고점, 저점을 하나의 섹터로 하고 그 섹터의 최저값을 시작 및 끝 지점으로 사이클 정의
        data = SPINDLE_LOAD[x:y]
        index = data.index[data ==min(data)]
        indexs.append(index[0])
        
    for i,j in enumerate(indexs[:-2]):                                      # 사이클에서 정상적인 범주와 비정상적인 범주 확인 후 처리
        a = j
        b = indexs[i+1]
        data = SPINDLE_LOAD[a:b]
        if data.isna().sum()==0:
            LOAD_DATA.append(data) 
        else:
            ERROR_DATA.append(data)
    print(len(LOAD_DATA),"\t",len(ERROR_DATA))

    return LOAD_DATA, ERROR_DATA

def data_preprocess(paths,m,val=False,error=False):
    if error==False:
        interval_define = interval
    else:
        interval_define = interval_err
    total_data = []
    total_error_data = []
    columns = []
    if val ==True:
        path = paths
        columns.append(path.split("/")[-1].split(".")[0])
        
        print(path)
        df = pd.read_csv(path)
        SPINDLE_SPEED = df[f"CDJK_MCB05_{m}_SPINDLE_SPEED"].copy()
        SPINDLE_LOAD = df[f"CDJK_MCB05_{m}_SPINDLE_LOAD"].copy()
        total_data, total_error_data = interval_define(SPINDLE_SPEED,SPINDLE_LOAD, m)
    else: 
        for path in sorted(paths):
            columns.append(path.split("/")[-1].split(".")[0])
            
            print(path)
            df = pd.read_csv(path)
            SPINDLE_SPEED = df[f"CDJK_MCB05_{m}_SPINDLE_SPEED"].copy()
            SPINDLE_LOAD = df[f"CDJK_MCB05_{m}_SPINDLE_LOAD"].copy()
            LOAD_DATA, ERROR_DATA = interval_define(SPINDLE_SPEED,SPINDLE_LOAD, m)

            total_data.extend(LOAD_DATA)
            total_error_data.append(ERROR_DATA)
        total_error_data = pd.DataFrame([total_error_data])
        total_error_data.columns = columns

    return total_data, total_error_data

paths = sorted(glob.glob("첨단정공 5월 데이터/*"))
train_path = paths[:19]
val_path = paths[19]      #64
test_path = paths[20:]

print("train")
train_data, train_error_data = data_preprocess(train_path,"A1")

print("val")
val_data, val_error_data = data_preprocess(val_path,"A1",val=True)

print("test")
test_data, test_error_data = data_preprocess(test_path,"A1")


from scipy.signal import find_peaks
from numpy import trapz

peaks_train_num = []
peaks_val_num = []
peaks_test_num = []

peaks_train_xvalue = []
peaks_val_xvalue = []
peaks_test_xvalue = []

peaks_train_yvalue = []
peaks_val_yvalue = []
peaks_test_yvalue = []

peaks_train_area = []
peaks_val_area = []
peaks_test_area = []

peaks_train_avg = []
peaks_val_avg = []
peaks_test_avg = []

peaks_train_var = []
peaks_val_var = []
peaks_test_var = []


for i in range(0,len(train_data)):
    train_x = np.array(train_data[i])
    peaks, properties = find_peaks(train_x, distance=9)
    peaks_train_num.append(len(peaks))
    
    # peaks_x_values = peaks
    peaks_train_xvalue.append(peaks)
    
    # peaks_y_values = train_x[peaks]
    peaks_train_yvalue.append(train_x[peaks])
    
    peaks_avg = np.mean(peaks_train_yvalue[i])
    peaks_train_avg.append(np.round(peaks_avg,2))
    
    peaks_var = np.var(peaks_train_yvalue[i])
    peaks_train_var.append(np.round(peaks_var,2))
    
    y = train_data[i]
    area = trapz(y)
    peaks_train_area.append(area)

for i in range(0,len(val_data)):
    val_x = np.array(val_data[i])
    peaks, properties = find_peaks(val_x, distance=9)
    peaks_val_num.append(len(peaks))
    
    # peaks_x_values = peaks
    peaks_val_xvalue.append(peaks)
    
    # peaks_y_values = val_x[peaks]
    peaks_val_yvalue.append(val_x[peaks])
    
    peaks_avg = np.mean(peaks_val_yvalue[i])
    peaks_val_avg.append(np.round(peaks_avg,2))
    
    peaks_var = np.var(peaks_val_yvalue[i])
    peaks_val_var.append(np.round(peaks_var,2))
    
    y = val_data[i]
    area = trapz(y)
    peaks_val_area.append(area)
    
for i in range(0,len(test_data)):
    test_x = np.array(test_data[i])
    peaks, properties = find_peaks(test_x, distance=9)
    peaks_test_num.append(len(peaks))
    
    # peaks_x_values = peaks
    peaks_test_xvalue.append(peaks)
    
    # peaks_y_values = test_x[peaks]
    peaks_test_yvalue.append(test_x[peaks])
    
    peaks_avg = np.mean(peaks_test_yvalue[i])
    peaks_test_avg.append(np.round(peaks_avg,2))
    
    peaks_var = np.var(peaks_test_yvalue[i])
    peaks_test_var.append(np.round(peaks_var,2))
    
    y = test_data[i]
    area = trapz(y)
    peaks_test_area.append(area)
    
peak_train_merge = []
peak_train_df = pd.DataFrame()    
peak_train_df['peak_avg'] = peaks_train_avg 
peak_train_df['peak_var'] = peaks_train_var
peak_train_df['peak_area'] = peaks_train_area

for i in range(0,len(peak_train_df)):
    peak_train_merge.append(peak_train_df.iloc[i].to_list())

peak_val_merge = []
peak_val_df = pd.DataFrame()    
peak_val_df['peak_avg'] = peaks_val_avg 
peak_val_df['peak_var'] = peaks_val_var
peak_val_df['peak_area'] = peaks_val_area

for i in range(0,len(peak_val_df)):
    peak_val_merge.append(peak_val_df.iloc[i].to_list())
    
peak_test_merge = []
peak_test_df = pd.DataFrame()    
peak_test_df['peak_avg'] = peaks_test_avg 
peak_test_df['peak_var'] = peaks_test_var
peak_test_df['peak_area'] = peaks_test_area

for i in range(0,len(peak_test_df)):
    peak_test_merge.append(peak_test_df.iloc[i].to_list())
    
def  isBiggerThanFive(x):
      return x==7
  
new_peaks_train_num =list(filter(isBiggerThanFive,peaks_train_num))
new_peaks_val_num =list(filter(isBiggerThanFive,peaks_val_num))
new_peaks_test_num =list(filter(isBiggerThanFive,peaks_test_num))


from sklearn.ensemble import IsolationForest


train = list(peak_train_merge)
val = list(peak_val_merge)
test = list(peak_test_merge)


clf = IsolationForest(contamination=0.07,random_state=42)
clf.fit(train)
# val_predict = clf.predict(val)
test_predict = clf.predict(test)

# peak_val_isolation = list(val_predict)
peak_test_isolation = list(test_predict)

def  abnormal(x):
      return x==-1

def  normal(x):
      return x==1

    
peak_test_abnormal_num =list(filter(abnormal,test_predict))
peak_test_normal_num =list(filter(normal,test_predict))


peak_normal = list(filter(lambda x: peak_test_isolation[x] == 1, range(len(peak_test_isolation))))
peak_abnormal = list(filter(lambda x: peak_test_isolation[x] == -1, range(len(peak_test_isolation))))


start_index = []

for i in range(0,len(peak_abnormal)):
    k = peak_abnormal[i]
    start_index.append(test_data[k].index[0])

index_df = pd.DataFrame()
index_df['start_index'] = start_index

index_df.to_excel('C:/Users/KSW/Desktop/A2_abnormal_index.csv')


import math

peak_abnormal_range = math.floor(len(peak_abnormal)/30)
peak_normal_range = math.floor(len(peak_normal)/30)

for i in range(0,len(peak_abnormal),peak_abnormal_range):
    k = peak_abnormal[i]
    if k <= 4095:
        test_df = pd.read_csv(test_path[0])
        test_df_speed = test_df['CDJK_MCB05_A1_SPINDLE_SPEED']
        test_df_load = test_df['CDJK_MCB05_A1_SPINDLE_LOAD']
        plt.subplot(2, 1, 1)
        plt.plot(test_df_speed, color = 'r', linewidth = 2, linestyle = 'solid', label = "A1_SPINDLE_SPEED")
        plt.xlim(test_data[k].index[0],test_data[k].index[-1])
        plt.xticks(rotation = 45)
        plt.title('spindle speed')

        
        plt.subplot(2, 1, 2)
        plt.plot(test_df_load, color = 'r', linewidth = 2, linestyle = 'solid', label = "A1_SPINDLE_LOAD")
        plt.xlim(test_data[k].index[0],test_data[k].index[-1])
        plt.xticks(rotation = 45)
        plt.title('spindle load')
        
        plt.tight_layout()
        plt.show()
        plt.close('all')   
    elif k > 4095:
        test_df = pd.read_csv(test_path[1])
        test_df_speed = test_df['CDJK_MCB05_A1_SPINDLE_SPEED']
        test_df_load = test_df['CDJK_MCB05_A1_SPINDLE_LOAD']
        plt.subplot(2, 1, 1)
        plt.plot(test_df_speed, color = 'r', linewidth = 2, linestyle = 'solid', label = "A1_SPINDLE_SPEED")
        plt.xlim(test_data[k].index[0],test_data[k].index[-1])
        plt.xticks(rotation = 45)
        plt.title('spindle speed')
        
        plt.subplot(2, 1, 2)
        plt.plot(test_df_load, color = 'r', linewidth = 2, linestyle = 'solid', label = "A1_SPINDLE_LOAD")
        plt.xlim(test_data[k].index[0],test_data[k].index[-1])
        plt.xticks(rotation = 45)
        plt.title('spindle load')

        plt.tight_layout()
        plt.show()
        plt.close('all')   