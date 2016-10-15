
# coding: utf-8

# ###### 

# ###### 步驟
# 1. 讀取 地區資訊 和 東方果實蠅的東西
#     out_put = di, df
# 2. 進行 data_organization()
#     out_put = X, y, 分級初始值, 級距, 級數
# 3. 進行 data balancing
#     out_put = X, y
# 3.5 讀取欲預測之天氣()
#     Out_put = 欲預測之天氣_X
# 4. 重複十次以下步驟 : 尚未完成
#     input = X, y, 欲預測之天氣, 害蟲名稱
# 
#     把data 八二分  計算 score 的值，儲存到buffer_scoring
#     用 training data 預測   with   X = 輸入的天氣，儲存到buffer_predict
# 
#     svm(kernel = rbf, linear, poly, sigmoid)
#     RandomForest(n_estimator = 20, 25, 30, 35)
# 5. 計算
# buffer_scoring 的平均與標準差
# 如何選取預測值
#     出現最多次的標籤
#     把 scoring的平方 乘上 buffer_predict 除以 sum(scoring的平方)
#     選取 top 7 scoring 的 prediction ，選取出現最多次的標籤
# 6. 儲存資訊 in list 包括 害蟲名稱 縣市名稱 n天後預測 預測規模 誤差 初始值 級距 級數



import codecs
import pandas as pd
from collections import Counter
import time
import os 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import RandomizedPCA
from sklearn.learning_curve import learning_curve
from bs4 import BeautifulSoup
import datetime
import csv
from sklearn import svm
import numpy as np
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import pylab as pl
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import seaborn

print('成功')

def read_station_data():
    with open('.\\data\\地區資訊new.csv', 'r') as f: 
        reader = csv.reader(f)
        station_data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        f.close()
    return station_data
def read_pest_data(pest):
    with open('.\\data\\蔬果重要害蟲防疫旬報_'+pest+'_new.csv','r') as f:
        reader = csv.reader(f)
        pest_data = list(list(rec) for rec in csv.reader(f, delimiter=','))
        f.close()
    return pest_data

def todate():
    import time
    todate_year = time.strftime("%Y")
    todate_month = time.strftime("%m")
    if int(todate_month) <= 9:
        todate_month = todate_month.strip('0') 
    todate_day = time.strftime("%d")
    return todate_year, todate_month, todate_day

def period20d():
    period20d_year = (datetime.datetime.now()-datetime.timedelta(days=20)).strftime("%y")
    period20d_month = (datetime.datetime.now()-datetime.timedelta(days=20)).strftime("%m")
    if int(period20d_month) <= 9:
        period20d_month = period20d_month.strip('0') 
    period20d_day = (datetime.datetime.now()-datetime.timedelta(days=20)).strftime("%d")
    return period20d_year, period20d_month, period20d_day

def open_data(filename, row_sd, n_start, n_multi, n_time):

    print(filename)
    with open(filename,'r',encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        buffer = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        f.close()
    X, y = data_organization(buffer, row_sd, n_multi, n_start, n_time)
    return X, y


def data_organization(buffer, row_sd, n_start, n_multi, n_time):
    
    while [] in buffer:
        buffer.remove([])
    input_x=[]
    real_y=[]

    range_level = []
    range_level.append(0)
    for i in range(n_time):
        range_level.append(n_start*pow(n_multi,i))
    range_level[-1]=(10000)
   


    for buffera in buffer:
        for i in range(n_time):
            if float(buffera[6]) < range_level[i+1] and float(float(buffera[6])) >= range_level[i]:
                buffera[6] = str(i+1)
        input_x.append(buffera[7:])
        real_y.append(float(buffera[6]))

        
    buffer_a = []
    buffer_b = []
    min_length=50
    max_length=30
    for element in input_x:
        if len(element) > max_length:
            max_length = len(element)
        for elementa in element:
            if elementa == '':
                buffer_a.append(0)
            elif elementa == "X" or elementa =='T':
                buffer_a.append(0)
            else:
                buffer_a.append(float(elementa))
        buffer_b.append(buffer_a)
        buffer_a = []
    for element in buffer_b:
        while len(element) != max_length:
            element.append(0)
            if len(element) > max_length:
                break #just in case
    X = np.array(buffer_b)
    y = np.array(real_y)
    return X, y

def make_data_balanced(X,y):

    uni_class = np.unique(y)
    
    buffer_X =[]
    buffer_y =[]
    
    if len(uni_class) == 2:
        buffer_X,buffer_y = make_data_balanced_2(X,y)
        
    elif len(uni_class) == 1:
        print('天啊，我竟然也遇到這個狀況......那就不處理，直接送出去')
        buffer_X = X
        buffer_y = y
        
    else:
        ratio = ration(y)
        X = X.tolist()
        y = y.tolist()
        buffer_X = X
        buffer_y = y
        count_ratio = 0
        while ratio > 2.1:
            
            count_ratio += 1
            uni_class, uni_class_voting, max_index, min_index, ratio = ration_oo(buffer_y)
            for i in range(len(buffer_y)):
                if y[i] == uni_class[min_index]:
                    for repeat in range(round(ratio)-1):
                        buffer_X.append(X[i]) 
                        buffer_y.append(y[i])
            
            ratio = ration(buffer_y)

    buffer_X = np.array(buffer_X)
    buffer_y = np.array(buffer_y)
    return buffer_X, buffer_y

def make_data_balanced_2(X,y):

    uni_class, uni_class_voting, max_index, min_index, ratio = ration_oo(y)
    if len(uni_class) == 2:
        ratio = ration(y)
        buffer_X = X
        buffer_y = y
        while ratio >= 2.1:
            buffer_X = buffer_X.tolist()
            buffer_y = buffer_y.tolist()
            for i in range(len(y)):
                if y[i] == uni_class[min_index]:
                    for repeat in range(int(ratio)-1):
                        buffer_X.append(X[i])
                        buffer_y.append(y[i])
                else:
                    buffer_X.append(X[i])
                    buffer_y.append(y[i])
            ratio = ration(buffer_y)
        buffer_X = np.array(buffer_X)
        buffer_y = np.array(buffer_y)
        
    else:
        buffer_X = X
        buffer_y = y
    return buffer_X, buffer_y

print('成功')

def ration_oo(y):
    y = np.array(y).astype(int)
    uni_class = np.unique(y)
    uni_class_voting = np.bincount(y)
    
    index = (uni_class_voting == 0)
    buffer = []
    for i in range(len(uni_class_voting)):
        if not index[i]:
            buffer.append(uni_class_voting[i])
    uni_class_voting = np.array(buffer)
    
    max_index = np.argmax(uni_class_voting)
    min_index = np.argmin(uni_class_voting)

    ratio = uni_class_voting[max_index]/uni_class_voting[min_index]
    ratio = int(round(ratio))
    
    return uni_class, uni_class_voting, max_index, min_index, ratio
def ration(y):
    y = np.array(y).astype(int)
    uni_class = np.unique(y)
    uni_class_voting = np.bincount((y))
    
    index = (uni_class_voting == 0)
    buffer = []
    for i in range(len(uni_class_voting)):
        if not index[i]:
            buffer.append(uni_class_voting[i])
    uni_class_voting = np.array(buffer)
    
    max_index = np.argmax(uni_class_voting)
    min_index = np.argmin(uni_class_voting)

    ratio = uni_class_voting[max_index]/uni_class_voting[min_index]
    ratio = int(round(ratio))
    
    return ratio
print('成功')


def predict_X(row_sd):
    sub_buffer=[]
    todate_year, todate_month, todate_day = todate()
    
    if int(todate_day) > 20:
        sub_buffer, need_to_continue =write_day10(row_sd)

    if int(todate_day) < 20 and int(todate_month) != 1:
        sub_buffer, need_to_continue =write_gap_month(row_sd)

    if int(todate_day) < 20 and int(todate_month) == 1:
        sub_buffer, need_to_continue =write_gap_year(row_sd)
    #print(sub_buffer)
    buffera = []
    if need_to_continue == 0:
        for i in range(len(sub_buffer)):
            if sub_buffer[i] == '':
                buffera.append(0)
            elif sub_buffer[i] == "X":
                buffera.append(0)
            elif sub_buffer[i] == 'T':
                buffera.append(0)
            elif i == 0 or i == 1:
                buffera.append(sub_buffer[i])
            else:
                buffera.append(float(sub_buffer[i]))

    return buffera

def write_gap_year(row_sd):
    need_to_continue = 0
    todate_year, todate_month, todate_day = todate()
    period20d_year, period20d_month, period20d_day = period20d()
    row_sd = row_sd[:2]
    
    filename_last_month = '.\\data\\' + row_sd[0] + row_sd[1] + '20'+str(period20d_year) + str(period20d_month) + '.txt' 
    with open(filename_last_month,'r') as f:
        data = f.read()
        f.close()
    sp = BeautifulSoup(data,'lxml')
    tra = sp.find_all('tr')

    try:
        tra.remove(tra[0])
        tra.remove(tra[0])
        tra.remove(tra[0])
    except:
        need_to_continue = 1
        print('tra=[]')
        return row_sd, need_to_continue
    
    traa_count=0
    for traa in tra:
        traa_count = traa_count+1
        if traa_count >= int(period20d_day):
            tda=traa.find_all('td')
            tdaa_count = 0
            for tdaa in tda:
                tdaa_count = tdaa_count +1
                if  tdaa_count == 1:
                    row_sd.append(tdaa.text)
                if  tdaa_count == 8 or tdaa_count==9 or tdaa_count==11 or tdaa_count==18:
                    row_sd.append(tdaa.text.replace('\xa0',''))
                    
    filename = '.\\data\\' + row_sd[0] + row_sd[1] + str(todate_year) + str(todate_month) + '.txt'
    with open(filename,'r') as f:
        data = f.read()
        f.close()
    sp = BeautifulSoup(data,'lxml')
    tra = sp.find_all('tr')
    
    try:
        tra.remove(tra[0])
        tra.remove(tra[0])
        tra.remove(tra[0])
    except:
        need_to_continue = 1
        print('tra=[]')
        return row_sd, need_to_continue
    
    traa_count=0
    for traa in tra:
        traa_count = traa_count+1
        if traa_count < int(todate_day):
            tda=traa.find_all('td')
            tdaa_count = 0
            for tdaa in tda:
                tdaa_count = tdaa_count +1
                if  tdaa_count == 1:
                    row_sd.append(tdaa.text)
                if  tdaa_count == 8 or tdaa_count==9 or tdaa_count==11 or tdaa_count==18:
                    row_sd.append(tdaa.text.replace('\xa0',''))
    
    print('成功')
    return row_sd, need_to_continue

def write_gap_month(row_sd):
    need_to_continue = 0
    todate_year, todate_month, todate_day = todate()
    period20d_year, period20d_month, period20d_day = period20d()
    row_sd = row_sd[:2]
    
    filename_last_month = '.\\data\\' + row_sd[0] + row_sd[1] + '20' + str(period20d_year) + str(period20d_month) + '.txt'
    with open(filename_last_month,'r') as f:
        data = f.read()
        f.close()
    sp = BeautifulSoup(data,'lxml')
    tra = sp.find_all('tr')

    try:
        tra.remove(tra[0])
        tra.remove(tra[0])
        tra.remove(tra[0])
    except:
        need_to_continue = 1
        print('tra=[]')
        return row_sd, need_to_continue
    
    traa_count=0
    for traa in tra:
        traa_count = traa_count+1
        if traa_count >= int(period20d_day):
            tda=traa.find_all('td')
            tdaa_count = 0
            for tdaa in tda:
                tdaa_count = tdaa_count +1
                if  tdaa_count == 1:
                    row_sd.append(tdaa.text)
                if  tdaa_count == 8 or tdaa_count==9 or tdaa_count==11 or tdaa_count==18:
                    row_sd.append(tdaa.text.replace('\xa0',''))
                    
    filename = '.\\data\\' + row_sd[0] + row_sd[1] + str(todate_year) + str(todate_month) + '.txt'
    with open(filename,'r') as f:
        data = f.read()
        f.close()
    sp = BeautifulSoup(data,'lxml')
    tra = sp.find_all('tr')
    try:
        tra.remove(tra[0])
        tra.remove(tra[0])
        tra.remove(tra[0])
    except:
        need_to_continue = 1
        print('tra=[]')
        return row_sd, need_to_continue
    
    traa_count=0
    for traa in tra:
        traa_count = traa_count+1
        if traa_count < int(todate_day):
            tda=traa.find_all('td')
            tdaa_count = 0
            for tdaa in tda:
                tdaa_count = tdaa_count +1
                if  tdaa_count == 1:
                    row_sd.append(tdaa.text)
                if  tdaa_count == 8 or tdaa_count==9 or tdaa_count==11 or tdaa_count==18:
                    row_sd.append(tdaa.text.replace('\xa0',''))
    
    print('成功')
    return row_sd, need_to_continue



def write_day10(row_sd):
    need_to_continue = 0
    todate_year, todate_month, todate_day = todate()
    filename = '.\\data\\' + row_sd[0] + row_sd[1] + todate_year + todate_month + '.txt' 
    row_sd = row_sd[:2]
    with open(filename,'r') as f:
        data = f.read()
        f.close()
    sp = BeautifulSoup(data,'lxml')
    tra = sp.find_all('tr')
    try:
        tra.remove(tra[0])
        tra.remove(tra[0])
        tra.remove(tra[0])
    except:
        need_to_return = 1
        print('tra=[]')
        return row_sd, need_to_return
    
    traa_count=0
    for traa in tra:
        traa_count = traa_count+1
        if traa_count < int(todate_day) and traa_count>= int(todate_day)-20 :
            tda=traa.find_all('td')
            tdaa_count = 0
            for tdaa in tda:
                tdaa_count = tdaa_count +1
                if  tdaa_count == 1:
                    row_sd.append(tdaa.text)
                if  tdaa_count == 8 or tdaa_count==9 or tdaa_count==11 or tdaa_count==18:
                    row_sd.append(tdaa.text.replace('\xa0',''))
    print('成功')
    return row_sd, need_to_continue

def second_possible(data):
    
    b = np.unique(data)
    c = np.bincount(data)
    c = c.tolist()
    while 0 in c:
        c.remove(0)
    try:
        max_index = np.argmax(c)
        b = b.tolist()
        b.remove(b[max_index])
        c.remove(c[max_index])
        max_index = np.argmax(c)
        value = b[max_index]
    except:
        value="x"
        pass
    return value

def predict_weather(X, y, pest_name, predict_X, DF, row_sd):
    import statistics
    column = ['病蟲害','縣市名稱','時間','規模','準確率','第二規模','第二準確率']
    predict_X = np.array(predict_X)

    data = X
    score = []
    predict = []
    prediction = 0
    city_name = ''
    score_standard_deviation = 0
    buffer_X = X
    buffer_predict_X = predict_X
    #print(predict_X)
    city_name = row_sd[0] + row_sd[1]
        
    
    if True:
        try:
            city_name = predict_X[0] + predict_X[1]
        except:
            for day in [1, 2, 3, 4, 5, 6, ]:
                DF2=pd.DataFrame([[pest_name, city_name, day, "-1", "0", '0', "0"]], columns=column)
                DF = DF.append(DF2, ignore_index=True)
            return DF
        
        for day in [1, 2, 3, 4, 5, 6, 7]:
            print('day=',day)
            predict=[]
            score=[]
            X = buffer_X
            predict_X = buffer_predict_X
            score=[]
            predict=[]
            score_SD = []            
            buffera_X=[]
            for row_X in X:
                buffera_X.append(row_X[:(20-day)*5])
                
            if (len(predict_X) < 65 and day == 7) or (len(predict_X) < 70 and day == 6) or (len(predict_X) < 75 and day == 5) or (len(predict_X) < 80 and day == 4) or (len(predict_X) < 85 and day == 3) or (len(predict_X) < 90 and day == 2) or (len(predict_X) < 95 and day == 1):
                print('insufficient data')
                DF2=pd.DataFrame([[pest_name, city_name, day, "-1", "-1", "-1", "-1"]], columns=column)
                DF = DF.append(DF2, ignore_index=True)
                continue
            if len(np.unique(y)) == 1:
                print('insufficient label')
                DF2=pd.DataFrame([[pest_name, city_name, day, y[0], "-2", '-2', "-2"]], columns=column)
                DF = DF.append(DF2, ignore_index=True)
                continue
            buffera_predict_X = predict_X[-(20-day)*5:]
            X = buffera_X
            predict_X = buffera_predict_X
            
            X = np.array(X)
            predict_X = predict_X.reshape(1,-1)
            for val in range(20,35,1):
                for repeata in range(1):
                    clf = RandomForestClassifier(n_estimators = val)
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    except ValueError:
                        print('the number of sample is too less')
                        break
                    except:
                        print('something wrong happen')
                        break
                    
                    if len(y_test) == 1:
                        X = X.reshape(1,-1)
                    
                    y = np.array(y).astype(int)
                    uni_class = np.unique(y)
                    uni_class_voting = np.bincount((y))
                    max_index = np.argmax(uni_class_voting)
                    a = [y_test==max_index]
                    buffer_X_test=[]
                    buffer_y_test=[]
                    for i in range(len(a[0])):
                        if a[0][i]:
                            buffer_X_test.append(X_test[i])
                            buffer_y_test.append(y_test[i])
                    X_test = buffer_X_test
                    y_test = buffer_y_test
                    
                    if X_test == []:
                        print('Found array with 0 feature(s) (shape=(1, 0)) while a minimum of 1 is required.')
                        continue
                    
                    try:
                        score.append(clf.fit(X_train,y_train).score(X_test, y_test))
                        predict.append(clf.fit(X_train,y_train).predict(predict_X)[0])
                    except:
                        print('The number of classes has to be greater than one; got 1')
                        #DF2=pd.DataFrame([[pest_name, city_name, day, y_train[0], "all_label", 'are same']], columns=column)
                        #DF = DF.append(DF2, ignore_index=True)
                        #print(X_test, y_test)
                        pass


            del clf
            for kernel in ['rbf','poly','sigmoid']:#,'sigmoid'
                for C in [1000,300,100]:
                    for gamma in [0.01, 0.001]:
                        for i in range(1):
                            try:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                            except ValueError:
                                print('the number of sample is too less')
                                break
                            except:
                                print('something wrong happen')
                                break
                            
                            y = np.array(y).astype(int)
                            uni_class = np.unique(y)
                            uni_class_voting = np.bincount((y))
                            max_index = np.argmax(uni_class_voting)
                            a = [y_test==max_index]
                            buffer_X_test=[]
                            buffer_y_test=[]
                            for i in range(len(a[0])):
                                if a[0][i]:
                                    buffer_X_test.append(X_test[i])
                                    buffer_y_test.append(y_test[i])
                            X_test = buffer_X_test
                            y_test = buffer_y_test
                            
                            clf = SVC(C=C, gamma=gamma, kernel=kernel)
                            
                            if X_test == []:
                                print('Found array with 0 feature(s) (shape=(1, 0)) while a minimum of 1 is required.')
                                continue
                    
                            try:
                                score.append(clf.fit(X_train,y_train).score(X_test, y_test))
                                #print("predict",clf.fit(X_train,y_train).predict(predict_X)[0])
                                predict.append(clf.fit(X_train,y_train).predict(predict_X)[0])
                                predict.append(clf.fit(X_train,y_train).predict(predict_X)[0])
                            except ValueError:
                                print('The number of classes has to be greater than one; got 1')
                                #DF2=pd.DataFrame([[pest_name, city_name, day, y_train[0], "X", "X"]], columns=column)
                                #DF = DF.append(DF2, ignore_index=True)
                                continue_the_loop = 1
                                pass
                            


            if (not score) or (not predict) or (len(score) == 1):
                DF2=pd.DataFrame([[pest_name, city_name, day, '0', "0", "0", "0"]], columns=column)
                DF = DF.append(DF2, ignore_index=True)
                break
            
            d = {x:predict.count(x) for x in predict}
            print(d)
            counts = Counter(d)
            prediction = round((counts.most_common(1)[0][0]),3)
            total_accuracy = 0
            for i in counts.most_common():
                total_accuracy = total_accuracy + (i[1])
            
            try:
                prediction_1, prediction_2 = counts.most_common(2)
                DF2=pd.DataFrame([[pest_name, city_name, day, prediction_1[0], round(prediction_1[1]/total_accuracy,3), prediction_2[0], round(prediction_2[1]/total_accuracy,3)]], columns=column)
                DF = DF.append(DF2, ignore_index=True)
            except:
                prediction_1 = counts.most_common(1)
                DF2=pd.DataFrame([[pest_name, city_name, day, prediction_1[0][0], round(prediction_1[0][1]/total_accuracy,3), '0', "0"]], columns=column)
                DF = DF.append(DF2, ignore_index=True)
                pass

    return DF

def create_html(pest_name, DF):
    pest_name_list=['東方果實蠅','瓜實蠅','斜紋夜蛾','甜菜夜蛾']
    pest_name_elist=['Bactroceraxxdorsalis','Bactroceraxxcucurbitae','Spodopteraxxlitura','Spodopteraxxexigua']
    for i, name in enumerate(pest_name_list):
        if pest_name == name:
            pest_name_eng = pest_name_elist[i]

    df = DF.values.tolist()
    buffer='['
    for lista_i, lista in enumerate(df):
        buffer = buffer + "["
        for element_i, element in enumerate(lista):
            if element_i==0:
                buffer = buffer + "\'" + str(element) + "\'"
            elif element_i==1 or element_i==4 or element_i == 5:
                buffer = buffer + ",\'" + str(element) + "\'"
            else:
                buffer = buffer + ',' + str(element)
        buffer = buffer + "]" + ","
    buffer = buffer[:-1]
    buffer = buffer + "]"
    
    filename_html = ('.\\data\\web\\' + pest_name_eng + ".html")
    
    html_text="""<!doctype html><html data-brackets-id='1'>
<head data-brackets-id='2'>
    <meta data-brackets-id='3' charset="UTF-8" content="width=device-width, initial-scale=1.0" name="viewport">
    <title data-brackets-id='4'>基本格式</title>
	<script data-brackets-id='5' language="javascript">
    """ + "var the_data = " + buffer + """;
	 
	</script>
    <link rel="stylesheet" type="text/css" href="others.css" />
</head>
<body>
<h2 class="col-12" align=center>台灣蔬果農業害蟲預測</h2>
    
    <div class="col-3 k " width="9">
        <img class="map" id="taiwan_map" src="happytaiwan.png" alt="台灣地圖" data-image-width="573">
    </div>
    <div class="point">"""+station_img(pest_name)+"""
    </div>
    
		<table id="table1" border=1 align = center>
			<tr>
				<th colspan="8">害蟲名稱</th>
			</tr>
			<tr>
				<th colspan="8">縣市鄉鎮名稱</th>
			</tr>
			<tr>
				<td>預測天數</td>
				<td>一天後</td>
				<td>兩天後</td>
				<td>三天後</td>
                <td>四天後</td>
				<td>五天後</td>
				<td>六天後</td>
                <td>七天後</td>
			</tr>
			<tr>
				<td>預測規模</td>
				<td>0</td>
				<td>0</td>
				<td>0</td>
                <td>0</td>
				<td>0</td>
				<td>0</td>
                <td>0</td>
			</tr>
			<tr>
				<td>預測率</td>
				<td>0</td>
				<td>0</td>
				<td>0</td>
                <td>0</td>
				<td>0</td>
                <td>0</td>
				<td>0</td>
			</tr>
            <tr>
				<td>第二可能規模</td>
				<td>0</td>
				<td>0</td>
				<td>0</td>
                <td>0</td>
				<td>0</td>
                <td>0</td>
				<td>0</td>
			</tr>
            <tr>
				<td>預測率</td>
				<td>0</td>
				<td>0</td>
				<td>0</td>
                <td>0</td>
				<td>0</td>
                <td>0</td>
				<td>0</td>
			</tr>
		</table>
	<script data-brackets-id='91' language="javascript">
			window.onload = function(){
					var plusBtn_1 = document.querySelector("div.point")
					if (plusBtn_1){
						plusBtn_1.addEventListener('click', function(e) {
						var sub_table = Array();
						var object_id = e.target.id;
						
    					for(i=0;i<the_data.length;i+=1){
							if ((object_id == the_data[i][1])&&(object_id)){
								sub_table.push(the_data[i]);
							}
						}
						var my_table = "<div id='tablePrint'>";
							var my_table = "<table><tr><th colspan='8'>""" + pest_name + """</th></tr>";
							my_table += "<tr><th colspan='8'>"+sub_table[0][1]+"</th></tr>";
							my_table += "<tr><td>預測天數</td><td>一天後</td><td>兩天後</td><td>三天後</td><td>四天後</td><td>五天後</td><td>六天後</td><td>七天後</td></tr>";
							my_table += "<tr><td>預測規模</td><td>"+sub_table[0][3]+"</td><td>"+sub_table[1][3]+"</td><td>"+sub_table[2][3]+"</td><td>"+sub_table[3][3]+"</td><td>"+sub_table[4][3]+"</td><td>"+sub_table[5][3]+"</td><td>"+sub_table[6][3]+"</td></tr>";
							my_table += "<tr><td>預測率</td><td>"+sub_table[0][4]+"</td><td>"+sub_table[1][4]+"</td><td>"+sub_table[2][4]+"</td><td>"+sub_table[3][4]+"</td><td>"+sub_table[4][4]+"</td><td>"+sub_table[5][4]+"</td><td>"+sub_table[6][4]+"</td></tr>";
                            my_table += "<tr><td>第二可能規模</td><td>"+sub_table[0][5]+"</td><td>"+sub_table[1][5]+"</td><td>"+sub_table[2][5]+"</td><td>"+sub_table[3][5]+"</td><td>"+sub_table[4][5]+"</td><td>"+sub_table[5][5]+"</td><td>"+sub_table[6][5]+"</td></tr>";
                            my_table += "<tr><td>預測率</td><td>"+sub_table[0][6]+"</td><td>"+sub_table[1][6]+"</td><td>"+sub_table[2][6]+"</td><td>"+sub_table[3][6]+"</td><td>"+sub_table[4][6]+"</td><td>"+sub_table[5][6]+"</td><td>"+sub_table[6][6]+"</td></tr></table>";

							document.getElementById('table1').innerHTML = my_table;
						})
						}
				}
		</script>
    <div id="footer">NCTU, NYMU 2016 iGEM collaboration</div>
    <div id="footer">UPDATE:""" + str(datetime.datetime.now()) + """</div>
    <div class="sub">
        請點選位置圖示來獲得最新資料
    </div>
</body>
</html>"""
    with codecs.open(filename_html,'w', "utf-8-sig") as f:
        f.write(html_text)
        f.close

def station_img(pest_name):
    if pest_name == "東方果實蠅":
        station_text = """
        <img id="台中市石岡區" src="pest-observation.png" alt="" style="position:absolute;top:357px;left:280px" width="9">
		<img id="台中市后里區" src="pest-observation.png" alt="" style="position:absolute;top:347px;left:265px" width="9">
		<img id="台中市和平區" src="pest-observation.png" alt="" style="position:absolute;top:360px;left:330px" width="9">
		<img id="台中市東勢區" src="pest-observation.png" alt="" style="position:absolute;top:362px;left:305px" width="9">
		<img id="台東縣台東地區" src="pest-observation.png" alt="" style="position:absolute;top:640px;left:360px" width="9">
		<img id="台南市大內區" src="pest-observation.png" alt="" style="position:absolute;top:612px;left:198pX" width="9">
		<img id="台南市玉井區" src="pest-observation.png" alt="" style="position:absolute;top:622px;left:208px" width="9">
		<img id="台南市官田區" src="pest-observation.png" alt="" style="position:absolute;top:602px;left:190px" width="9">
		<img id="台南市東山區" src="pest-observation.png" alt="" style="position:absolute;top:587px;left:220px" width="9">
		<img id="台南市南化區" src="pest-observation.png" alt="" style="position:absolute;top:612px;left:230px" width="9">
		<img id="宜蘭縣冬山鄉" src="pest-observation.png" alt="" style="position:absolute;top:280px;left:470px" width="9">
		<img id="宜蘭縣員山鄉" src="pest-observation.png" alt="" style="position:absolute;top:265px;left:450Px" width="9">
		<img id="宜蘭縣頭城鎮" src="pest-observation.png" alt="" style="position:absolute;top:230px;left:483px" width="9">
		<img id="花蓮縣玉溪地區" src="pest-observation.png" alt="" style="position:absolute;top:547px;left:404px" width="9">
		<img id="花蓮縣花蓮市" src="pest-observation.png" alt="" style="position:absolute;top:382px;left:458px" width="9">
		<img id="花蓮縣富里鄉" src="pest-observation.png" alt="" style="position:absolute;top:593px;left:390px" width="9">
		<img id="花蓮縣瑞穗鄉" src="pest-observation.png" alt="" style="position:absolute;top:534px;left:395px" width="9">
		<img id="金門縣金門地區" src="pest-observation.png" alt="" style="position:absolute;top:140px;left:95px" width="9">
		<img id="南投縣中寮鄉" src="pest-observation.png" alt="" style="position:absolute;top:440px;left:273px" width="9">
		<img id="南投縣水里鄉" src="pest-observation.png" alt="" style="position:absolute;top:460px;left:288px" width="9">
		<img id="南投縣埔里鎮" src="pest-observation.png" alt="" style="position:absolute;top:426px;left:297px" width="9">
		<img id="屏東縣里港鄉" src="pest-observation.png" alt="" style="position:absolute;top:700px;left:226px" width="9">
		<img id="屏東縣佳冬鄉" src="pest-observation.png" alt="" style="position:absolute;top:780px;left:230px" width="9">
		<img id="屏東縣枋山地區" src="pest-observation.png" alt="" style="position:absolute;top:820px;left:258px" width="9">
		<img id="屏東縣枋寮地區" src="pest-observation.png" alt="" style="position:absolute;top:790px;left:240px" width="9">
		<img id="屏東縣長治鄉" src="pest-observation.png" alt="" style="position:absolute;top:720px;left:241px" width="9">
		<img id="屏東縣高樹鄉" src="pest-observation.png" alt="" style="position:absolute;top:690px;left:244px" width="9">
		<img id="屏東縣麟洛鄉" src="pest-observation.png" alt="" style="position:absolute;top:730px;left:235px" width="9">
		<img id="苗栗縣三灣鄉" src="pest-observation.png" alt="" style="position:absolute;top:273px;left:322px" width="9">
		<img id="苗栗縣公館鄉" src="pest-observation.png" alt="" style="position:absolute;top:300px;left:300px" width="9">
		<img id="苗栗縣卓蘭鄉" src="pest-observation.png" alt="" style="position:absolute;top:340px;left:300px" width="9">
		<img id="桃園市大溪區" src="pest-observation.png" alt="" style="position:absolute;top:220px;left:393px" width="9">
		<img id="桃園縣復興區" src="pest-observation.png" alt="" style="position:absolute;top:260px;left:410px" width="9">
		<img id="桃園市龍潭區" src="pest-observation.png" alt="" style="position:absolute;top:230px;left:388px" width="9">
		<img id="高雄市大社區" src="pest-observation.png" alt="" style="position:absolute;top:706px;left:197px" width="9">
		<img id="高雄市六龜區" src="pest-observation.png" alt="" style="position:absolute;top:640px;left:248px" width="9">
		<img id="高雄市田寮區" src="pest-observation.png" alt="" style="position:absolute;top:674px;left:197px" width="9">
		<img id="高雄市杉林區" src="pest-observation.png" alt="" style="position:absolute;top:650px;left:230px" width="9">
		<img id="高雄市阿蓮區" src="pest-observation.png" alt="" style="position:absolute;top:678px;left:187px" width="9">
		<img id="高雄市旗山區" src="pest-observation.png" alt="" style="position:absolute;top:678px;left:220px" width="9">
		<img id="高雄市燕巢區" src="pest-observation.png" alt="" style="position:absolute;top:690px;left:200px" width="9">
		<img id="雲林縣斗六市" src="pest-observation.png" alt="" style="position:absolute;top:485px;left:236px" width="9">
		<img id="雲林縣古坑鄉" src="pest-observation.png" alt="" style="position:absolute;top:503px;left:240px" width="9">
		<img id="雲林縣林內鄉" src="pest-observation.png" alt="" style="position:absolute;top:475px;left:244px" width="9">
		<img id="新竹縣北埔鄉" src="pest-observation.png" alt="" style="position:absolute;top:262px;left:340px" width="9">
		<img id="新竹縣芎林鄉" src="pest-observation.png" alt="" style="position:absolute;top:248px;left:353px" width="9">
		<img id="新竹縣峨眉鄉" src="pest-observation.png" alt="" style="position:absolute;top:262px;left:328px" width="9">
		<img id="新竹縣新埔鄉" src="pest-observation.png" alt="" style="position:absolute;top:222px;left:340px" width="9">
		<img id="新竹縣關西鄉" src="pest-observation.png" alt="" style="position:absolute;top:240px;left:368px" width="9">
		<img id="嘉義縣中埔鄉" src="pest-observation.png" alt="" style="position:absolute;top:555px;left:227px" width="9">
		<img id="嘉義縣竹崎鄉" src="pest-observation.png" alt="" style="position:absolute;top:530px;left:232px" width="9">
		<img id="嘉義縣梅山鄉" src="pest-observation.png" alt="" style="position:absolute;top:520px;left:242px" width="9">
		<img id="嘉義縣番路鄉" src="pest-observation.png" alt="" style="position:absolute;top:550px;left:245px" width="9">
		<img id="嘉義縣義竹鄉" src="pest-observation.png" alt="" style="position:absolute;top:570px;left:157px" width="9">
		<img id="彰化縣二林鎮" src="pest-observation.png" alt="" style="position:absolute;top:425px;left:197px" width="9">
		<img id="彰化縣社頭鄉" src="pest-observation.png" alt="" style="position:absolute;top:440px;left:240px" width="9">
		<img id="彰化縣花壇鄉" src="pest-observation.png" alt="" style="position:absolute;top:410px;left:233px" width="9">
		<img id="彰化縣員林市" src="pest-observation.png" alt="" style="position:absolute;top:422px;left:240px" width="9">
		<img id="彰化縣溪州鄉" src="pest-observation.png" alt="" style="position:absolute;top:458px;left:220px" width="9">
		<img id="澎湖縣澎湖地區" src="pest-observation.png" alt="" style="position:absolute;top:550px;left:50px" width="9">"""
    if pest_name == "瓜實蠅":
        station_text = """
        <img id="台中市霧峰區" src="pest-observation.png" alt="" style="position:absolute;top:405px;left:263px" width="9">
		<img id="台南市東山區" src="pest-observation.png" alt="" style="position:absolute;top:587px;left:220px" width="9">
		<img id="台南市七股區" src="pest-observation.png" alt="" style="position:absolute;top:635px;left:145px" width="9">
		<img id="台南市佳里區" src="pest-observation.png" alt="" style="position:absolute;top:620px;left:155px" width="9">
		<img id="宜蘭縣壯圍鄉" src="pest-observation.png" alt="" style="position:absolute;top:240px;left:495px" width="9">
		<img id="花蓮縣玉溪地區" src="pest-observation.png" alt="" style="position:absolute;top:547px;left:404px" width="9">
		<img id="花蓮縣花蓮市" src="pest-observation.png" alt="" style="position:absolute;top:382px;left:458px" width="9">
		<img id="花蓮縣瑞穗鄉" src="pest-observation.png" alt="" style="position:absolute;top:534px;left:395px" width="9">
		<img id="金門縣金門地區" src="pest-observation.png" alt="" style="position:absolute;top:140px;left:95px" width="9">
		<img id="南投縣埔里鎮" src="pest-observation.png" alt="" style="position:absolute;top:426px;left:297px" width="9">
		<img id="屏東縣萬丹鄉" src="pest-observation.png" alt="" style="position:absolute;top:740px;left:220px" width="9">
		<img id="苗栗縣後龍鎮" src="pest-observation.png" alt="" style="position:absolute;top:275px;left:285px" width="9">
		<img id="雲林縣二崙鄉" src="pest-observation.png" alt="" style="position:absolute;top:470px;left:190px" width="9">
		<img id="新竹縣峨眉鄉" src="pest-observation.png" alt="" style="position:absolute;top:262px;left:328px" width="9">
		<img id="嘉義縣義竹鄉" src="pest-observation.png" alt="" style="position:absolute;top:570px;left:157px" width="9">
		<img id="彰化縣員林市" src="pest-observation.png" alt="" style="position:absolute;top:422px;left:240px" width="9">
        """
    if pest_name == "甜菜夜蛾":
        station_text= """
        <img id="彰化縣大城鄉" src="pest-observation.png" alt="" style="position:absolute;top:450px;left:180px" width="9">
		<img id="彰化縣田尾鄉" src="pest-observation.png" alt="" style="position:absolute;top:435px;left:220px" width="9">
		<img id="彰化縣福興鄉" src="pest-observation.png" alt="" style="position:absolute;top:405px;left:200px" width="9">
		<img id="新竹縣竹北市" src="pest-observation.png" alt="" style="position:absolute;top:217px;left:320px" width="9">
		<img id="新北市蘆洲區" src="pest-observation.png" alt="" style="position:absolute;top:170px;left:421px" width="9">
		<img id="雲林縣褒忠鄉" src="pest-observation.png" alt="" style="position:absolute;top:475px;left:175px" width="9">
		<img id="高雄市岡山區" src="pest-observation.png" alt="" style="position:absolute;top:690px;left:180px" width="9">
		<img id="苗栗縣公館鄉" src="pest-observation.png" alt="" style="position:absolute;top:300px;left:305px" width="9">
		<img id="宜蘭縣三星地區" src="pest-observation.png" alt="" style="position:absolute;top:273px;left:470px" width="9">
		<img id="台中市大甲區" src="pest-observation.png" alt="" style="position:absolute;top:335px;left:250px" width="9">
		<img id="台中市龍井區" src="pest-observation.png" alt="" style="position:absolute;top:365px;left:230px" width="9">
		<img id="台中市霧峰區" src="pest-observation.png" alt="" style="position:absolute;top:405px;left:263px" width="9">
        """
    if pest_name == "斜紋夜蛾":
        station_text= """
        <img id="彰化縣大城鄉" src="pest-observation.png" alt="" style="position:absolute;top:450px;left:180px" width="9">
		<img id="彰化縣田尾鄉" src="pest-observation.png" alt="" style="position:absolute;top:435px;left:220px" width="9">
		<img id="彰化縣福興鄉" src="pest-observation.png" alt="" style="position:absolute;top:405px;left:200px" width="9">
		<img id="新竹縣竹北市" src="pest-observation.png" alt="" style="position:absolute;top:217px;left:320px" width="9">
		<img id="新北市蘆洲區" src="pest-observation.png" alt="" style="position:absolute;top:170px;left:421px" width="9">
		<img id="雲林縣褒忠鄉" src="pest-observation.png" alt="" style="position:absolute;top:475px;left:175px" width="9">
		<img id="高雄市岡山區" src="pest-observation.png" alt="" style="position:absolute;top:690px;left:180px" width="9">
		<img id="苗栗縣後龍鎮" src="pest-observation.png" alt="" style="position:absolute;top:275px;left:285px" width="9">
		<img id="苗栗縣公館鄉" src="pest-observation.png" alt="" style="position:absolute;top:300px;left:305px" width="9">
		<img id="花蓮縣吉安鄉" src="pest-observation.png" alt="" style="position:absolute;top:410px;left:452px" width="9">
		<img id="花蓮縣新秀地區" src="pest-observation.png" alt="" style="position:absolute;top:383px;left:450px" width="9">
		<img id="宜蘭縣三星地區" src="pest-observation.png" alt="" style="position:absolute;top:273px;left:470px" width="9">
		<img id="台南市官田區" src="pest-observation.png" alt="" style="position:absolute;top:602px;left:190px" width="9">
		<img id="台中市大甲區" src="pest-observation.png" alt="" style="position:absolute;top:335px;left:250px" width="9">
		<img id="台中市龍井區" src="pest-observation.png" alt="" style="position:absolute;top:365px;left:230px" width="9">
		<img id="台中市霧峰區" src="pest-observation.png" alt="" style="position:absolute;top:405px;left:263px" width="9">

        """
    return station_text

station_data = read_station_data()
station_data = station_data[1:]
count_sd = 0

for i, pest_name in enumerate(['東方果實蠅','瓜實蠅','甜菜夜蛾','斜紋夜蛾']):

    column = ['病蟲害','縣市名稱','時間','規模','準確率','第二規模','第二準確率']
    DF = pd.DataFrame(columns=column)
    for row_sd in station_data:
        count_sd = count_sd +1
        filename = ('.\\data\\'+pest_name+ row_sd[0]+ row_sd[1]+str(20)+'.csv')
        if os.path.isfile(filename):
            #if count_sd < 4 :
            #    continue
            print(datetime.datetime.now())
            if pest_name == '東方果實蠅':
                 X, y = open_data(filename, row_sd, 16, 4, 4) ## output as nump
            elif pest_name == '瓜實蠅':
                 X, y = open_data(filename, row_sd, 9, 3, 3)
            elif pest_name == '斜紋夜蛾':
                 X, y = open_data(filename, row_sd, 12, 4, 3)
            elif pest_name == '甜菜夜蛾':
                 X, y = open_data(filename, row_sd, 4, 3, 3)

            X, y = make_data_balanced(X,y)
            predicta = predict_X(row_sd)
            DF = predict_weather(X, y, pest_name, predicta, DF, row_sd)
            print(datetime.datetime.now())
            print('')

    
    create_html(pest_name, DF)

