
# coding: utf-8

# In[1]:

import csv
import os
from bs4 import BeautifulSoup
import datetime
import time
import urllib
import requests
import socket
import urllib.request,urllib.parse,urllib.error
import random


# In[2]:

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


# In[3]:

def period10d():
    period10d_year = (datetime.datetime.now()-datetime.timedelta(days=10)).strftime("%Y")
    period10d_month = (datetime.datetime.now()-datetime.timedelta(days=10)).strftime("%m")
    if int(period10d_month) <= 9:
        period10d_month = period10d_month.strip('0') 
    period10d_day = (datetime.datetime.now()-datetime.timedelta(days=10)).strftime("%d")
    return period10d_year, period10d_month, period10d_day


# In[4]:

def todate():
    import time
    todate_year = time.strftime("%Y")
    todate_month = time.strftime("%m")
    if int(todate_month) <= 9:
        todate_month = todate_month.strip('0') 
    todate_day = time.strftime("%d")
    return todate_year, todate_month, todate_day


# In[5]:

def write_gap_year(row_sd):
    need_to_continue = 0
    todate_year, todate_month, todate_day = todate()
    period10d_year, period10d_month, period10d_day = period10d()
    
    filename_last_month = '.\\data\\' + row_sd[0] + row_sd[1] + str(period10_year) + str(period10_month) + '.txt' 
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
        if traa_count >= int(day_10_ago):
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
        if traa_count < int(day):
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
    period10d_year, period10d_month, period10d_day = period10d()
    
    filename_last_month = '.\\data\\' + row_sd[0] + row_sd[1] + str(period10_year) + str(period10_month) + '.txt'
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
        if traa_count >= int(day_10_ago):
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
    tra.remove(tra[0])
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
        if traa_count < int(day):
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
    with open(filename,'r') as f:
        data = f.read()
        f.close()
    sp = BeautifulSoup(data,'lxml')
    tra = sp.find_all('tr')
    tra.remove(tra[0])
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
        if traa_count < int(todate_day) and traa_count>= int(todate_day)-10 :
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

def update_thi_month_data(row_sd):
    
    todate_year, todate_month, todate_day = todate()
    filename = ('.\\'+ row_sd[0]+ row_sd[1]+todate_year+todate_month+'.txt')
    print(filename)
    
    proxy_support = urllib.request.ProxyHandler({'sock5': 'localhost:1080'})
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)
    
    if int(todate_month) < 10:
        url ='http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station='+str(row_sd[2])+'&stname='+str(row_sd[5])+'&datepicker='+todate_year+'-0'+todate_month  
    else:
        url ='http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station='+str(row_sd[2])+'&stname='+str(row_sd[5])+'&datepicker='+todate_year+'-'+todate_month 
    time.sleep(random.randint(40,45))
    
    try:
        errorc=0
        response = urllib.request.urlopen(url)
        html = response.read()
    except urllib.error.URLError as e:
        if hasattr(e, 'reason'):
            print('We failed to reach a server.')
            print('Reason: ', e.reason)
            errorc=1
            pass
        elif hasattr(e, 'code'):
            print('The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
            errorc=1
            pass
        else:
            print('well, sonething wrong did happen, try again then')
            errorc = 1
    else:
        print('everything is fine')
        
                
    if errorc==0:
        filename = '.\\data\\' + str(row_sd[0]) + str(row_sd[1]) + str(todate_year) + str(todate_month) + '.txt'
        with open(filename, 'w') as f:
            f.write(str(html))
            f.close()
        time.sleep(random.randint(4,14))
    else:
        time.sleep(random.randint(1,4))
    return errorc
print('完成')

def update_last_month_data(row_sd):
    
    period10d_year, period10d_month, period10d_day = period10d()
    filename = ('.\\'+ row_sd[0]+ row_sd[1]+period10d_year+period10d_month+'.txt')
    print(filename)
    
    proxy_support = urllib.request.ProxyHandler({'sock5': 'localhost:1080'})
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)
    
    if int(period10d_month) < 10:
        url ='http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station='+str(row_sd[2])+'&stname='+str(row_sd[5])+'&datepicker='+period10d_year+'-0'+period10d_month  
    else:
        url ='http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station='+str(row_sd[2])+'&stname='+str(row_sd[5])+'&datepicker='+period10d_year+'-'+period10d_month 
    time.sleep(random.randint(40,45))
    
    try:
        errorc=0
        response = urllib.request.urlopen(url)
        html = response.read()
    except urllib.error.URLError as e:
        if hasattr(e, 'reason'):
            print('We failed to reach a server.')
            print('Reason: ', e.reason)
            errorc=1
            pass
        elif hasattr(e, 'code'):
            print('The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
            errorc=1
            pass
        else:
            print('well, sonething wrong did happen, try again then')
            errorc = 1
    else:
        print('everything is fine')
        
                
    if errorc==0:
        filename = '.\\data\\' + str(row_sd[0]) + str(row_sd[1]) + str(period10d_year) + str(period10d_month) + '.txt'
        with open(filename, 'w') as f:
            f.write(str(html))
            f.close()
        time.sleep(random.randint(4,14))
    else:
        time.sleep(random.randint(1,4))
    return errorc
print('完成')


# #### 每天早上更新資料

# In[6]:

count_sd = 0
station_data = read_station_data()
station_data = station_data[1:]
for row_sd in station_data:
    count_sd = count_sd+1
    if count_sd >= 1:
        errorc = 1  
        while errorc == 1:
            errorc = update_thi_month_data(row_sd)
        time.sleep(6)


# In[29]:

count_sd = 0
station_data = read_station_data()
station_data = station_data[1:]
for row_sd in station_data:
    count_sd = count_sd+1
    if count_sd >= 1:
        errorc = 1  
        while errorc == 1:
            errorc = update_last_month_data(row_sd)
        time.sleep(6)

