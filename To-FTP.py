
# coding: utf-8

# In[1]:

import ftplib
import urllib
import urllib.request,urllib.parse,urllib.error
import time

pest = ['Bactroceraxxdorsalis','Bactroceraxxcucurbitae','Spodopteraxxlitura','Spodopteraxxexigua']

server_name = 'http://185.27.134.9/'
user_name= 'b8_18854503'
password = '01025091'

os_path = "C://Users//Aaron//Pests_prediction_github//data//web//"

proxy_support = urllib.request.ProxyHandler({'sock5': 'localhost:1080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

print("logging in ...")


with ftplib.FTP('ftp.byethost8.com') as ftp:
    
    ftp.login(user_name, password)
    ftp.dir('htdocs/web')
    ftp.cwd('htdocs/web')
    print(ftp.nlst())
    for pest_name in pest:
        time.sleep(30)
        filename = pest_name + '.html'
        fp=open(os_path+filename,'rb')
        if filename not in ftp.nlst():
            ftp.storbinary('STOR '+ filename,fp)
            print("file upload")
        else:
            ftp.delete(filename)
            ftp.storbinary('STOR '+ filename,fp)
            print("file updated")
    ftp.close
    ftp.quit

