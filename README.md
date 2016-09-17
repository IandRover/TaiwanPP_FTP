# TaiwanPP_FTP
pest prediction in Taiwan

TaiwanPP is a website aims to provide pest prediction for plants growers as well as government officials
The code consist of 6 parts, and we have done the first three parts, including basic climatological data establishment ,a comparison table and a pest population data.
  1. basic climatological data <br />
    basic climatological data are climatological data collected from 2013 from CWB Observation Data Inquire System<br /> (http://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp) <br />
    The files are in "basic climatological data.zip".
    They are named by the rules "County Name + Town Name + Year + Month + .txt" <br />
  2. Meteorological_stations_table <br />
    Every Meteorological stations get its own id and Name. <br />
    Since not all pest station has its own corresponding Meteorological stations, we manually select the nearest Meteorological stations for each pest station.<br />
    Thus, this table is required to form a valid and colplete url to reuqest climatological data.<br />
    The files called "Meteorological_stations_table.csv". 
  3. pest population data <br />
    Pest population data is collected from Taiwan Agricultural Research Institute<br /> (http://data.coa.gov.tw/Service/OpenData/DataFileService.aspx?UnitId=140).<br />
    The web updated every 1~3 weeks. We, however did not request the data in code, since the web have limitation in requesting data.<br />
    Also, some of the Station Names are imcomplete and false. <br />
    We provide the manually corrected pest population data.<br />
    The files are in "pest population data.zip". 
    Those csv files are name by "PestName + County Name + Town Name + .csv "

Download all the above files to a folder, and create another folder named "data" inside it.
    
The last 3 parts include the following parts:
  1. Climatological_Ddata_Update.py <br />
    We crawl data from CWB Observation Data Inquire System (http://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp) <br />
  2. raw data process + ML&Prediction + writing html pages <br />
    We change the raw html data in a form that we can easily perform machine learning.<br />
    Then perform Prediction with the lastest 10 days climatological data<br />
    name: CORE.py
    Since we did not establish a database as well as use dJANGO, we have the program automatically output an html file 
  3. upload html files to FTP <br />
    We upload the html files to FTP in this procedure<br />
    name:TO_FTP.py

Download all the above files to a folder, the path name may need to be changed


