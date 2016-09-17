# TaiwanPP_FTP
pest prediction in Taiwan

TaiwanPP is a website aims to provide pest prediction for plants growers as well as government officials
The code consist of 6 parts, and we have done the first three parts, including basic climatological data establishment ,a comparison table and a pest population data.
  1. basic climatological data <br />
    basic climatological data are climatological data collected from 2013 from CWB Observation Data Inquire System (http://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp) 
    They are named by the rules <County Name><Town Name><Year><Month><.txt>
  2. comparison table
    Every Meteorological stations get its own id and Name. 
    Since not all pest station has its own corresponding Meteorological stations, we manually select the nearest Meteorological stations for each pest station.
    Thus, this table is required to form a valid and colplete url to reuqest climatological data.
  3. pest population data
    Pest population data is collected from Taiwan Agricultural Research Institute (http://data.coa.gov.tw/Service/OpenData/DataFileService.aspx?UnitId=140).
    The web updated every 1~3 weeks. We, however did not request the data in code, since the web have limitation in requesting data.
    Also, some of the Station Names are imcomplete and false. 
    We provide the manually corrected pest population data.
    
    
The last 3 parts include the following parts:
  1. climatological data Update
    We crawl data from CWB Observation Data Inquire System (http://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp) 
  2. raw data process + ML&Prediction + writing html pages 
    We change the raw html data in a form that we can easily perform machine learning.
    Then perform Prediction with the lastest 10 days climatological data
    Since we did not establish a database as well as use dJANGO, we have the program automatically output an html file 
  3. upload html files to FTP
    We upload the html files to FTP in this procedure


