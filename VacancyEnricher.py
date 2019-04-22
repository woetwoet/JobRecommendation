import pandas as pd
import os
from VacancyData import VacancyData

# Set the variables
cvsastextPath = r"C:\Users\woutv\Jupyter\Jobs\inputdata\CvsText\\"


# Load data part
qd = VacancyData()
profiles = qd.getDataAsPandas()[2]
profiles['CvLocatie']= profiles['CvLocatie'].fillna('')

# add a column to set the filename
profiles['cvlocatietext'] = profiles.apply(lambda row: row.CvLocatie.replace("/Data/Profielen/","").replace("/","_")+'.txt', axis=1)
profiles['cvastext']=profiles.apply(lambda row:'', axis=1)

# for index, row in profiles.iterrows():
for index, row in profiles.iterrows():
    # print(getattr(row, "cvlocatietext"), getattr(row, "cvastext"))
    cvlocationastext = cvsastextPath + getattr(row, "cvlocatietext")
    if os.path.isfile(cvlocationastext):
        with open(cvlocationastext, 'r',errors='ignore') as file:
            data = file.read().replace('\n', '')
            profiles.at[index,'cvastext'] = data
        print('++++++ Following cv has been added : '+cvlocationastext)
            
    else:
        print('++++++ Following cv is not found : '+cvlocationastext)
    
profiles.to_csv(r'C:\Users\woutv\Jupyter\Jobs\inputdata\ProfielenMetCV.csv', sep=';', encoding='utf-8')


