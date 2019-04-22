import pandas as pd
import csv

class VacancyData:

    matchingPath = r"C:\Users\woutv\Jupyter\Jobs\inputdata\ProfielSolicitaties.csv"
    vacanciesPath = r"C:\Users\woutv\Jupyter\Jobs\inputdata\Vacatures.csv"
    profilePath = r"C:\Users\woutv\Jupyter\Jobs\inputdata\Profielen.csv"
    profileTestPath = r"C:\Users\woutv\Jupyter\Jobs\inputdata\ProfielenTest.csv"

    def getMatchings(self):    
        return self.getData()[0]
        
    def getVacancies(self):
        return self.getData()[1]

    def getProfiles(self):
        return self.getData()[2]

    def getData(self):

        return (
                csv.DictReader(open(self.matchingPath,encoding="utf-8-sig"), delimiter=";"),
                csv.DictReader(open(self.vacanciesPath,encoding="utf-8-sig"), delimiter=";"),
                csv.DictReader(open(self.profilePath,encoding="utf-8-sig"), delimiter=";"),                
                csv.DictReader(open(self.profileTestPath,encoding="utf-8-sig"), delimiter=";"),                
               )
    def getDataAsPandas(self):
        return (
                pd.read_csv(self.matchingPath, sep=';', error_bad_lines=False, encoding="utf-8-sig"),
                pd.read_csv(self.vacanciesPath, sep=';', error_bad_lines=False, encoding="utf-8-sig"),
                pd.read_csv(self.profilePath, sep=';', error_bad_lines=False, encoding="utf-8-sig"),
                pd.read_csv(self.profileTestPath, sep=';', error_bad_lines=False, encoding="utf-8-sig")
        )




