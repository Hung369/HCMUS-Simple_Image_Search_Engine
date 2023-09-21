import pandas as pd


class Dataset:
    def __init__(self, filename):
        self.filename = filename
        string = self.filename.split('.')
        if string[1] == 'json':
            self.df = pd.read_json(self.filename).transpose()
        elif string[1] == 'csv':
            self.df = pd.read_csv(self.filename)

    def getDataFrame(self):
        return self.df

    def getInfo(self): # get the basic information about dataset
        return self.df.info()

    def toList(self):
        return self.df.values.tolist()

    def toCSV(self, filename): #save to csv file
        form = str(filename)+".csv"
        self.df.to_csv(form, index=False)
