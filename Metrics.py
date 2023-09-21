import pandas as pd


class Stats:
    def __init__(self, data):
        self.df = data

    def access(self, value):
        self.similar = self.df.loc[self.df['query'] == value, 'similar'].item()

    def scoring(self, found_result): # calculating recall, precision and F1 score
        common = list(set(found_result) & set(self.similar))
        self.recall = 100*(len(common) / len(self.similar))
        self.precision = 100*(len(common) / len(found_result))
        if self.recall == 0 and self.precision == 0:
            self.f1 = 0
        else:
            self.f1 = (2*self.recall*self.precision) / (self.recall + self.precision)

    def saveStats(self, size_mem, time): # save all stats in result.txt file
        f = open("result.txt", "w")
        f.write("Result of the survey\n")
        f.write(f"Memory size: {size_mem} bytes\n")
        f.write(f"Runtime: {time}\n")
        f.write(f"Recall = {self.recall}%\n")
        f.write(f"Precision = {self.precision}%\n")
        f.write(f"F1 = {self.f1}%\n")
