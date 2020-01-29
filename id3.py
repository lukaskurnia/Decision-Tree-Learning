import pandas as pd
from math import log
    
class DecisionTreeLearning:
    # ATTRIBUTE
    # 
    # df, full dataframe
    # target, target attribute

    # constructor
    def __init__(self):
        print("hello world")
        self.readCsv('play')

    # read tennis.csv
    def readCsv(self, target):
        self.df = pd.read_csv('data/tennis.csv')
        self.target = target
        
    # print tennis.csv
    def printCsv(self):
        print(self.df)

    # build tree by current dataframe
    def build(self):
        self.Tree = _build(self.df)

    # private method for build tree recursively
    def _build(self, df):
        currentAttr = getMaxGainAttr(self.df)

    def entropy(self, df, attr):
        #dataframe row
        row = df.shape[0]

        #get unique value
        unique = df[attr].unique().tolist()

        entropy = 0
        for u in unique:
            #get occurence
            freq = (df[attr]==u).sum()
            entropy += (-1) * freq/row * log((freq/row),2)

        return entropy

    def infoGain(self, df, attr, entropy):
        #dataframe row
        row = df.shape[0]

        #get unique value
        unique = df[attr].unique().tolist()

        gain = entropy
        for u in unique:
            #get occurence
            freq = (df[attr]==u).sum()
            newdf = self.splitHorizontalKeepValue(df, attr, u)
            e = self.entropy(newdf, self.target)
            gain -= freq/row * e
        
        return gain
        
        
    def getMaxGainAttr(self, df):
        features = self.dropAttr(df)
        gains = []
        for feature in features:
            gains.append(self.infoGain(df, feature, self.entropy(df, self.target)))
        
        return max(gains)
    
    def splitHorizontalKeepValue(self, df, attr, val):
        newdf = df[df[attr]==val]
        return newdf
        
    def dropAttr(self, df, attr):
        return df.drop(columns=attr)