import pandas as pd
import numpy as np 
from tree import Tree
from math import log

# Constant class for continuous value
HIGH = 'high'
LOW = 'low'

class DecisionTreeLearning:
    # ATTRIBUTE
    # 
    # df, full dataframe
    # target, target attribute
    # tree, decision tree
    # attributes, list of attributes with its unique value
    # gainRatio (default = false), set true to use gain ratio to select max attribute
    # pruned (default = false), set true to use post-pruning rule

    # constructor
    def __init__(self, filename, target, gainRatio=False, pruned=False):
        self.readCsv(filename, target)
        self.gainRatio = gainRatio
        self.pruned = pruned

    # read file csv with given filename in folder data
    def readCsv(self, filename, target):
        self.df = pd.read_csv('data/' + filename + '.csv')
        self.target = target
        self.attributes = {}
        for col in self.df.columns:
            if not(self.isContinuous(col)):
                self.attributes[col] = self.df[col].unique().tolist()
            else:
                self.attributes[col] = [HIGH, LOW]

    def isContinuous(self, attr):
        return not ((self.df[attr].dtypes == 'bool') or (self.df[attr].dtypes == 'object'))

    # build tree by current dataframe
    def build(self):
        self.tree = self._build(self.df)

    # private method for build tree recursively
    def _build(self, df):
        if df.shape[1] == 1:
            # empty attribute 
            mode = self.df[self.target].mode()[0]
            return Tree(mode)

        if len(df[self.target].unique()) == 1:
            # entropy = 0 
            return Tree(df[self.target][0])

        attr = self.getMaxGainAttr(df)
        tree = Tree(attr)
        
        if self.isContinuous(attr) == True:
            treshold = self.getBestTreshold(df, attr)
            lowDf, highDf = self.splitHorizontalContinuous(df, attr, treshold)
            
            childLowDf = self.dropAttr(lowDf, attr)
            childLowTree = self._build(childLowDf)
            tree.addChild('< ' + str(treshold) , childLowTree)

            childHighDf = self.dropAttr(highDf, attr)
            childHighTree = self._build(childHighDf)
            tree.addChild('>= ' + str(treshold) , childHighTree)

            return tree

        for value in self.attributes[attr]:
            splittedDf = self.splitHorizontalKeepValue(df, attr, value)

            if splittedDf.empty == True:
                # empty example 
                originalDf = self.splitHorizontalKeepValue(self.df, attr, value)
                mode = originalDf[self.target].mode()[0]
                childTree = Tree(mode)
            else:
                childDf = self.dropAttr(splittedDf, attr)
                childTree = self._build(childDf)
            tree.addChild(value, childTree)
        
        return tree

    def printTree(self):
        print(self.tree)
        
    def entropy(self, df, attr):
        #dataframe row
        row = df.shape[0]

        #get unique value
        unique = df[attr].unique().tolist()
        # print(df[attr])

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

    def gainRatio(self, df, attr):
        e = self.entropy(df, self.target)
        return self.infoGain(df,attr,e)/self.entropy(df,attr) 
                
    def getMaxGainAttr(self, df):
        df = self.makeDiscrete(df)
        features = df.columns.values
        # print(features)
        features = features[features!=self.target]
        # print(features)
        gains = []
        for feature in features:
            gains.append(self.infoGain(df, feature, self.entropy(df, self.target)))
        
        #get index of max attributes
        maxindex = 0
        for i in range (1,len(gains)):
            if (gains[maxindex]<gains[i]):
                maxindex = i

        return features[maxindex]
    
    def splitHorizontalKeepValue(self, df, attr, val):
        newdf = df[df[attr]==val]
        return newdf.reset_index(drop=True)

    # return 2 dataframes, by a treshold value
    def splitHorizontalContinuous(self, df, attr, val):
        lowDf = df[df[attr] < val].reset_index(drop=True)
        highDf = df[df[attr] >= val].reset_index(drop=True)
        return lowDf, highDf

    def splitHorizontalDiscardValue(self, df, attr, val):
        newdf = df[df[attr]!=val]
        return newdf.reset_index(drop=True)
    
    # return 2 dataframes, first dataframe size is given percetage of original set, 
    # and the second is the rest of it
    def splitByPercentage(self, percentage):
        idx = (round(self.df.shape[0]*percentage/100))
        return np.split(self.df, [idx])
        
    def dropAttr(self, df, attr):
        return df.drop(columns=attr)

    def sortValue(self, df, attr):
        return df.sort_values(by=[attr])

    #handling missing value: get the most frequent value with the same target
    def handling_missing_value(self):
        if self.df.isnull().values.any():
         missing_columns = self.df.columns[self.df.isna().any()].tolist()
        for col in missing_columns:
            mode = self.df.mode()[col][0]
            rows = pd.isnull(self.df).any(1).nonzero()[0].tolist()
            self.df[col][rows]=mode

    def getAllTreshold(self, df, attr):
        sortedDf = self.sortValue(df,attr)
        # print(sortedDf)
        listClass = sortedDf[self.target].values
        # print(listClass)
        listCandidateForC = []
        i = 0
        while(i < len(listClass)-1):
            if(listClass[i] != listClass[i+1]):
                listAttr = sortedDf[attr].values
                # print(listAttr[i])
                # print(listAttr[i+1])
                listCandidateForC.append((listAttr[i]+listAttr[i+1])/2)
                # print(i)
                # print(i+1)
            # print(listClass[i])
            i += 1
        return listCandidateForC
        # for i in range(len(listClass)):

    def infoGainContinuous(self, df, attr, entropy, treshold):
        sortedDf = self.sortValue(df,attr)
        row = sortedDf.shape[0]
        gain = entropy
        # print("ini entropy " , gain)
        dfLessThanTreshold = sortedDf[sortedDf[attr] < treshold]
        dfGreaterThanTreshold = sortedDf[sortedDf[attr] >= treshold]
        less = self.entropy(dfLessThanTreshold, self.target)
        greater = self.entropy(dfGreaterThanTreshold, self.target)
        # print(dfLessThanTreshold.shape[0]/row * less + dfGreaterThanTreshold.shape[0]/row * greater)
        gain -= (dfLessThanTreshold.shape[0]/row * less + dfGreaterThanTreshold.shape[0]/row * greater)
        return gain

    def getBestTreshold(self, df, attr):
        candidate = self.getAllTreshold(df,attr)
        # print(candidate)
        gains = []
        for value in candidate:
            gains.append(self.infoGainContinuous(df, attr, self.entropy(df, self.target), value))
        
        # print(gains)
        #get index of max attributes
        maxindex = 0
        for i in range (1,len(gains)):
            if (gains[maxindex]<gains[i]):
                maxindex = i

        return candidate[maxindex]

    def makeDiscrete(self, df):
        newDf = df.copy()
        for col in newDf.columns:
            if(self.isContinuous(col)):
                treshold = self.getBestTreshold(df,col)
                row = newDf[col].shape[0]
                for i in range(0, row):
                    if(newDf[col][i] < treshold):
                        newDf[col][i] = LOW
                    else:
                        newDf[col][i] = HIGH
        return newDf

    



    


    
