import pandas as pd
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

    # constructor
    def __init__(self):
        self.readCsv('coba', 'play')

    # read file csv with given filename in folder data
    def readCsv(self, filename, target):
        self.df = pd.read_csv('data/' + filename + '.csv')
        self.target = target
        self.attributes = {}
        for col in self.df.columns:
            if not(self.isContinuos(col)):
                self.attributes[col] = self.df[col].unique().tolist()
            else:
                self.attributes[col] = [HIGH, LOW]

    def isContinuos(self, attr):
        return not ((self.df[attr].dtypes == 'bool') or (self.df[attr].dtypes == 'object'))

    # build tree by current dataframe
    def build(self):
        self.tree = self._build(self.df)

    # private method for build tree recursively
    def _build(self, df):
        if df.shape[1] == 1:
            return Tree(df[self.target][0])
        if len(df[self.target].unique()) == 1:
            return Tree(df[self.target][0])

        attr = self.getMaxGainAttr(df)
        tree = Tree(attr)
        values = df[attr].unique().tolist()
        for value in values:
            splittedDf = self.splitHorizontalKeepValue(df, attr, value)
            childDf = self.dropAttr(splittedDf, attr)
            childTree = self._build(childDf)
            tree.addChild(value, childTree)
        
        return tree

    def __str__(self):
        print(self.tree)
        
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

    def gainRatio(self, df, attr):
        e = self.entropy(df, self.target)
        return self.infoGain(df,attr,e)/self.entropy(df,attr) 
                
    def getMaxGainAttr(self, df):
        features = df.columns.values
        features = features[features!=self.target]
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

    def splitHorizontalDiscardValue(self, df, attr, val):
        newdf = df[df[attr]!=val]
        return newdf.reset_index(drop=True)
    
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
