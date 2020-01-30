import pandas as pd
from tree import Tree
from math import log
    
class DecisionTreeLearning:
    # ATTRIBUTE
    # 
    # df, full dataframe
    # target, target attribute
    # tree, decision tree

    # constructor
    def __init__(self):
        print("hello world")
        self.readCsv('species')

    # read tennis.csv
    def readCsv(self, target):
        self.df = pd.read_csv('data/iris.csv')
        self.target = target
        
    # print tennis.csv
    def printCsv(self):
        print(self.df)

    # build tree by current dataframe
    def build(self):
        self.tree = self._build(self.df)

    # private method for build tree recursively
    def _build(self, df):
        print(df)
        attr = self.getMaxGainAttr(df)
        if self.entropy(df, attr) == 0:
            return Tree(df[self.target][0])
        
        tree = Tree(attr)
        values = df[attr].unique().tolist()
        for value in values:
            print('attribut :', attr)
            print('value :', value)
            splittedDf = self.splitHorizontalKeepValue(df, attr, value)
            print('splittedDf')
            print(splittedDf)
            childDf = self.dropAttr(splittedDf, attr)
            print('childDf')
            print(childDf)
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
        return newdf

    def splitHorizontalDiscardValue(self, df, attr, val):
        newdf = df[df[attr]!=val]
        return newdf
    
    def dropAttr(self, df, attr):
        return df.drop(columns=attr)

    def sortValue(self, df, attr):
        return df.sort_values(by=[attr])

    


    