from id3 import DecisionTreeLearning

dtl = DecisionTreeLearning("tennis", "play", isPrune=True)
# print(dtl.splitHorizontalKeepValue(dtl.df, 'outlook', 'sunny'))
# e = dtl.entropy(dtl.splitHorizontalKeepValue(dtl.df, 'outlook', 'sunny'), 'play')
# print(e)
# g = dtl.infoGain(dtl.df, 'windy', e)
# print(g)
# a = dtl.gainRatio(dtl.df, 'humidity')
# print(a)
# attr = dtl.getMaxGainAttr(dtl.splitHorizontalKeepValue(dtl.df, 'outlook', 'sunny'))
# print(attr)

# dtl.test()

dtl.build()

dtl.printTree()

# dtl.printCsv()
# sort = dtl.sortValue(dtl.df,'sepal_width')
# print(sort)

# print(dtl.df.dtypes)

# print(dtl.df['windy'].dtypes == 'bool')