from id3 import DecisionTreeLearning

dtl = DecisionTreeLearning()
# print(dtl.splitHorizontalKeepValue(dtl.df, 'outlook', 'sunny'))
# e = dtl.entropy(dtl.splitHorizontalKeepValue(dtl.df, 'outlook', 'sunny'), 'play')
# print(e)
# # g = dtl.infoGain(dtl.df, 'windy', e)
# # print(g)
# attr = dtl.getMaxGainAttr(dtl.splitHorizontalKeepValue(dtl.df, 'outlook', 'sunny'))
# print(attr)

# dtl.test()

dtl.build()
dtl.printTree()
