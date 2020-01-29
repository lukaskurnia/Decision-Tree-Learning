from id3 import DecisionTreeLearning

classifier = DecisionTreeLearning()
# print(classifier.splitHorizontalKeepValue(classifier.df, 'play', 'yes'))
e = classifier.entropy(classifier.df, 'play')
print(e)
g = classifier.infoGain(classifier.df, 'windy', e)
print(g)
