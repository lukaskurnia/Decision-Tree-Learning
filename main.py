from id3 import DecisionTreeLearning

dtl = DecisionTreeLearning("tennis", "play", isPrune=True)
dtl.build()
dtl.printTree()

if (dtl.isPrune):
    print(dtl.rules)
