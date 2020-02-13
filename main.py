from id3 import DecisionTreeLearning

dtl = DecisionTreeLearning("iris", "species", isPrune=True)
dtl.build()
dtl.printTree()

if (dtl.isPrune):
    print(dtl.rules)
