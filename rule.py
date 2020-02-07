class Rules:
    def __init__(self, tree):
        self.listOfRules = self.toRules(tree)
        
    def toRules(self,tree):
        if (tree.children):
            conditions = []
            for child in tree.children:
                condition = Rule(tree.name, child[0])
                for childrenCondition in (self.toRules(child[1])):
                    childrenCondition.append(condition)
                    conditions.append(childrenCondition)
            return conditions
        else:
            return [[Rule(tree.name)]]
    
    def printRules(self):
        rulecopy = self.listOfRules.copy()
        for rules in rulecopy:
            ans = rules.pop(0).label
            print("if ",end="")
            for rule in rules:
                print(rule.label, end="=")
                print(rule.value,end=", ")
            print("then ", end="")
            print(ans)

class Rule:
    def __init__(self,label,value=""):
        self.label = label
        self.value = value