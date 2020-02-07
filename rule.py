class RulesContainer:
    def __init__(self, tree):
        self.listOfRules = self.toRules(tree)
        
    def toRules(self,tree):
        if (tree.children):
            conditions = []
            for child in tree.children:
                condition = Condition(tree.name, child[0])
                for childrenCondition in (self.toRules(child[1])):
                    childrenCondition.append(condition)
                    conditions.append(childrenCondition)
            return conditions
        else:
            return [[Condition(tree.name)]]
    
    def printRules(self):
        rulecopy = self.listOfRules.copy()
        for rules in rulecopy:
            ans = rules.pop(0).label
            print("if ",end="")
            for condition in rules:
                print(condition.label, end="=")
                print(condition.value, end="")
                if (condition == rules[len(rules)-1]):
                    print(" ", end="")
                else:
                    print(" ",end="and ")
            print("then ", end="")
            print(ans)

class Condition:
    def __init__(self,label,value=""):
        self.label = label
        self.value = value
    
    def __str__(self):
        return (self.label + "=" + str(self.value))
    
    def __repr__(self):
        return (self.label + "=" + str(self.value))