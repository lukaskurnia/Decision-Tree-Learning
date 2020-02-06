class Tree:
    def __init__(self, name):
        self.name = name
        self.children = []
    
    def isEmpty(self):
        return self.children == []
    
    def addChild(self, val, child):
        self.children.append([val, child])

    def toRules(self):
        if (self.children):
            conditions = []
            for child in self.children:
                condition = (self.name, child[0])
                print(condition)
                for childrenCondition in (child[1].ruleBuilder()):
                    childrenCondition.append(condition)
                    conditions.append(childrenCondition)
            return conditions
        else:
            return [[self.name]]

    def __str__(self, level=0, value=''):
        if (level == 0):
            ret = repr(self.name)+"\n"
        else:
            ret = "   "*(level-1)+"|--"+repr(self.name)+" ("+ str(value) +")\n"
        for child in self.children:
            ret += child[1].__str__(level=level+1, value=child[0])
        return ret

    def __repr__(self):
            return '<tree node representation>'
        
# tree = Tree('afd')
# treeb = Tree('bfdsaf')
# treec = Tree('cdasdas')
# treed = Tree('ddasdas')
# treee = Tree('edasdas')
# treef = Tree('fdasdas')
# treeg = Tree('gdasdas')
# treeh = Tree('hdasdas')
# treei = Tree('i')
# treej = Tree('j')
# treek = Tree('k')
# treel = Tree('l')
# treem = Tree('m')
# treen = Tree('n')
# treeo = Tree('o')
# treep = Tree('p')
# treeq = Tree('q')

# tree.addChild('b-val',treeb)
# tree.addChild('c-val',treec)
# treeb.addChild('d-val',treed)
# treec.addChild('f-val',treef)
# treec.addChild('g-val',treeg)
# treec.addChild('h-val',treeh)
# treed.addChild('i-val',treei)
# tree.addChild('wakgeng', treej)
# tree.addChild('coba', treek)
# treej.addChild('cobalagi', treel)
# treeb.addChild('cobaocba', treem)

# print(tree)