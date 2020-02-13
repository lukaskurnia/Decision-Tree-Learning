class Tree:
    def __init__(self, name):
        self.name = name
        self.children = []
    
    def isEmpty(self):
        return self.children == []
    
    def addChild(self, val, child):
        self.children.append([val, child])

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