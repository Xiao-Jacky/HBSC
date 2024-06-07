from entity import OrderDict, Triplet
from BlockType import BlockType
from BasicBlock import BasicBlock

class Cfg:

    __slots__ = ['_name','_basicBlocks','_length', '_dispatchers','_fallbacks', '_functions','_loaders','_tooshort']

    def __init__(self,name,basicBlocks):
        self._name = name
        self._basicBlocks :OrderDict = basicBlocks
        self._length = self._basicBlocks.length
        self._dispatchers = []
        self._fallbacks = []
        self._functions = []
        self._loaders = []
        self._tooshort = False

    def loaddata(self,dispatcher,fallbacks,functions,loaders,short):
        self._dispatchers = dispatcher
        self._fallbacks = fallbacks
        self._functions = functions
        self._loaders = loaders
        self._tooshort = short

    def storetxt(self):
        basicblocks:BasicBlock = self.basicBlocks
        text = self.name + '\n'
        for pair in basicblocks:
            basicblock:BasicBlock = pair.value
            text = text + str(basicblock)+'\n'
            for inst in basicblock.instructions:
                text = text + hex(inst.pc) + ' ' + str(inst) + '\n'
            if basicblock.hasSuccessor():
                text = text + 'successor: ' + str(basicblock.successors) + '\n'
        path = txtpath + self.name + ".txt"
        with open(path,"w")as f:
            f.write(text)

    def storejson(self):
        basicblocks:BasicBlock = self.basicBlocks
        basicblocksdict = {"contractname":self.name, 
                           "basicblocks":[],
                           "dispatchers":[],
                           "fallbacks":[],
                           "functions":[],
                           "loaders":[],
                           "short":None}
        for pair in basicblocks:
            offset:int = pair.key
            basicblock:BasicBlock = pair.value
            ins = []
            for i in basicblock.instructions:
                ins.append({"pc":i.pc,"opname":i.name,"operand":i.operand})
            pres = []
            succs = []
            for p in basicblock.predecessors:
                pres.append(p.offset)
            for s in basicblock.successors:
                succs.append(s.offset)
            bbdict={"offset":offset,
                    "instructions":ins,
                    "predecessors":pres,
                    "successors":succs,
                    "stackbalance":basicblock.stackbalance,
                    "blocktype":basicblock.type.name}
            basicblocksdict["basicblocks"].append(bbdict)
            basicblocksdict["dispatchers"] = self.dispatchers
            basicblocksdict["fallbacks"] = self.fallbacks
            basicblocksdict["functions"] = self.functions
            basicblocksdict["loaders"] = self.loaders
            basicblocksdict["short"] = self.short
        return basicblocksdict

    def storedot(self):
        basicblocks:BasicBlock = self.basicBlocks
        filestr = 'digraph{\n'
        for pair in basicblocks:
            basicblock:BasicBlock = pair.value
            newstr = '{}[label="{}"]\n'.format(basicblock.offset,basicblock)
            filestr += newstr
            for successor in basicblock.successors:
                filestr += '{} -> {}\n'.format(basicblock.offset, successor.offset)
        filestr += '\n}'
        path = dotpath + self.name + ".dot"
        with open(path,"w")as f:
            f.write(filestr)

    def printblocks(self):
        for i in self._basicBlocks:
            print(i)
    
    def tooshort(self):
        self._tooshort = True

    def add_dispatcher(self,offset):
        if offset not in self._dispatchers:
            self._dispatchers.append(offset)

    def add_loader(self,offset):
        if offset not in self._loaders:
            self._loaders.append(offset)
    
    def add_fallback(self,offset):
        if offset not in self._fallbacks:
            self._fallbacks.append(offset)

    def add_function(self,offset):
        if offset not in self._functions:
            self._functions.append(offset)

    @property
    def short(self):
        return self._tooshort

    @property
    def functions(self):
        return self._functions
    
    @property
    def loaders(self):
        return self._loaders

    @property
    def dispatchers(self):
        return self._dispatchers
    
    @property
    def dispatcher_num(self):
        return len(self._dispatchers)
    
    @property
    def fallbacks(self):
        return self._fallbacks
    
    @property
    def fallback_num(self):
        return len(self._fallbacks)

    @property
    def name(self):
        return self._name

    @property
    def basicBlocks(self):
        return self._basicBlocks
    
    @property
    def basicblocks(self):
        return self._basicBlocks
    
    @property
    def start(self):
        return self._basicBlocks[0]
    
    @property
    def length(self):
        return self._length

    
    



    
        