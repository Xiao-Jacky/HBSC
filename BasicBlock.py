from pyevmasm.evmasm import Instruction
from BlockType import BlockType

class BasicBlock:
    __slots__ = ['_offset', '_instructions', '_predecessors', '_successors', '_stackBalance', '_blocktype',
                 '_hascaller', '_retain']

    def __init__(self, offset):
        self._offset = offset            # 基本块偏移量
        self._instructions = []          # 存储块中的指令列表
        self._predecessors = []          # 存储前驱块的列表
        self._successors = []            # 存储后继块的列表
        self._stackBalance = 0
        self._blocktype = BlockType.UNDEFINED
        self._hascaller = None
        self._retain = True

    def calculateStackBalance(self) -> int:
        balance = 0
        for o in self._instructions:
            balance -= o.pops
            balance += o.pushes
        return balance

    def add_instruction(self, instruction):
        self._instructions.append(instruction)
        if hasattr(instruction, "pops"):
            self._stackBalance -= instruction.pops
            self._stackBalance += instruction.pushes

    def add_All(self, instructions: list):
        self._instructions.extend(instructions)
        self._stackBalance = self.calculateStackBalance()

    def __repr__(self):
        if self.type == BlockType.STOP:
            return '<STOP BLOCK>'
        return '<cfg BasicBlock@{:x}-{:x}> {}'.format(self.start.pc, self.end.pc, self.type)

    def __str__(self) -> str:
        if self.type == BlockType.STOP:
            return '<STOP BLOCK>'
        return '<cfg BasicBlock@{:x}-{:x}> {}'.format(self.start.pc, self.end.pc, self.type)

    def getInstruction(self, index):
        if index < self.length and index >= self.length * -1:
            return self._instructions[index]

    @property
    def offset(self):
        return self._offset

    def isEmpty(self):
        return self.length == 0

    @property
    def instructions(self):
        return self._instructions

    @property
    def length(self):
        return len(self._instructions)

    @property
    def start(self) -> Instruction:
        if self.length:
            return self._instructions[0]

    @property
    def end(self) -> Instruction:
        if self.length:
            return self._instructions[-1]

    @property
    def lastsecond(self) -> Instruction:
        if self.length > 1:
            return self._instructions[-2]
        return None

    @property
    def stackbalance(self):
        return self._stackBalance

    @property
    def type(self):
        return self._blocktype

    def setType(self, type: BlockType):
        self._blocktype = type

    @property
    def retain(self):
        return self._retain

    def setretain(self, flag: bool):
        self._retain = flag

    # def getPredecessor(self, key):
    #     return self._predecessors.get(key, [])

    # def getSuccessors(self, key):
    #     return self._successors.get(key, [])

    @property
    def predecessors(self):
        return self._predecessors

    @property
    def successors(self):
        return self._successors

    def hasSuccessor(self):
        return len(self._successors) > 0

    def hasPredecessor(self):
        return len(self._predecessors) > 0

    def add_successor(self, next: 'BasicBlock'):
        if next.offset == self.offset or next.offset == 0:
            return
        if next not in self._successors:
            self._successors.append(next)
        if self not in next._predecessors:
            next._predecessors.append(self)

    def checkcaller(self):
        for i in self._instructions:
            if "CALL" in i.name:
                self._hascaller = True
                return
        self._hascaller = False

    @property
    def ends_with_jump(self):
        return self.end.name == 'JUMP'

    @property
    def ends_with_jumpi(self):
        return self.end.name == 'JUMPI'

    @property
    def ends_with_jump_or_jumpi(self):
        return self.end.is_branch()

    @property
    def is_terminator(self):
        return self.end.is_terminator

    def has_caller(self):
        if self._hascaller == True:
            return True
        return False

    def equals(self, other) -> bool:
        if isinstance(other, BasicBlock):
            if id(self) == id(other): return True
            if other == None: return False
            return self.offset == other.offset
        return False

