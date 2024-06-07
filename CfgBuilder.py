from pyevmasm.evmasm import Instruction
from Cfg import Cfg
from BlockType import BlockType
from BasicBlock import BasicBlock
from SymbolicStack import SymbolicStack
from entity import OrderDict, Triplet, Stack, Logger, Instruct
import json
from pyevmasm.evmasm import disassemble_all

class CfgBuilder:
    LOOP_DEPTH = 1000
    REMOVE_ORPHAN_BLOCKS = True
    BLOCK_LIMIT = 200000

    def __init__(self):
        self.log = Logger("2023-5-15")

    def buildCfg(self, name, bytecode) -> Cfg:
        if bytecode == '':
            return None

        self.name = name

        opcode = self.disassemble(bytecode)
        blocks = self.generateBasicBlocks(opcode)
        self.calculateSuccessors(blocks)
        self.resolveOrphanJumps(blocks)

        self.removeOrphanBlocks(blocks)
        self.addSuperStop(blocks)

        return Cfg(name, blocks)

    # Disassembling bytecode into opcodes
    def disassemble(self, bytecode):
        if isinstance(bytecode, str):
            bytecode = bytecode.replace('\n', '')
        if bytecode.startswith('0x'):
            bytecode = bytes.fromhex(bytecode[2:])
        else:
            bytecode = bytes.fromhex(bytecode)
        return disassemble_all(bytecode)


    # Divide the opcode into basic blocks
    def generateBasicBlocks(self, opcode) -> OrderDict:
        result = OrderDict()
        current = BasicBlock(0)
        for op in opcode:
            if op.is_terminator:
                current.add_instruction(op)
                result.update(current.offset, current)
                current = BasicBlock(op.pc + 1)
            elif op.semantics == "JUMPDEST" and not current.isEmpty():
                result.update(current.offset, current)
                current = BasicBlock(op.pc)
                current.add_instruction(op)
            else:
                current.add_instruction(op)
        if not current.isEmpty():
            result.update(current.offset, current)
        return result

    # Build connections between basic blocks
    def calculateSuccessors(self, basicblocks: 'OrderDict'):
        for pair in basicblocks:
            offset = pair.key
            basicblock: BasicBlock = pair.value
            ins: list = basicblock.instructions
            lastin: Instruction = basicblock.end
            # JUMP
            if lastin.name == "JUMP" and len(ins) > 1:
                # PUSH
                lastsecond: Instruction = ins[-2]
                if lastsecond.group == "Push Operations":
                    jumpoffset = lastsecond.operand
                    if jumpoffset in basicblocks.keys():
                        basicblock.add_successor(basicblocks.get(jumpoffset))
                    else:
                        self.log.addDirectJumpTargetErrors(self.name, offset, jumpoffset)
                else:
                    pass
            # JUMPI
            elif lastin.name == "JUMPI" and len(ins) > 1:
                # NEXT
                nextoffset = lastin.pc + lastin.size
                nextblock: BasicBlock = basicblocks.get(nextoffset)
                if (nextblock is not None):
                    basicblock.add_successor(nextblock)
                # PUSH
                lastsecond: Instruction = ins[-2]
                if lastsecond.group == "Push Operations":
                    jumpoffset = lastsecond.operand
                    if jumpoffset in basicblocks.keys():
                        basicblock.add_successor(basicblocks.get(jumpoffset))
                    else:
                        self.log.addDirectJumpTargetErrors(self.name, offset, jumpoffset)

            # other terminator
            elif lastin.is_terminator:
                pass

            # last instruction
            elif offset == basicblocks.end.key:
                pass
            # ELSE commom block next
            else:
                jumpoffset = lastin.pc + lastin.size
                if jumpoffset in basicblocks.keys():
                    basicblock.add_successor(basicblocks.get(jumpoffset))

    # Dynamic execution of simulation stacks
    def resolveOrphanJumps(self, basicblocks: 'OrderDict'):
        blockcount = 0
        errors = dict()
        visited = set()
        current: BasicBlock = basicblocks.start.value
        stack: SymbolicStack = SymbolicStack()
        dfs_depth = 0
        queue = Stack()
        queue.push(Triplet(current, stack, dfs_depth))
        while not queue.isEmpty():
            element: Triplet = queue.pop()
            current: BasicBlock = element.elem1
            stack: SymbolicStack = element.elem2
            dfs_depth = element.elem3
            # print(current)

            # execuate all instructions in block except the last one
            for i in range(current.length - 1):
                op: Instruction = current.getInstruction(i)
                # print(op,stack)
                try:
                    stack.execuate(op)
                except Exception as e:
                    self.log.addStackExceededErrors(self.name, current.offset, op.pc)
                    if current.offset in errors:
                        errors[current.offset] += 1
                    else:
                        errors[current.offset] = 1

            lastin: Instruction = current.end
            nextoffset = 0

            # check orphan jump and reslove
            if lastin.name == "JUMP":
                try:
                    nextoffset = stack.peek()
                    if nextoffset != 0 and nextoffset != None:
                        nextbb: BasicBlock = basicblocks.get(nextoffset)
                        if nextbb is not None:
                            current.add_successor(nextbb)
                        else:
                            self.log.addOrphanJumpTargetNullErrors(self.name, current.offset, nextoffset)
                    else:
                        raise Exception()
                except:
                    self.log.addOrphanJumpTargetUnknownErrors(self.name, current.offset)

            # execute last instruction
            blockcount += 1
            if blockcount >= CfgBuilder.BLOCK_LIMIT:
                self.log.addBlockLimitErrors(self.name, CfgBuilder.BLOCK_LIMIT)
                return
            try:
                stack.execuate(lastin)
                # print(stack)
            except Exception as e:
                self.log.addStackExceededErrors(self.name, current.offset, lastin.pc)
                if current.offset in errors:
                    errors[current.offset] += 1
                else:
                    errors[current.offset] = 1

            # Add next elements for DFS

            if dfs_depth < CfgBuilder.LOOP_DEPTH:
                # next block
                if lastin.name != "JUMP":
                    for successor in current.successors:
                        edge = Triplet(current.offset, successor.offset, stack)
                        if (edge not in visited) and ((current.offset not in errors) or errors[current.offset] < 50):
                            visited.add(edge)
                            queue.push(Triplet(successor, stack.copy(), dfs_depth + 1))

                # get the orphan jump offset
                elif nextoffset != 0:
                    edge = Triplet(current.offset, nextoffset, stack)
                    if edge not in visited:
                        visited.add(edge)
                        nextbb: BasicBlock = basicblocks.get(nextoffset)
                        if (nextbb is not None) and ((current.offset not in errors) or errors[current.offset] < 50):
                            queue.push(Triplet(nextbb, stack.copy(), dfs_depth + 1))
            else:
                self.log.addLoopDepthExceededErrors(self.name, current.offset)
                if current.offset in errors:
                    errors[current.offset] += 1
                else:
                    errors[current.offset] = 1

    def removeOrphanBlocks(self, basicblocks: 'OrderDict'):
        candidateOffset = 0
        visited = set()
        queue = Stack()
        queue.push(basicblocks.start.value)
        while not queue.isEmpty():
            candidate: BasicBlock = queue.pop()
            # candidate.setType(BlockType.COMMON)
            visited.add(candidate)
            candidateOffset = max(candidateOffset, candidate.offset)
            for successor in candidate.successors:
                if successor not in visited:
                    queue.push(successor)

        # remove higher offset blocks
        offsetlist = list(basicblocks.keys())
        for offset in offsetlist:
            if offset > candidateOffset:
                basicblocks.remove(offset)

        # remove other blocks
        removelist = []
        for of, block in basicblocks:
            if block not in visited:
                removelist.append(of)
        for offset in removelist:
            basicblocks.remove(offset)

    def addSuperStop(self, basicblocks: 'OrderDict'):
        stop = BasicBlock(-1)
        stop.setType(BlockType.STOP)
        for pair in basicblocks:
            offset = pair.key
            basicblock: BasicBlock = pair.value
            if not basicblock.hasSuccessor():
                basicblock.add_successor(stop)
        basicblocks.update(-1, stop)

    def rebuildCfg(self, blockjson):
        blockdict = json.loads(blockjson)
        name = blockdict["contractname"]
        blocks = blockdict["basicblocks"]
        basicblocks = OrderDict()
        for block in blocks:
            offset = block["offset"]
            instructions = block["instructions"]
            newblock = BasicBlock(offset)
            for instruction in instructions:
                ins = Instruct(instruction["pc"], instruction["opname"], instruction["operand"])
                newblock.add_instruction(ins)
            newblock.setType(BlockType[block["blocktype"]])
            basicblocks.update(offset, newblock)
        for block in blocks:
            offset = block["offset"]
            sucs = block["successors"]
            current: BasicBlock = basicblocks.get(offset)
            for nextoffset in sucs:
                next = basicblocks.get(nextoffset)
                if next is not None:
                    current.add_successor(next)
        cfg = Cfg(name, basicblocks)
        cfg.loaddata(blockdict["dispatchers"], blockdict["fallbacks"], blockdict["functions"], blockdict["loaders"],
                     blockdict["short"])
        return cfg



