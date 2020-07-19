#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools


class Loop:
    numloopssofar = 0

    @classmethod
    def createLoop(cls,name:str=None):
        if not name:
            Loop.numloopssofar += 1
            name = f'loop{Loop.numloopssofar}'
        return cls(isroot=False,name=name)

    @classmethod
    def createRoot(cls):
        return cls(isroot=True,name=None)

    def __init__(self,isroot:bool,name:str):
      self.isroot = isroot
      if not isroot:
        self.name = name
      self.subloops = []

    def perfectnest(self):
        result = [self]
        while True:
            if len(result[-1].subloops)!=1:
                break
            result.append(result[-1].subloops[0])
        return result

    def new_subloop(self):
        newloop =  Loop.createLoop()
        self.subloops.append(newloop)
        return newloop

    def add_subloop(self,subloop):
        self.subloop.append(subloop)
        return subloop

    def clone(self):
        if self.isroot:
            result = Loop(isroot=True,name=None)
        else:
            result = Loop(isroot=False,name=self.name)
        result.subloops = self.subloops.copy()
        return result

    def __str__(self) -> str :
        return '\n'.join(self.to_lines(0))

    def to_lines(self,indent:int = 0):
        block = False
        subindent = indent
        if not self.isroot:
            block = len(self.subloops) > 1
            yield "    "*indent + f"#pragma clang loop id({self.name})"
            yield "    "*indent + "for (...)" + (" {" if block else "")
            subindent += 1

        if self.subloops:
            for subloop in self.subloops:
                yield from subloop.to_lines(indent=subindent)
        else:
            yield "    "*subindent + "code;"

        if block:
            yield "    "*indent + "}"


# Replace oldloop with newloop in loop nest while cloning all loop to the path there
def gist(root,childids,oldloop,newloop):
    if not childids:
        assert root==oldloop
        return newloop

    newroot = root.clone()
    cur = newroot
    for z in childids[:-1]:
        n = cur.subloops[z].clone()
        cur.subloops[z] = n
        cur = n

    lastid = childids[-1]
    assert cur.subloops[lastid] == oldloop
    cur.subloops[lastid] = newloop
    return newroot


def transform_node(root, callback):
  def do_transform_node(loop, childids,  callback):
    if not loop.isroot:
      for newsubtree in callback(loop):
            newroot = gist(root, childids, loop, newsubtree[0])
            yield tuple(list(newsubtree) + [newroot])
    for i,child in enumerate(loop.subloops):
        yield from do_transform_node(child, childids + [i], callback)
  yield from do_transform_node(root,[],callback)


class Experiment:
    def __init__(self,loopnest,pragmalist):
        self.loopnest = loopnest
        self.pragmalist = pragmalist
        self.derived_from = None
        self.derivitives = []
    
    def add_subexperiment(self,subexp):
        subexp.derived_from = self
        self.derivitives.append(subexp)

    def derivitives_recursive(self):
        yield self
        for n in self.derivitives:
            yield from n.derivitives_recursive()

    def __str__(self):
        return '\n'.join(self.to_lines())

    def to_lines(self):
        if self.pragmalist:
            return self.pragmalist
        else:
            return  self.loopnest.to_lines(0)
    

class Tiling:
    tilesizes = [2,4]

    @staticmethod
    def do_subtile(loop):
        # End tiling here
        yield [],[],loop.subloops,[]

        if len(loop.subloops) == 1:
             yield from Tiling.do_tile(loop.subloops[0])

    @staticmethod
    def do_tile(loop):
        for subfloors,subtiles,subbody,subsizes in Tiling.do_subtile(loop):
            for tilesize in Tiling.tilesizes:
                yield [Loop.createLoop()] + subfloors, [Loop.createLoop()] + subtiles, subbody, [tilesize] + subsizes

    @staticmethod
    def gen_tiling(loop: Loop):
        for floors,tiles,body,sizes in Tiling.do_tile(loop):
            cur = floors[0]
            for floor in floors[1:]:
                cur.subloops = [floor]
                cur = floor
            for tile in tiles:
                cur.subloops = [tile]
                cur = tile
            cur.subloops = body

            floorids = [floor.name for floor in floors]
            tileids = [tile.name for tile in tiles]
            sizes = [str(s) for s in sizes]
            pragma = f"#pragma clang transform tile on({loop.name}) sizes({','.join(sizes)}) floor_ids({','.join(floorids)}) tile_ds({','.join(tileids)})"
            yield floors[0],[pragma]


class Thrading:
    @staticmethod
    def gen_threading(loop: Loop):
        parallel_loop = Loop.createLoop()
        parallel_loop.subloops = loop.subloops
        pragma = f"#pragma clang transform parallelize_thread on({loop.name}) parallel_id({parallel_loop.name})"
        yield parallel_loop,[pragma]


class Interchange:
    @staticmethod
    def gen_interchange(loop: Loop):
        orignest = loop.perfectnest()
        for perm in itertools.permutations(orignest):
            if perm[0] == loop:
                continue

            nests = orignest
            while perm[-1] == nests[-1]:
                perm = perm[:-1]
                nests = nests[:-1]

            newperm = [Loop.createLoop() for l in perm]
            for p,c in zip(newperm,newperm[1:]):
                p.subloops = [c]
            newperm[-1].subloops = nests[-1].subloops
            
            permids = [p.name for p in perm]
            newpermids = [p.name for p in newperm]
            pragma = f"#pragma clang transform interchange on({nests[0].name}) permutation({','.join(permids)}) permuted_ids({','.join(newpermids)})"
            yield newperm[0],[pragma]


tiling_enabled = True
threading_enabled = True
interchange_enabled = True
def do_transformations(loop):
    if tiling_enabled:
        yield from Tiling.gen_tiling(loop)

    if threading_enabled:
        yield from Thrading.gen_threading(loop)

    if interchange_enabled:
        yield from Interchange.gen_interchange(loop)


def derive_expriments(baseexperiment: Experiment):
    oldroot = baseexperiment.loopnest # type: Loop
    for newsubloop,pragma,newroot in transform_node(oldroot, do_transformations):
        x = Experiment(newroot, baseexperiment.pragmalist + pragma)
        baseexperiment.add_subexperiment(x)


def expand_searchtree(baseexperiment: Experiment, remaining_depth: int):
    if remaining_depth <= 0:
        return
    derive_expriments(baseexperiment)
    for e in baseexperiment.derivitives:
        expand_searchtree(e,remaining_depth= remaining_depth-1)


def gen_input() -> Loop:
    example = Loop.createRoot()
    example.new_subloop()
    example.new_subloop().new_subloop()
    return example


def as_dot(baseexperiment: Experiment):
    yield "digraph G {"
    yield "rankdir=LR;"

    for experiment in baseexperiment.derivitives_recursive():
        desc = ''.join(l + "\\l" for l in experiment.to_lines())
        yield f'n{id(experiment)}[shape=box penwidth=2 fillcolor="azure:powderblue"style="filled" gradientangle=315 fontname="Calibri Light" label="{desc}"];'

        if parent := experiment.derived_from:
            yield f"n{id(parent)} -> n{id(experiment)};"
        yield ""

    yield "}"


def main() -> int:
    global tiling_enabled,threading_enabled
    parser = argparse.ArgumentParser(description="Loop transformation search tree proof-of-concept")
    parser.add_argument('--maxdepth', type=int, default=2)

    parser.add_argument('--tiling', action='store_true', default=True)
    parser.add_argument('--tiling-sizes', nargs='*', type=int, default=[2,4])

    parser.add_argument('--threading', action='store_true', default=True)

    parser.add_argument('--interchange', action='store_true', default=True)

    args = parser.parse_args()
    maxdepth  = args.maxdepth

    tiling_enabled = args.tiling
    Tiling.tilesizes = args.tiling_sizes

    threading_enabled = args.threading

    interchange_enabled = args.interchange

    example = gen_input()
    root = Experiment(example,[])
    expand_searchtree(root,remaining_depth=maxdepth)
    
    for line in as_dot(root):
        print(line)

    return 0


if __name__ == '__main__':
    if errcode := main():
        exit(errcode)
