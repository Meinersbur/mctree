#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import csv
import datetime
import heapq
import itertools
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import tool.invoke as invoke
from tool.support import *


def mcount(seq):
    result = 0
    for count, _ in seq:
        result += count
    return result


def mselect(seq, idx: int):
    pos = idx
    for count, elt in seq:
        assert count >= 0
        if pos < count:
            return pos, elt
        pos -= count
    assert False, "Larger index than total elements"


def mcall(seq, loopcounter, idx: int):
    pos, elt = mselect(seq, idx)
    return elt(loopcounter, pos)


def neighbors(seq):
    it = iter(seq)
    prev = next(it)
    for item in it:
        yield prev, item
        prev = item


transformers = []


class Loop:
    @classmethod
    def createLoop(cls, loopcounter=None, name: str = None):
        if not name:
            name = f'loop{loopcounter.nextId()}'
        return cls(isroot=False, isstmt=False,isloop=True, istransformable=True, name=name)

    @classmethod
    def createStmt(cls):
        return cls(isroot=False, isstmt=True,isloop=False,istransformable=True,name=None)

    @classmethod
    def createAnonLoop(cls):
        return cls(isroot=False, isstmt=False,isloop=True, istransformable=False, name=None)

    @classmethod
    def createRoot(cls):
        return cls(isroot=True,isstmt=False,isloop=False, istransformable=False, name=None)

    def __init__(self, isroot: bool, isstmt: bool, isloop: bool, istransformable: bool, name: str):
        if isroot or isstmt or not istransformable:
            assert name == None

        self.isroot = isroot
        self.isstmt = isstmt
        self.isloop = isloop
        self.transformable = istransformable

        self.name = name
        self.subloops = []
        self.filename = None
        self.line = None
        self.column = None
        self.function = None

    def selector(self):
        looptransformers = (t(self) for t in transformers)
        for transformer in looptransformers:
            if transformer == None:
                continue
            yield transformer.get_num_children(), transformer.get_child

        def make_replace_loop_closure(i,subloop):
            def replace_loop(loopcounter, idx: int):
                newsubloop, pragmas = subloop.get_child(loopcounter, idx)
                assert isinstance(newsubloop,list), "Please return a list of replacement loops"
                newloop = self.clone()
                newloop.subloops = newloop.subloops[:i] + newsubloop + newloop.subloops[i+1:]
                #newloop.subloops[i] = newsubloop
                return [newloop], pragmas
            return replace_loop

        for i, subloop in enumerate(self.subloops):
            assert subloop == self.subloops[i]          
            yield subloop.get_num_children(), make_replace_loop_closure(i, subloop)

    def get_num_children(self):
        return mcount(self.selector())

    def get_child(self, loopcounter, idx: int):
        return mcall(self.selector(), loopcounter, idx)

    def perfectnest(self):
        assert self.transformable
        assert not self.isroot

        result = [self]
        while True:
            lastsubloops = result[-1].subloops
            if len(lastsubloops) != 1:
                break
            if not lastsubloops[0].isloop:
                break
            if not lastsubloops[0].transformable:
                break

            result.append(lastsubloops[0])
        return result

    def subloops_recursive(self):
        yield self
        for n in self.subloops:
            yield from n.subloops_recursive()

    def new_subloop(self,loopcounter):
        newloop = Loop.createLoop(loopcounter)
        self.subloops.append(newloop)
        return newloop

    def new_substmt(self):
        newstmt = Loop.createStmt()
        self.subloops.append(newstmt)
        return newstmt

    def add_subloop(self, subloop):
        self.subloops.append(subloop)
        return subloop

    def clone(self):
        if self.isroot:
            result = Loop(isroot=True , isstmt=False, isloop=False,  istransformable=False, name=None)
        elif self.isloop:
            result = Loop(isroot=False, isstmt=False, isloop=True,istransformable=self.transformable, name=self.name)
        elif self.isstmt:
            result = Loop(isroot=False , isstmt=True, isloop=False,  istransformable=False, name=None)
        result.subloops = self.subloops.copy()
        result.filename = self.filename
        result.line = self.line
        result.column = self.column
        result.function = self.function
        return result

    def __str__(self) -> str:
        return '\n'.join(self.to_lines(0))

    def to_lines(self, indent: int = 0):
        block = False
        subindent = indent
        if self.isroot:
            pass
        elif self.isloop:
            block = len(self.subloops) != 1
            yield "    "*indent + f"#pragma clang loop id({self.name})"
            loc = ""
            if self.filename:
                loc = f" /* {self.filename}:{self.line}:{self.column} */"
            yield "    "*indent + "for (...)" + (" {" if block else "") + loc
            subindent += 1
        elif self.isstmt:
            yield "    "*subindent + "stmt;"

        if self.subloops:
            for subloop in self.subloops:
                yield from subloop.to_lines(indent=subindent)

        if block:
            yield "    "*indent + "}"


# Replace oldloop with newloop in loop nest while cloning all loop to the path there
def gist(root, childids, oldloop, newloop):
    assert newloop

    if not childids:
        assert root == oldloop
        return newloop

    newroot = root.clone()
    cur = newroot
    for z in childids[:-1]:
        n = cur.subloops[z].clone()
        cur.subloops[z] = n
        cur = n

    lastid = childids[-1]
    assert cur.subloops[lastid] == oldloop
    if newloop == None:
        del cur.subloops[lastid]
    else:
        cur.subloops[lastid] = newloop
    return newroot


def json_to_loops(topmost, loopcounter):
    result = Loop.createRoot()
    for tm in topmost:
        kind = tm['kind']
        if kind == 'loop':
            loop = Loop.createLoop(loopcounter)
            loop.filename = mkpath(tm["path"])
            loop.line = tm["line"]
            loop.column = tm["column"]
            loop.function = tm["function"]
            sublooproot = json_to_loops(tm["children"], loopcounter)
            loop.subloops = sublooproot.subloops
            result.add_subloop(loop)
        elif kind == 'stmt':
            stmt = Loop.createStmt()
            stmt.filename = mkpath(tm["path"])
            stmt.line = tm["line"]
            stmt.column = tm["column"]
            stmt.function = tm["function"]
            result.add_subloop(stmt)
        else:
            assert False, "unknown kind"
    return result


class LoopCounter:
    def __init__(self):
        self.prevloopid = 0

    def clone(self):
        result = LoopCounter()
        result.prevloopid = self.prevloopid
        return result

    def nextId(self):
        self.prevloopid += 1
        return self.prevloopid 

class LoopNestExperiment:
    def __init__(self, loopnest, pragmalist, loopcounter):
        assert loopnest.isroot
        self.loopnest = loopnest
        self.pragmalist = pragmalist
        self.derived_from = None
        self.loopcounter = loopcounter

    def __str__(self):
        return '\n'.join(self.to_lines(printloopnest=True))

    def to_lines(self,printloopnest=False):
        if self.pragmalist:
            yield from self.pragmalist
        
        if not self.pragmalist or printloopnest:
            if self.pragmalist:
                yield ""
            yield from self.loopnest.to_lines(0)

    def get_num_children(self):
        return self.loopnest.get_num_children()

    def get_child(self, idx: int):
        resetloopcounter = self.loopcounter.clone()
        loopnest, pragmalist = self.loopnest.get_child(resetloopcounter, idx)
        assert len(loopnest)==1
        assert loopnest[0].isroot
        result = LoopNestExperiment(loopnest[0], self.pragmalist + pragmalist, loopcounter=resetloopcounter)
        result.derived_from = self
        return result


class Experiment:
    def __init__(self):
        self.nestexperiments = []
        self.derived_from = None
        self.derivatives = dict()
        self.duration = None
        self.exppath = None
        self.expnumber = None
        self.depth = 0

    def clone(self):
        result = Experiment()
        result.nestexperiments = self.nestexperiments.copy()
        return result

    def selector(self):
        def make_make_child_closure(i, nestexperiment):
            def make_child(loopcounter, idx: int):
                assert loopcounter == None
                if result := self.derivatives.get(idx):
                    return result

                newnestexp = nestexperiment.get_child(idx)
                result = Experiment()
                result.derived_from = self
                result.nestexperiments = self.nestexperiments.copy()
                result.nestexperiments[i] = newnestexp

                self.derivatives[idx] = result
                return result
            return make_child

        for i, nestexperiment in enumerate(self.nestexperiments):          
            yield nestexperiment.get_num_children(), make_make_child_closure(i,nestexperiment)

    def get_num_children(self):
        return mcount(self.selector())

    def get_child(self, idx: int):
        return mcall(self.selector(), None, idx=idx)

    def children(self):
        for idx in range(self.get_num_children()):
            yield self.get_child(idx)

    def derivatives_recursive(self, max_depth=None, filter=None, descendfilter=None):
        if filter and not filter(self):
            return
        yield self

        if max_depth != None and max_depth == 0:
            return
        if descendfilter and not descendfilter(self):
            return

        for n in self.children():
            yield from n.derivatives_recursive(max_depth=max_depth-1 if max_depth != None else None,filter=filter,descendfilter=descendfilter)

    def __str__(self):
        return '\n'.join(self.to_lines())

    def to_lines(self,printloopnest=False):
        if self.expnumber != None:
            yield f"Experiment {self.expnumber}"
        if self.duration != None and self.duration != math.inf:
            yield f"Exec time {self.duration}"

        isFirst = True
        for nestex in self.nestexperiments:
            if not isFirst:
                yield ""
            funcname = None
            for n in nestex.loopnest.subloops_recursive():
                if n.function:
                    funcname = n.function
                    break
            if funcname:
                yield "Function " + funcname + ":"
            for x in nestex.to_lines(printloopnest=printloopnest):
                yield "  " + x
            isFirst = False


class Tiling:
    @staticmethod
    def get_factory(tilesizes):
        def factory(loop):
            if loop.isloop and loop.transformable:
                return Tiling(loop, tilesizes)
            return None
        return factory

    def __init__(self, loop, tilesizes):
        self.loop = loop
        self.tilesizes = tilesizes
        self.num_children = mcount(self.selector())

    def apply_transform(self,loopnest,loopcounter,sizes,peel):
        d = len(sizes)
        assert d >= 1
        origloops = loopnest[:d]
        keeploops = loopnest[d:]
        floors = list([Loop.createLoop(name=f'floor{loopcounter.nextId()}') for i in range(0, d)])
        tiles = list([Loop.createLoop(name=f'tile{loopcounter.nextId()}') for i in range(0, d)])
        newloops = floors + tiles
        for outer, inner in neighbors(newloops):
            outer.subloops = [inner]
        newloops[-1].subloops = origloops[-1].subloops

        origloopids = [l.name for l in origloops]
        floorids = [floor.name for floor in floors]
        tileids = [tile.name for tile in tiles]
        sizes = [str(s) for s in sizes]
        peelclause = " peel(rectangular)" if peel else ""
        pragma = f"#pragma clang loop({','.join(origloopids)}) tile sizes({','.join(sizes)}){peelclause} floor_ids({','.join(floorids)}) tile_ids({','.join(tileids)})"
        return [newloops[0]], [pragma]

    def selector(self):
        loopnest = self.loop.perfectnest()
        n = len(loopnest)
        tilesizes = self.tilesizes

        def make_child_closure(d,enable_peeling):
            def make_child(loopcounter,idx: int):
                sizes = []
                leftover = idx
                for i in range(d):
                    sizes.append(tilesizes[leftover % len(tilesizes)])
                    leftover //= len(tilesizes)
                assert leftover == 0
                return self.apply_transform(loopnest=loopnest,loopcounter=loopcounter,sizes=sizes,peel=enable_peeling)
            return make_child

        for d in range(1, n+1):
            yield len(tilesizes)**d, make_child_closure(d,enable_peeling=False)
            yield len(tilesizes)**d, make_child_closure(d,enable_peeling=True)

    def get_num_children(self):
        return self.num_children

    def get_child(self, loopcounter, idx: int):
        return mcall(self.selector(), loopcounter, idx)


class Threading:
    @staticmethod
    def get_factory():
        def factory(loop: Loop):
            if loop.isloop and loop.transformable:
                return Threading(loop)
            return None
        return factory

    def __init__(self, loop):
        self.loop = loop

    def get_num_children(self):
        return 1

    def get_child(self, loopcounter, idx: int):
        parallel_loop = Loop.createAnonLoop()
        parallel_loop.subloops = self.loop.subloops
        pragma = f"#pragma clang loop({self.loop.name}) parallelize_thread"
        return [parallel_loop], [pragma]


class Interchange:
    @staticmethod
    def get_factory():
        def factory(loop: Loop):
            if loop.isloop and loop.transformable:
                return Interchange(loop)
            return None
        return factory

    def __init__(self, loop: Loop):
        self.loop = loop
        self.num_children = mcount(self.selector())

    def selector(self):
        loopnest = self.loop.perfectnest()
        n = len(loopnest)

        # If there is nothing to permute
        if n <= 1:
            return

        def make_make_child_closure(d):
            def make_child(loopcounter, idx: int):
                orignest = loopnest.copy()
                remaining = loopnest.copy()
                perm = []

                i = idx

                # New topmost loop cannot be the old topmost
                # Otherwise it should be an interchange of the nested loop; this also excludes the identity permutation
                select = i % (len(remaining) - 1) + 1
                i //= (len(remaining) - 1)
                perm.append(remaining[select])
                del remaining[select]

                while remaining:
                    select = i % len(remaining)
                    i //= len(remaining)
                    perm.append(remaining[select])
                    del remaining[select]

                assert i == 0
                assert len(perm) == n
                assert len(remaining) == 0

                # Strip trailing unchanged loops
                while perm[-1] == orignest[-1]:
                    del perm[-1]
                    del orignest[-1]

                newperm = [Loop.createLoop(name=f'perm{loopcounter.nextId()}') for l in perm]
                for p, c in neighbors(newperm):
                    p.subloops = [c]
                newperm[-1].subloops = orignest[-1].subloops

                nestids = [p.name for p in orignest]
                permids = [p.name for p in perm]
                newpermids = [p.name for p in newperm]
                pragma = f"#pragma clang loop({','.join(nestids)}) interchange permutation({','.join(permids)}) permuted_ids({','.join(newpermids)})"
                return [newperm[0]], [pragma]
            return make_child

        num_children = (n-1)
        for i in range(1, n-1):
            num_children *= i
        for d in range(1, n+1):
            yield num_children, make_make_child_closure(d)

    def get_num_children(self):
        return self.num_children

    def get_child(self, loopcounter, idx: int):
        return mcall(self.selector(), loopcounter, idx)


class Reversal:
    @staticmethod
    def get_factory():
        def factory(loop: Loop):
            if loop.isloop and loop.transformable:
                return Reversal(loop)
            return None
        return factory

    def __init__(self, loop):
        self.loop = loop

    def get_num_children(self):
        return 1

    def get_child(self, loopcounter, idx: int):
        reversed_loop = Loop.createLoop(name=f'rev{loopcounter.nextId()}')
        reversed_loop.subloops = self.loop.subloops
        pragma = f'#pragma clang loop({self.loop.name}) reverse reversed_id({reversed_loop.name})'
        return [reversed_loop], [pragma]


class Unrolling:
    @staticmethod
    def get_factory(factors, enable_full):
        def factory(loop: Loop):
            if loop.isloop and loop.transformable:
                return Unrolling(loop,factors,enable_full)
            return None
        return factory

    def __init__(self, loop: Loop, factors, enable_full):
        self.loop = loop
        self.factors = factors
        self.enable_full = enable_full
        self.num_children = mcount(self.selector())

    def selector(self):
        loop = self.loop

        def make_full_unrolling(loopcounter, idx: int):
            assert idx == 0
            unrolled_loop = Loop.createAnonLoop()
            unrolled_loop.subloops = loop.subloops
            pragma = f"#pragma clang loop({self.loop.name}) unrolling full"
            return [unrolled_loop], [pragma]

        def make_partial_unrolling(loopcounter, idx: int):
            factor = self.factors[idx]
            unrolled_loop = Loop.createLoop(name=f'unroll{loopcounter.nextId()}')
            unrolled_loop.subloops = loop.subloops
            pragma = f"#pragma clang loop({self.loop.name}) unrolling factor({factor})"
            return [unrolled_loop], [pragma]                

        if self.enable_full:
            yield 1, make_full_unrolling
        yield len(self.factors), make_partial_unrolling

    def get_num_children(self):
        return self.num_children

    def get_child(self, loopcounter, idx: int):
        return mcall(self.selector(), loopcounter, idx)


class UnrollingAndJam:
    @staticmethod
    def get_factory(factors, enable_full):
        def factory(loop: Loop):
            if loop.isloop and loop.transformable:
                return UnrollingAndJam(loop,factors,enable_full)
            return None
        return factory

    def __init__(self, loop: Loop, factors, enable_full):
        self.loop = loop
        self.factors = factors
        self.enable_full = enable_full
        self.num_children = mcount(self.selector())

    def selector(self):
        loop = self.loop

        # Require at least one subloop, not necessarily perfectly nested
        # TODO: jam depth, currently not supported by pragma-clang-loop
        jammable_subloops = []
        for l in loop.subloops:
            if l.isloop and l.transformable:
                jammable_subloops.append(l)
        if not jammable_subloops:
            return

        def jam_content(loopcounter,subloops,repeats):
            new_subloops = []
            streak = []

            def do_streak():
                nonlocal streak,new_subloops
                if not streak:
                    return 
                new_subloops += streak*repeats
                streak = []

            for sub in subloops:
                if sub.isloop and sub.transformable:
                    do_streak()
                    jam = Loop.createLoop(name=f'jam{loopcounter.nextId()}')
                    jam.subloops =  [s for s in sub.subloops]*repeats
                    new_subloops.append(jam)
                else:
                    streak.append(sub)
            do_streak()
            return new_subloops
                 
        def make_full_unrollingandjam(loopcounter, idx: int):
            assert idx == 0

            # FIXME: Should jam repeat as often as we unroll, but don't know the trip count here
            new_subloops = jam_content(loopcounter,loop.subloops, 1)

            pragma = f"#pragma clang loop({self.loop.name}) unrollingandjam full"
            return new_subloops, [pragma]

        def make_partial_unrollingandjam(loopcounter, idx: int):
            factor = self.factors[idx]

            new_subloops = jam_content(loopcounter,loop.subloops, factor)

            unrolled_loop = Loop.createLoop(name=f'unroll{loopcounter.nextId()}')
            unrolled_loop.subloops = new_subloops
            pragma = f"#pragma clang loop({self.loop.name}) unrollingandjam factor({factor})"
            return [unrolled_loop], [pragma]                

        if self.enable_full:
            yield 1,make_full_unrollingandjam
        yield len(self.factors),make_partial_unrollingandjam

    def get_num_children(self):
        return self.num_children

    def get_child(self, loopcounter, idx: int):
        return mcall(self.selector(), loopcounter, idx)



class ArrayPacking:
    @staticmethod
    def get_factory(arrays):
        def factory(loop: Loop):
            if loop.isloop and loop.transformable:
                return ArrayPacking(loop,arrays)
            return None
        return factory

    def __init__(self, loop: Loop, arrays):
        self.loop = loop
        self.arrays = list(arrays)
        self.num_children = mcount(self.selector())

    def selector(self):
        loop = self.loop

        def make_array_packing(loopcounter, idx: int):
            assert 0 <= idx < len(self.arrays)
            packed_loop = Loop.createAnonLoop()
            packed_loop.subloops = self.loop.subloops
            pragma = f"#pragma clang loop({self.loop.name}) pack array({self.arrays[idx]})"
            return [packed_loop], [pragma]         

        yield len(self.arrays), make_array_packing

    def get_num_children(self):
        return self.num_children

    def get_child(self, loopcounter, idx: int):
        return mcall(self.selector(), loopcounter, idx)


class Fission:
    @staticmethod
    def get_factory():
        def factory(loop):
            if loop.isloop and loop.transformable:
                return Fission(loop)
            return None
        return factory

    def __init__(self, loop):
        self.loop = loop
        self.num_children = mcount(self.selector())

    def selector(self):
        loop = self.loop
        subcount = len(loop.subloops)

        # One split point
        # TODO: arbitrary number of split points (2^n possibilities)
        def make_fission(loopcounter,idx: int):
            split_at = idx+1
            assert 0 < split_at < subcount
            head_loop = Loop.createLoop(name=f"head{loopcounter.nextId()}")
            head_loop.subloops = loop.subloops[:split_at]
            tail_loop = Loop.createLoop(name=f"tail{loopcounter.nextId()}")
            tail_loop.subloops = loop.subloops[split_at:]
            pragma = f"#pragma clang loop({loop.name}) fission split_at({idx})"
            return [head_loop,tail_loop], [pragma]

        yield subcount-1,make_fission


    def get_num_children(self):
        return self.num_children

    def get_child(self, loopcounter, idx: int):
        return mcall(self.selector(), loopcounter, idx)



class Fusion:
    @staticmethod
    def get_factory():
        def factory(loop):
            return Fusion(loop)
        return factory

    def __init__(self, loop):
        self.loop = loop
        self.num_children = mcount(self.selector())

    def selector(self):
        loop = self.loop
        subcount = len(loop.subloops)
        fusablecount = 0

        def can_fuse(i):
            fuse_loops = loop.subloops[i:i+2]
            return fuse_loops[0].isloop and fuse_loops[1].isloop and fuse_loops[0].transformable and fuse_loops[1].transformable 

        for i in range(subcount-1):
            if can_fuse(i):
                fusablecount+=1

        # Fuse two neighboring loops
        # TODO: Fuse any number of neigboring loops
        def make_fusion(loopcounter,idx: int):
            fuseidx = 0
            for i in range(subcount-1):
                if can_fuse(i):
                    if fuseidx == idx:
                        fusei = i
                        break
                    fuseidx+=1

            fuse_loops = loop.subloops[fusei:fusei+2]

            # TODO: parent can be a root/sequence as well
            parent_loop = loop.clone()
            fused_loop = Loop.createLoop(name=f"fuse{loopcounter.nextId()}")
            parent_loop.subloops = []
            for l in loop.subloops:
                if l == fuse_loops[0]:
                    parent_loop.subloops.append(fused_loop)
                elif l == fuse_loops[1]:
                    pass
                else:
                    parent_loop.subloops.append(l)
            fused_loop.subloops = [outer for l in fuse_loops for outer in l.subloops]
            #fused_loop.subloops = [inner for l in fuse_loops for outer in l.subloops for inner in outer.subloops]

            # Loop fusion is special as we transform the parent, but add the pragma applies to the children while the parent keeps its name. The parent also does not need to be 'transformable'
            pragma = f"#pragma clang loop({fuse_loops[0].name},{fuse_loops[1].name}) fuse fused_id({fused_loop.name})"
            return [parent_loop], [pragma]

        yield fusablecount,make_fusion


    def get_num_children(self):
        return self.num_children

    def get_child(self, loopcounter, idx: int):
        return mcall(self.selector(), loopcounter, idx)



def as_dot(baseexperiment: Experiment, max_depth=None, filter=None, decendfilter=None, loopneststructure=False):
    yield 'digraph G {'
    yield '  rankdir=LR;'
    yield ''

    for experiment in baseexperiment.derivatives_recursive(max_depth=max_depth, filter=filter, descendfilter=decendfilter):
        desc = ''.join(l + '\\l' for l in experiment.to_lines(printloopnest=loopneststructure))

        if experiment.duration == math.inf:
            fillcolor = 'lightpink:crimson'
        elif experiment.duration != None:
            fillcolor = 'darkseagreen1:lawngreen'
        else:
            fillcolor = 'azure:powderblue'

        yield f'  n{id(experiment)}[shape=box color="grey30" penwidth=2 fillcolor="{fillcolor}" style="filled,rounded" gradientangle=315 fontname="Calibri Light" label="{desc}"];'

        if parent := experiment.derived_from:
            yield f'  n{id(parent)} -> n{id(experiment)};'
        yield ''

    yield '}'


# Decorator
commands = {}
def subcommand(name):
    def command_func(_func):
        global commands
        commands[name] = _func
        return _func
    return command_func


def add_boolean_argument(parser, name, default=False, dest=None, help=None):
    """Add a boolean argument to an ArgumentParser instance."""

    for i in range(2):
        if name[0] == '-':
            name = name[1:]

    destname = dest or name.replace('-', '_')

    onhelptext = None
    offhelptext = None
    if help is not None:
        onhepltext = help + (" (default)" if default else "")
        offhelptext = "Disable " + help + (" (default)" if default else "")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--' + name, dest=destname,
                       action='store_true', help=onhelptext)
    group.add_argument('--no-' + name, dest=destname,
                       action='store_false', help=offhelptext)
    defaults = {destname: default}
    parser.set_defaults(**defaults)


@subcommand("example")
def example(parser, args):
    if parser:
        add_boolean_argument(parser,'loopneststructure')
    if args:
        loopcounter = LoopCounter()
        example = Loop.createRoot()
        example.new_subloop(loopcounter).new_substmt()
        outer = example.new_subloop(loopcounter)
        outer.new_subloop(loopcounter).new_substmt()
        #outer.new_substmt()

        root = Experiment()
        root.nestexperiments.append(LoopNestExperiment(example, [], loopcounter))

        for line in as_dot(root, max_depth=args.maxdepth,loopneststructure=args.loopneststructure):
            print(line)
        return 0


def read_json(files):
    root = Experiment()
    for fn in files:
        with mkpath(fn).open() as fo:
            data = json.load(fo)
        loopnests = data["scops"]
        loopcounter = LoopCounter()
        for ln in loopnests:            
            nestroot = json_to_loops(ln["children"], loopcounter)
            exroot = LoopNestExperiment(nestroot, [], loopcounter=loopcounter)
            root.nestexperiments.append(exroot)
    return root


@subcommand("jsonfile")
def jsonfile(parser, args):
    if parser:
        parser.add_argument('filename', nargs='+')
    if args:
        root = read_json(files=args.filenames)
        for line in as_dot(root, max_depth=args.maxdepth):
            print(line)
        return 0


def make_ccline(ccargs, ccfiles=None, outfile=None, debuginfo=None, extraflags=[]):
    ccfiles = first_defined(ccfiles, ccargs.ccfiles)
    outfile = first_defined(outfile, ccargs.o)

    cmdline = [ccargs.cc]
    cmdline += ccfiles
    cmdline += ccargs.ccflags
    cmdline += ['-iquote', ccargs.cwd]
    cwdincdirs = set()
    for ccf in ccargs.ccfiles:
        cwdincdirs.add((ccargs.cwd / ccf).parent)
    for incdir  in cwdincdirs:
        cmdline += ['-iquote', incdir]
    cmdline += ['-flegacy-pass-manager', '-ferror-limit=1',
                '-mllvm', '-polly', '-mllvm', '-polly-process-unprofitable',
                '-mllvm', '-polly-reschedule=0', '-mllvm', '-polly-pattern-matching-based-opts=0']
    if ccargs.polybench_time:
        cmdline += ['-DPOLYBENCH_TIME=1'] 
    if debuginfo:
        # FIXME: loopnests.json overwritten if multiple files passed to clang
        cmdline += ['-g', '-gcolumn-info']
    if os.name == 'nt':
        cmdline += ['-l', r"C:\Users\meinersbur\build\llvm-project\release\lib\libomp.dll.lib"]
    else:
        cmdline += ['-fopenmp']
    cmdline += ['-mllvm', '-polly-omp-backend=LLVM', '-mllvm', '-polly-scheduling=static']
    cmdline += ['-Werror=pass-failed']
    cmdline += extraflags
    cmdline += ['-o', outfile]
    return cmdline



def run_exec(experiment,cwd,exefile,execopts):
    print("Time the output...")
    polybench_time = execopts.polybench_time

    p = invoke.diag(exefile, *execopts.args, timeout=execopts.timeout, std_prefixed=cwd / 'exec.txt',
                appendenv={'LD_LIBRARY_PATH': execopts.ld_library_path}, cwd=cwd, onerror=invoke.Invoke.EXCEPTION,return_stdout=polybench_time)

    if polybench_time:
        experiment.duration = datetime.timedelta(seconds=float(p.stdout.rstrip().splitlines()[-1]))
        print(f"Execution completed in {p.walltime}; polybench measurement: {experiment.duration}")
    else:
        experiment.duration = p.walltime
        print(f"Execution completed in {p.walltime}")
    experiment.exppath = cwd


def extract_loopnests(tempdir, ccargs, execopts):
    print("Extract loop nests...")

    extractloopnest = tempdir / 'base'
    extractloopnest.mkdir(parents=True, exist_ok=True)
    exefile = extractloopnest / ccargs.o.name

    cmdline = make_ccline(ccargs, outfile=exefile, 
        extraflags=['-mllvm', '-polly-dump-loopnest'], debuginfo=True)
    invoke.diag(*cmdline, cwd=extractloopnest, onerror=invoke.Invoke.EXCEPTION)

    loopnestfiles = [extractloopnest /  f"{ccfile.stem}-loopnest.json" for ccfile in ccargs.ccfiles]
    loopnestfiles = [f for f in loopnestfiles if f.is_file()]
    root = read_json(files=loopnestfiles)
    root.expnumber = 0
    root.depth = 1

    run_exec(experiment=root,cwd=extractloopnest,exefile=exefile,execopts=execopts)

    return root


expnumber = 1
def run_experiment(tempdir: pathlib.Path, experiment: Experiment, ccargs, execopts, writedot: bool, dotfilter, dotexpandfilter, root: Experiment):
    global expnumber
    expdir = tempdir / f"experiment{expnumber}"
    experiment.expnumber = expnumber
    logfile = expdir / 'desc.txt'
    dotfile = expdir / 'graph.dot'
    expnumber += 1
    expdir.mkdir(parents=True, exist_ok=True)
    experiment.exppath = expdir

    print(f"Run next experiment in {expdir}")
    print(experiment)

    with logfile.open('w+')as f:
        for line in experiment.to_lines():
            f.write(line)
            f.write('\n')

    contents = {}
    for f in ccargs.ccfiles:
        contents[f.resolve()] = f.open('r').readlines()

    for x in experiment.nestexperiments:
        rootloopnestexperiment = x
        while rootloopnestexperiment.derived_from != None:
            rootloopnestexperiment = rootloopnestexperiment.derived_from

        first = None

        for loop in rootloopnestexperiment.loopnest.subloops_recursive():
            if not loop.isloop:
                continue
            filename = mkpath(loop.filename).resolve()
            line = loop.line-1     # is one-based
            column = loop.column-1 # is one-based
            name = loop.name

            if (first == None) or (first > (line, column)):
                first = (line, column)

            contentlines = contents.get(filename)
            assert contentlines, "Loopnest's file not matched with intput file?!?"

            oldline = contentlines[line]
            # FIXME: if multiple loops per line, ensure that later are replaced first
            newline = oldline[:column] + f"\n#pragma clang loop id({name})\n" + oldline[column:]
            contentlines[line] = newline

        oldline = contentlines[first[0]]
        newline = oldline[:first[1]] + '\n' + ''.join(s + '\n' for s in reversed(x.pragmalist)) + oldline[first[1]:]
        contentlines[first[0]] = newline

    # Writeback files in new dir
    newccfiles = []
    for k, content in contents.items():
        filename = expdir / k.name
        createfile(filename, ''.join(content))
        newccfiles.append(filename)
        pass

    exefile = expdir / ccargs.o.name
    cmdline = make_ccline(ccargs, ccfiles=newccfiles, outfile=exefile)
    try:
        invoke.diag(*cmdline, cwd=expdir, onerror=invoke.Invoke.EXCEPTION, std_prefixed=expdir / 'cc.txt')
    except subprocess.CalledProcessError:
        # Compilation failed; illegal program
        experiment.duration = math.inf
        return

    try:
        run_exec(experiment=experiment,cwd=expdir,exefile=exefile,execopts=execopts)
    except subprocess.TimeoutExpired:
        # Assume failure
        experiment.duration = math.inf
        return

    with logfile.open('a') as f:
        f.write(f"Measured time is {experiment.duration}\n")

    if writedot:
        with dotfile.open('w+') as f:
            for line in as_dot(root, filter=dotfilter, decendfilter=dotexpandfilter):
                print(line, file=f)


def parse_cc_cmdline(cmdline):
    cc = cmdline[0]
    ccflags = []
    ccfiles = []
    o = None

    i = 1
    while i < len(cmdline):
        arg = cmdline[i]
        nextarg = cmdline[i+1] if i + 1 < len(cmdline) else None

        if arg == '-o' and nextarg != None:
            o = mkpath(nextarg)
            i += 2
            continue

        if arg.startswith('-'):
            ccflags.append(arg)
            i += 1
            continue

        fn = mkpath(arg)
        if not fn.suffix.lower() in {'.c', '.cxx', '.cpp'}:
            ccflags.append(arg)
            i += 1
            continue

        if not fn.is_file():
            ccflags.append(arg)
            i += 1
            continue

        ccfiles.append(fn)
        i += 1

    result = argparse.Namespace()
    result.cwd = pathlib.Path.cwd()
    result.cc = cc
    result.ccfiles = [f.resolve() for f in ccfiles]
    result.o = o
    result.ccflags = ccflags
    return result


class PriorityQueue:
    class Item:
        def __init__(self, priority: float, item):
            self.priority = priority
            self.item = item

        def __eq__(self, other):
            return self.priority == other.priority

        def __ne__(self, other):
            return self.priority != other.priority

        def __gt__(self, other):
            return self.priority > other.priority

        def __lt__(self, other):
            return self.priority < other.priority

        def __ge__(self, other):
            return self.priority >= other.priority

        def __le__(self, other):
            return self.priority <= other.priority

    def __init__(self, *args, key=lambda x: x):
        self.key = key
        self.elts = [PriorityQueue.Item(key(x), x) for x in args]
        heapq.heapify(self.elts)

    def push(self, item):
        heapq.heappush(self.elts, PriorityQueue.Item(self.key(item), item))

    def pop(self):
        return heapq.heappop(self.elts).item

    def top(self):
        return self.elts[0].item

    def replace(self, item):
        heapq.heapreplace(self.elts, PriorityQueue.Item(self.key(item), item))

    def update(self):
        self.replace(self.top())

    def __len__(self):
        return len(self.elts)

    def empty(self):
        return len(self.elts) == 0


@subcommand("autotune")
def autotune(parser, args):
    if parser:
        add_boolean_argument(parser, 'keep')
        parser.add_argument('--exec-arg', action='append')
        parser.add_argument('--exec-args', action='append')
        add_boolean_argument(parser, 'polybench-time')
        parser.add_argument('--ld-library-path', action='append')
        parser.add_argument('--outdir')
        parser.add_argument('--timeout', type=float,
                            help="Max exec time in seconds; default is no timout")
        parser.add_argument('ccline', nargs=argparse.REMAINDER)
    if args:
        ccargs = parse_cc_cmdline(args.ccline)
        ccargs.polybench_time = args.polybench_time

        execopts = argparse.Namespace()

        execopts.ld_library_path = None
        if args.ld_library_path != None:
            execopts.ld_library_path = ':'.join(args.ld_library_path)

        execopts.timeout = None
        if args.timeout != None:
            execopts.timeout = datetime.timedelta(seconds=args.timeout)
        execopts.polybench_time = args.polybench_time

        execopts.args = shcombine(arg=args.exec_arg,args=args.exec_args)

        outdir = mkpath(args.outdir)
        maxdepth = 0
        num_experiments = 0

        with contextlib.ExitStack() as stack:
            if args.keep:
                d = tempfile.mkdtemp(dir=outdir, prefix='mctree-')
            else:
                d = stack.enter_context(tempfile.TemporaryDirectory(dir=outdir, prefix='mctree-'))
            d = mkpath(d)

            bestfile = d / 'best.txt'
            csvfile = d / 'experiments.csv'
            newbestcsvfile = d / 'newbest.csv'
            csvlog = csvfile.open('w+')
            newbestlog = newbestcsvfile.open('w+')

            root = extract_loopnests(d, ccargs=ccargs, execopts=execopts)
            print("Baseline is")
            print(root)
            print("")

            def priorotyfunc(x): 
                return -math.inf if x.duration is None else x.duration.total_seconds()
            pq = PriorityQueue(root, key=priorotyfunc)
            closed = set()
            bestsofar = root

            csvlog.write(f"{root.expnumber},{root.duration.total_seconds()},{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
            newbestlog.write(f"{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")

            while not pq.empty():
                item = pq.top()

                if item.duration == None:
                    num_experiments += 1
                    run_experiment(d, item, ccargs=ccargs, execopts=execopts, 
                        writedot=num_experiments < 30, 
                        dotfilter=None,
                        dotexpandfilter=lambda n: n in closed,
                        root=root)
                    if item.duration == math.inf:
                        # Invalid pragmas? Remove experiment entirely
                        print("Experiment failed")
                        pq.pop()
                        continue

                    pq.update()
                    if bestsofar.duration > item.duration:
                        print(
                            f"New experiment better than old (old: {bestsofar.duration}, new: {item.duration})")
                        print(f"Path {item.exppath}\n")
                        print(item)
                        bestsofar = item
                        with bestfile.open('w+') as f:
                            f.write(f"Best experiment so far\n")
                            f.write(f"Time: {bestsofar.duration}\n")
                            f.write(f"Path: {bestsofar.exppath}\n\n")
                            for line in bestsofar.to_lines():
                                f.write(line)
                                f.write('\n')
                        newbestlog.write(f"{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
                        newbestlog.flush()
                    csvlog.write(f"{item.expnumber},{item.duration.total_seconds()},{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
                    csvlog.flush()
                    continue


                if not item in closed:
                    print(f"Selecting best experiment {item.duration} for expansion")
                    for child in item.children():
                        pq.push(child)

                    closed.add(item)
                    continue

                if item in closed and item.duration != None:
                    pq.pop()

            print("No more experiments!!?")


def main(argv: str) -> int:
    global transformers
    parser = argparse.ArgumentParser(description="Loop transformation search tree proof-of-concept", allow_abbrev=False)

    parser.add_argument('--maxdepth', type=int, default=2)
    add_boolean_argument(parser, "--tiling", default=True)
    parser.add_argument('--tiling-sizes')
    add_boolean_argument(parser, "--threading", default=True)
    add_boolean_argument(parser, "--interchange", default=True)
    add_boolean_argument(parser, "--reversal", default=True)
    add_boolean_argument(parser, "--unrolling", default=True)
    add_boolean_argument(parser, "--unrolling-full", default=True)
    parser.add_argument('--unrolling-factors')
    add_boolean_argument(parser, "--unrolling-and-jam", default=True)
    add_boolean_argument(parser, "--unrolling-and-jam-full", default=True)
    parser.add_argument('--unrolling-and-jam-factors')
    parser.add_argument('--packing-arrays',action='append')
    add_boolean_argument(parser, "--fission", default=True)
    add_boolean_argument(parser, "--fusion", default=True)

    subparsers = parser.add_subparsers(dest='subcommand')
    for cmd, func in commands.items():
        subparser = subparsers.add_parser(cmd)
        func(parser=subparser, args=None)
    args = parser.parse_args(str(v) for v in argv[1:])

    if args.tiling:
        tilesizes = [4, 16, 64, 256]
        if args.tiling_sizes != None:
            tilesizes = [int(s) for s in args.tiling_sizes.split(',')]
        transformers.append(Tiling.get_factory(tilesizes))
    if args.threading:
        transformers.append(Threading.get_factory())
    if args.interchange:
        transformers.append(Interchange.get_factory())
    if args.reversal:
        transformers.append(Reversal.get_factory())
    if args.unrolling:
        factors = [2, 4, 8]
        if args.unrolling_factors != None:
            factors = [int(s) for s in args.unrolling_factors.split(',')]
        transformers.append(Unrolling.get_factory(factors,args.unrolling_full))
    if args.unrolling_and_jam:
        factors = [2, 4, 8]
        if args.unrolling_and_jam_factors != None:
            factors = [int(s) for s in args.unroll_and_jam_factors.split(',')]
        transformers.append(UnrollingAndJam.get_factory(factors,args.unrolling_and_jam_full))
    pack_arrays = set()
    if args.packing_arrays:
            pack_arrays = set(arr for arrlist in args.packing_arrays for arr in arrlist.split(','))
    if pack_arrays:
            transformers.append(ArrayPacking.get_factory(pack_arrays))
    if args.fission:
        transformers.append(Fission.get_factory())
    if args.fusion:
        transformers.append(Fusion.get_factory())

    cmdlet = commands.get(args.subcommand)
    if not cmdlet:
        die("No command?")
    return cmdlet(parser=None, args=args)


if __name__ == '__main__':
    if errcode := main(argv=sys.argv):
        exit(errcode)
