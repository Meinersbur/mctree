#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import itertools
import json
import pathlib


class Loop:
    numloopssofar = 0

    @classmethod
    def createLoop(cls, name: str = None):
        if not name:
            Loop.numloopssofar += 1
            name = f'loop{Loop.numloopssofar}'
        return cls(isroot=False, name=name)

    @classmethod
    def createRoot(cls):
        return cls(isroot=True, name=None)

    def __init__(self, isroot: bool, name: str):
        self.isroot = isroot
        if not isroot:
            self.name = name
        self.subloops = []
        self.filename = None
        self.line = None
        self.column = None
        self.function = None
        self.entry = None
        self.exit = None

    def perfectnest(self):
        result = [self]
        while True:
            if len(result[-1].subloops) != 1:
                break
            result.append(result[-1].subloops[0])
        return result

    def subloops_recursive(self):
        yield self
        for n in self.subloops:
            yield from n.subloops_recursive()

    def new_subloop(self):
        newloop = Loop.createLoop()
        self.subloops.append(newloop)
        return newloop

    def add_subloop(self, subloop):
        self.subloops.append(subloop)
        return subloop

    def clone(self):
        if self.isroot:
            result = Loop(isroot=True, name=None)
        else:
            result = Loop(isroot=False, name=self.name)
        result.subloops = self.subloops.copy()
        result.filename = self.filename
        result.line = self.line
        result.column = self.column
        result.function = self.function
        result.entry = self.entry
        result.exit = self.exit
        return result

    def __str__(self) -> str:
        return '\n'.join(self.to_lines(0))

    def to_lines(self, indent: int = 0):
        block = False
        subindent = indent
        if not self.isroot:
            block = len(self.subloops) > 1
            yield "    "*indent + f"#pragma clang loop id({self.name})"
            loc = ""
            if self.filename:
                loc = f" /* {self.filename}:{self.line}:{self.column} */"
            yield "    "*indent + "for (...)" + (" {" if block else "") + loc
            subindent += 1

        if self.subloops:
            for subloop in self.subloops:
                yield from subloop.to_lines(indent=subindent)
        else:
            yield "    "*subindent + "code;"

        if block:
            yield "    "*indent + "}"


# Replace oldloop with newloop in loop nest while cloning all loop to the path there
def gist(root, childids, oldloop, newloop):
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
    cur.subloops[lastid] = newloop
    return newroot


def json_to_loops(topmost):
    result = Loop.createRoot()
    for tm in topmost:
        loop = Loop.createLoop()
        loop.filename = pathlib.Path(tm["filename"])
        loop.line = tm["line"]
        loop.column = tm["column"]
        loop.entry = tm["entry"]
        loop.exit = tm["exit"]
        loop.function = tm["function"]
        sublooproot = json_to_loops(tm["subloops"])
        loop.subloops = sublooproot.subloops
        result.add_subloop(loop)
    return result


def transform_node(root, callback):
    def do_transform_node(loop, childids,  callback):
        if not loop.isroot:
            for newsubtree in callback(loop):
                newroot = gist(root, childids, loop, newsubtree[0])
                yield tuple(list(newsubtree) + [newroot])
        for i, child in enumerate(loop.subloops):
            yield from do_transform_node(child, childids + [i], callback)
    yield from do_transform_node(root, [], callback)


class LoopNestExperiment:
    def __init__(self, loopnest, pragmalist):
        self.loopnest = loopnest
        self.pragmalist = pragmalist
        self.derived_from = None

    def __str__(self):
        return '\n'.join(self.to_lines())

    def to_lines(self):
        if self.pragmalist:
            return self.pragmalist
        else:
            return self.loopnest.to_lines(0)


class Experiment:
    def __init__(self):
        self.nestexperiments = []
        self.derived_from = None
        self.derivatives = []

    def clone(self):
        result = Experiment()
        result.nestexperiments = self.nestexperiments.copy()
        return result

    def add_subexperiment(self, subexp):
        subexp.derived_from = self
        self.derivatives.append(subexp)

    def derivatives_recursive(self):
        yield self
        for n in self.derivatives:
            yield from n.derivatives_recursive()

    def __str__(self):
        return '\n'.join(self.to_lines())

    def to_lines(self):
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
            for x in nestex.to_lines():
                yield "  " + x
            isFirst = False


class Tiling:
    tilesizes = [2, 4]

    @staticmethod
    def do_subtile(loop):
        # End tiling here
        yield [], [], loop.subloops, []

        if len(loop.subloops) == 1:
            yield from Tiling.do_tile(loop.subloops[0])

    @staticmethod
    def do_tile(loop):
        for subfloors, subtiles, subbody, subsizes in Tiling.do_subtile(loop):
            for tilesize in Tiling.tilesizes:
                yield [Loop.createLoop()] + subfloors, [Loop.createLoop()] + subtiles, subbody, [tilesize] + subsizes

    @staticmethod
    def gen_tiling(loop: Loop):
        for floors, tiles, body, sizes in Tiling.do_tile(loop):
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
            yield floors[0], [pragma]


class Thrading:
    @staticmethod
    def gen_threading(loop: Loop):
        parallel_loop = Loop.createLoop()
        parallel_loop.subloops = loop.subloops
        pragma = f"#pragma clang transform parallelize_thread on({loop.name}) parallel_id({parallel_loop.name})"
        yield parallel_loop, [pragma]


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
            for p, c in zip(newperm, newperm[1:]):
                p.subloops = [c]
            newperm[-1].subloops = nests[-1].subloops

            permids = [p.name for p in perm]
            newpermids = [p.name for p in newperm]
            pragma = f"#pragma clang transform interchange on({nests[0].name}) permutation({','.join(permids)}) permuted_ids({','.join(newpermids)})"
            yield newperm[0], [pragma]


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


def derive_loopnest_expriments(searchspacenode: Experiment, i: int):
    baseexperiment = searchspacenode.nestexperiments[i]

    oldroot = baseexperiment.loopnest  # type: Loop
    for newsubloop, pragma, newroot in transform_node(oldroot, do_transformations):
        x = LoopNestExperiment(newroot, baseexperiment.pragmalist + pragma)
        x.derived_from = baseexperiment

        y = searchspacenode.clone()
        y.nestexperiments[i] = x
        searchspacenode.add_subexperiment(y)


def derive_expriments(searchspacenode: Experiment):
    for i in range(len(searchspacenode.nestexperiments)):
        derive_loopnest_expriments(searchspacenode, i)


def expand_searchtree(baseexperiment: Experiment, remaining_depth: int):
    if remaining_depth <= 0:
        return

    derive_expriments(baseexperiment)
    for e in baseexperiment.derivatives:
        expand_searchtree(e, remaining_depth=remaining_depth-1)


def as_dot(baseexperiment: LoopNestExperiment):
    yield "digraph G {"
    yield "rankdir=LR;"

    for experiment in baseexperiment.derivatives_recursive():
        desc = ''.join(l + "\\l" for l in experiment.to_lines())
        yield f'n{id(experiment)}[shape=box color="grey30" penwidth=2 fillcolor="azure:powderblue" style="filled,rounded" gradientangle=315 fontname="Calibri Light" label="{desc}"];'

        if parent := experiment.derived_from:
            yield f"n{id(parent)} -> n{id(experiment)};"
        yield ""

    yield "}"


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
        pass
    if args:
        example = Loop.createRoot()
        example.new_subloop()
        example.new_subloop().new_subloop()

        root = Experiment()
        root.nestexperiments.append(LoopNestExperiment(example, []))
        expand_searchtree(root, remaining_depth=args.maxdepth)

        for line in as_dot(root):
            print(line)
        return 0


@subcommand("jsonfile")
def jsonfile(parser, args):
    if parser:
        parser.add_argument('filename', nargs='+')
    if args:
        root = Experiment()
        for fn in args.filename:
            with pathlib.Path(fn).open() as fo:
                data = json.load(fo)
            loopnests = data["loopnests"]
            for ln in loopnests:
                nestroot = json_to_loops(ln["topmost"])
                exroot = LoopNestExperiment(nestroot, [])
                root.nestexperiments.append(exroot)

        expand_searchtree(root, remaining_depth=args.maxdepth)

        for line in as_dot(root):
            print(line)
        return 0


def main(argv: str) -> int:
    global tiling_enabled, threading_enabled, interchange_enabled
    parser = argparse.ArgumentParser(
        description="Loop transformation search tree proof-of-concept", allow_abbrev=False)

    parser.add_argument('--maxdepth', type=int, default=2)
    add_boolean_argument(parser, "--tiling", default=True)
    parser.add_argument('--tiling-sizes', nargs='*', type=int, default=[2, 4])
    add_boolean_argument(parser, "--threading", default=True)
    add_boolean_argument(parser, "--interchange", default=True)

    subparsers = parser.add_subparsers(dest='subcommand')
    for cmd, func in commands.items():
        subparser = subparsers.add_parser(cmd)
        func(parser=subparser, args=None)
    args = parser.parse_args(str(v) for v in argv[1:])

    tiling_enabled = args.tiling
    Tiling.tilesizes = args.tiling_sizes
    threading_enabled = args.threading
    interchange_enabled = args.interchange

    cmdlet = commands.get(args.subcommand)
    if not cmdlet:
        die("No command?")
    return cmdlet(parser=None, args=args)


if __name__ == '__main__':
    if errcode := main(argv=sys.argv):
        exit(errcode)
