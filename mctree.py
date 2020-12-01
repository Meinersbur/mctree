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


def mcall(seq, idx: int):
    pos, elt = mselect(seq, idx)
    return elt(pos)


def neighbors(seq):
    it = iter(seq)
    prev = next(it)
    for item in it:
        yield prev, item
        prev = item


transformers = []


class Loop:
    numloopssofar = 0

    @classmethod
    def createLoop(cls, name: str = None):
        if not name:
            Loop.numloopssofar += 1
            name = f'loop{Loop.numloopssofar}'
        return cls(isroot=False, istransformable=True, name=name)

    @classmethod
    def createAnonLoop(cls):
        return cls(isroot=False, istransformable=False, name=None)

    @classmethod
    def createRoot(cls):
        return cls(isroot=True, istransformable=False, name=None)

    def __init__(self, isroot: bool, istransformable: bool, name: str):
        if isroot or not istransformable:
            assert name == None

        self.isroot = isroot
        self.transformable = istransformable

        self.name = name
        self.subloops = []
        self.isperfectnest = None
        self.filename = None
        self.line = None
        self.column = None
        self.function = None
        self.entry = None
        self.exit = None

    def selector(self):
        if not self.isroot and self.transformable:
            looptransformers = (t(self) for t in transformers)
            for transformer in looptransformers:
                yield transformer.get_num_children(), transformer.get_child
        for i, subloop in enumerate(self.subloops):
            subloop = self.subloops[i]

            def replace_loop(idx: int):
                newsubloop, pragmas = subloop.get_child(idx)
                newloop = self.clone()
                newloop.subloops[i] = newsubloop
                return newloop, pragmas
            yield subloop.get_num_children(), replace_loop

    def get_num_children(self):
        return mcount(self.selector())

    def get_child(self, idx: int):
        return mcall(self.selector(), idx)

    def perfectnest(self):
        assert self.transformable
        assert not self.isroot

        result = [self]
        while True:
            lastsubloops = result[-1].subloops
            if result[-1].isperfectnest == False:
                break
            if len(lastsubloops) != 1:
                break
            if not lastsubloops[0].transformable:
                break

            result.append(lastsubloops[0])
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
            result = Loop(isroot=True, istransformable=False, name=None)
        else:
            result = Loop(isroot=False, istransformable=self.transformable, name=self.name)
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


def json_to_loops(topmost):
    result = Loop.createRoot()
    for tm in topmost:
        loop = Loop.createLoop()
        loop.filename = mkpath(tm["path"])
        loop.line = tm["line"]
        loop.column = tm["column"]
        loop.entry = tm["entry"]
        loop.exit = tm["exit"]
        loop.function = tm["function"]
        loop.isperfectnest = tm.get('perfectnest')
        sublooproot = json_to_loops(tm["subloops"])
        loop.subloops = sublooproot.subloops
        result.add_subloop(loop)
    return result


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

    def get_num_children(self):
        return self.loopnest.get_num_children()

    def get_child(self, idx: int):
        loopnest, pragmalist = self.loopnest.get_child(idx)
        result = LoopNestExperiment(loopnest, self.pragmalist + pragmalist)
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
        for i, nestexperiment in enumerate(self.nestexperiments):
            def make_child(idx: int):
                if result := self.derivatives.get(idx):
                    return result

                newnestexp = nestexperiment.get_child(idx)
                result = Experiment()
                result.derived_from = self
                result.nestexperiments = self.nestexperiments.copy()
                result.nestexperiments[i] = newnestexp

                self.derivatives[idx] = result
                return result
            yield nestexperiment.get_num_children(), make_child

    def get_num_children(self):
        return mcount(self.selector())

    def get_child(self, idx: int):
        return mcall(self.selector(), idx)

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

    def to_lines(self):
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
            for x in nestex.to_lines():
                yield "  " + x
            isFirst = False


class Tiling:
    @staticmethod
    def get_factory(tilesizes):
        def factory(loop):
            return Tiling(loop, tilesizes)
        return factory

    def __init__(self, loop, tilesizes):
        self.loop = loop
        self.tilesizes = tilesizes
        self.num_children = mcount(self.selector())

    def selector(self):
        loopnest = self.loop.perfectnest()
        n = len(loopnest)
        tilesizes = self.tilesizes

        for d in range(1, n+1):
            def make_child(idx: int):
                origloops = loopnest[:d]
                keeploops = loopnest[d:]
                floors = list([Loop.createLoop() for i in range(0, d)])
                tiles = list([Loop.createLoop() for i in range(0, d)])
                newloops = floors + tiles
                for outer, inner in neighbors(newloops):
                    outer.subloops = [inner]
                newloops[-1].subloops = origloops[-1].subloops

                sizes = []
                leftover = idx
                for i in range(d):
                    sizes.append(tilesizes[leftover % len(tilesizes)])
                    leftover //= len(tilesizes)
                assert leftover == 0
                origloopids = [l.name for l in origloops]
                floorids = [floor.name for floor in floors]
                tileids = [tile.name for tile in tiles]
                sizes = [str(s) for s in sizes]
                pragma = f"#pragma clang loop({','.join(origloopids)}) tile sizes({','.join(sizes)}) floor_ids({','.join(floorids)}) tile_ids({','.join(tileids)})"
                return newloops[0], [pragma]

            yield len(tilesizes)**d, make_child

    def get_num_children(self):
        return self.num_children

    def get_child(self, idx: int):
        return mcall(self.selector(), idx)


class Threading:
    @staticmethod
    def get_factory():
        def factory(loop: Loop):
            return Threading(loop)
        return factory

    def __init__(self, loop):
        self.loop = loop

    def get_num_children(self):
        return 1

    def get_child(self, idx: int):
        parallel_loop = Loop.createAnonLoop()
        parallel_loop.subloops = self.loop.subloops
        pragma = f"#pragma clang loop({self.loop.name}) parallelize_thread"
        return parallel_loop, [pragma]


class Interchange:
    @staticmethod
    def get_factory():
        def factory(loop: Loop):
            return Interchange(loop)
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

        num_children = (n-1)
        for i in range(1, n-1):
            num_children *= i

        for d in range(1, n+1):
            def make_child(idx: int):
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

                newperm = [Loop.createLoop() for l in perm]
                for p, c in neighbors(newperm):
                    p.subloops = [c]
                newperm[-1].subloops = orignest[-1].subloops

                nestids = [p.name for p in orignest]
                permids = [p.name for p in perm]
                newpermids = [p.name for p in newperm]
                pragma = f"#pragma clang loop({','.join(nestids)}) interchange permutation({','.join(permids)}) permuted_ids({','.join(newpermids)})"
                return newperm[0], [pragma]
            yield num_children, make_child

    def get_num_children(self):
        return self.num_children

    def get_child(self, idx: int):
        return mcall(self.selector(), idx)


class Reversal:
    @staticmethod
    def get_factory():
        def factory(loop: Loop):
            return Reversal(loop)
        return factory

    def __init__(self, loop):
        self.loop = loop

    def get_num_children(self):
        return 1

    def get_child(self, idx: int):
        reversed_loop = Loop.createLoop()
        reversed_loop.subloops = self.loop.subloops
        pragma = f'#pragma clang loop({self.loop.name}) reverse reversed_id({reversed_loop.name})'
        return reversed_loop, [pragma]


def as_dot(baseexperiment: Experiment, max_depth=None, filter=None, decendfilter=None):
    yield 'digraph G {'
    yield '  rankdir=LR;'
    yield ''

    for experiment in baseexperiment.derivatives_recursive(max_depth=max_depth, filter=filter, descendfilter=decendfilter):
        desc = ''.join(l + '\\l' for l in experiment.to_lines())

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
        pass
    if args:
        example = Loop.createRoot()
        example.new_subloop()
        example.new_subloop().new_subloop()

        root = Experiment()
        root.nestexperiments.append(LoopNestExperiment(example, []))

        for line in as_dot(root, max_depth=args.maxdepth):
            print(line)
        return 0


def read_json(files):
    root = Experiment()
    for fn in files:
        with mkpath(fn).open() as fo:
            data = json.load(fo)
        loopnests = data["loopnests"]
        for ln in loopnests:
            nestroot = json_to_loops(ln["topmost"])
            exroot = LoopNestExperiment(nestroot, [])
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
    for ccf in ccargs.ccfiles:
        cmdline += ['-iquote', (ccargs.cwd / ccf).parent]
    cmdline += ['-mllvm', '-polly', '-mllvm', '-polly-process-unprofitable', '-mllvm', '-polly-position=early',
                '-mllvm', '-polly-reschedule=0', '-mllvm', '-polly-pattern-matching-based-opts=0']
    if debuginfo:
        # FIXME: loopnests.json overwritten if multiple files passed to clang
        cmdline += ['-g', '-gcolumn-info']
    if os.name == 'nt':
        cmdline += ['-l', r"C:\Users\meinersbur\build\llvm-project\release\lib\libomp.dll.lib"]
    else:
        cmdline += ['-fopenmp']
    cmdline += ['-mllvm', '-polly-omp-backend=LLVM']
    cmdline += ['-Werror=pass-failed']
    cmdline += extraflags
    cmdline += ['-o', outfile]
    return cmdline


def extract_loopnests(tempdir, ccargs, execargs):
    print("Extract loop nests...")

    extractloopnest = tempdir / 'base'
    extractloopnest.mkdir(parents=True, exist_ok=True)
    exefile = extractloopnest / ccargs.o.name

    cmdline = make_ccline(ccargs, outfile=exefile, 
        extraflags=['-mllvm', '-polly-output-loopnest=loopnests.json'], debuginfo=True)
    invoke.diag(*cmdline, cwd=extractloopnest, onerror=invoke.Invoke.EXCEPTION)

    loopnestfile = extractloopnest / 'loopnests.json'
    root = read_json(files=[loopnestfile])

    print("Time the output...")
    p = invoke.diag(exefile, timeout=execargs.timeout, 
        appendenv={'LD_LIBRARY_PATH': execargs.ld_library_path}, cwd=extractloopnest, onerror=invoke.Invoke.EXCEPTION)
    root.duration = p.walltime
    root.exppath = extractloopnest
    root.expnumber = 0
    root.depth = 1
    return root


expnumber = 1
def run_experiment(tempdir: pathlib.Path, experiment: Experiment, ccargs, execargs, writedot: bool, dotfilter, dotexpandfilter, root: Experiment):
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
            if loop.isroot:
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
        p = invoke.diag(exefile, cwd=expdir, onerror=invoke.Invoke.EXCEPTION, timeout=execargs.timeout,
                        std_prefixed=expdir / 'exec.txt', appendenv={'LD_LIBRARY_PATH': execargs.ld_library_path})
    except subprocess.TimeoutExpired:
        # Assume failure
        experiment.duration = math.inf
        return
    print(f"Execution completed in {p.walltime}")
    experiment.duration = p.walltime

    with logfile.open('a') as f:
        f.write(f"Execution completed in {p.walltime}\n")

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
        parser.add_argument('--ld-library-path', action='append')
        parser.add_argument('--outdir')
        parser.add_argument('--timeout', type=float,
                            help="Max exec time in seconds; default is no timout")
        parser.add_argument('ccline', nargs=argparse.REMAINDER)
    if args:
        ccargs = parse_cc_cmdline(args.ccline)

        execargs = argparse.Namespace()

        execargs.ld_library_path = None
        if args.ld_library_path != None:
            execargs.ld_library_path = ':'.join(args.ld_library_path)

        execargs.timeout = None
        if args.timeout != None:
            execargs.timeout = datetime.timedelta(seconds=args.timeout)

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

            root = extract_loopnests(d, ccargs=ccargs, execargs=execargs)
            print("Baseline is")
            print(root)
            print("")

            def priorotyfunc(x): 
                return -math.inf if x.duration is None else -x.duration.total_seconds()
            pq = PriorityQueue(root, key=priorotyfunc)
            closed = set()
            bestsofar = root

            csvlog.write(f"{root.expnumber},{root.duration.total_seconds()},{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
            newbestlog.write(f"{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")

            while not pq.empty():
                item = pq.top()

                if item.duration == None:
                    num_experiments += 1
                    run_experiment(d, item, ccargs=ccargs, execargs=execargs, 
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

                    maxdepth = item.depth+1
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

    subparsers = parser.add_subparsers(dest='subcommand')
    for cmd, func in commands.items():
        subparser = subparsers.add_parser(cmd)
        func(parser=subparser, args=None)
    args = parser.parse_args(str(v) for v in argv[1:])

    if args.tiling:
        tilesizes = [2, 4]
        if args.tiling_sizes != None:
            tilesizes = [int(s) for s in args.tiling_sizes.split(',')]
        transformers.append(Tiling.get_factory(tilesizes))
    if args.threading:
        transformers.append(Threading.get_factory())
    if args.interchange:
        transformers.append(Interchange.get_factory())
    if args.reversal:
        transformers.append(Reversal.get_factory())

    cmdlet = commands.get(args.subcommand)
    if not cmdlet:
        die("No command?")
    return cmdlet(parser=None, args=args)


if __name__ == '__main__':
    if errcode := main(argv=sys.argv):
        exit(errcode)
