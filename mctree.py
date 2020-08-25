#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import itertools
import json
import pathlib
import tempfile
import heapq
import contextlib
import subprocess
import math
import datetime
import csv
import tool.invoke as invoke
from tool.support import *


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
        self.transformable = True
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
            if not result[-1].subloops[0].transformable:
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
        sublooproot = json_to_loops(tm["subloops"])
        loop.subloops = sublooproot.subloops
        result.add_subloop(loop)
    return result


def transform_node(root, callback):
    def do_transform_node(loop, childids, callback):
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
        self.has_expanded = False
        self.duration = None
        self.exppath = None
        self.expnumber = None
        self.depth = None

    def clone(self):
        result = Experiment()
        result.nestexperiments = self.nestexperiments.copy()
        return result

    def add_subexperiment(self, subexp):
        subexp.derived_from = self
        subexp.depth = self.depth+1
        self.derivatives.append(subexp)

    def derivatives_recursive(self):
        yield self
        for n in self.derivatives:
            yield from n.derivatives_recursive()

    def __str__(self):
        return '\n'.join(self.to_lines())

    def to_lines(self):
        if self.expnumber != None:
            yield f"Experiment {self.expnumber}"
        if self.duration != None:
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
    tilesizes = [2, 4]

    @staticmethod
    def do_subtile(loop):
        # End tiling here
        yield [], [], [], loop.subloops, []

        if len(loop.subloops) == 1 and loop.subloops[0].transformable:
            yield from Tiling.do_tile(loop.subloops[0])

    @staticmethod
    def do_tile(loop):
        for origloops, subfloors, subtiles, subbody, subsizes in Tiling.do_subtile(loop):
            for tilesize in Tiling.tilesizes:
                yield [loop] + origloops, [Loop.createLoop()] + subfloors, [Loop.createLoop()] + subtiles, subbody, [tilesize] + subsizes

    @staticmethod
    def gen_tiling(loop: Loop):
        for origloops, floors, tiles, body, sizes in Tiling.do_tile(loop):
            cur = floors[0]
            for floor in floors[1:]:
                cur.subloops = [floor]
                cur = floor
            for tile in tiles:
                cur.subloops = [tile]
                cur = tile
            cur.subloops = body

            origloopids = [l.name for l in origloops] 
            floorids = [floor.name for floor in floors]
            tileids = [tile.name for tile in tiles]
            sizes = [str(s) for s in sizes]
            pragma = f"#pragma clang loop({','.join(origloopids)}) tile sizes({','.join(sizes)}) floor_ids({','.join(floorids)}) tile_ids({','.join(tileids)})"
            yield floors[0], [pragma]


class Threading:
    @staticmethod
    def gen_threading(loop: Loop):
        parallel_loop = Loop.createLoop()
        parallel_loop.transformable = False
        parallel_loop.subloops = loop.subloops
        #pragma = f"#pragma clang loop({loop.name}) parallelize_thread parallel_id({parallel_loop.name})"
        #yield parallel_loop, [pragma]
        pragma = f"#pragma clang loop({loop.name}) parallelize_thread"
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

            nestids = [p.name for p in nests]
            permids = [p.name for p in perm]
            newpermids = [p.name for p in newperm]
            pragma = f"#pragma clang loop({','.join(nestids)}) interchange permutation({','.join(permids)}) permuted_ids({','.join(newpermids)})"
            yield newperm[0], [pragma]


tiling_enabled = True
threading_enabled = True
interchange_enabled = True


def do_transformations(loop):
    if not loop.transformable:
        return

    if tiling_enabled:
        yield from Tiling.gen_tiling(loop)

    if threading_enabled:
        yield from Threading.gen_threading(loop)

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

    if not baseexperiment.has_expanded:
        derive_expriments(baseexperiment)
        baseexperiment.has_expanded = True
    for e in baseexperiment.derivatives:
        expand_searchtree(e, remaining_depth=remaining_depth-1)



def as_dot(baseexperiment: LoopNestExperiment):
    yield "digraph G {"
    yield "rankdir=LR;"

    for experiment in baseexperiment.derivatives_recursive():
        desc = ''.join(l + "\\l" for l in experiment.to_lines())

        if experiment.duration == math.inf:
            fillcolor = "lightpink:crimson"
        elif experiment.duration != None:
            fillcolor = "darkseagreen1:lawngreen"
        else:
            fillcolor = "azure:powderblue"

        yield f'n{id(experiment)}[shape=box color="grey30" penwidth=2 fillcolor="{fillcolor}" style="filled,rounded" gradientangle=315 fontname="Calibri Light" label="{desc}"];'

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
        expand_searchtree(root, remaining_depth=args.maxdepth)
        for line in as_dot(root):
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
    cmdline += ['-mllvm', '-polly', '-mllvm', '-polly-process-unprofitable', '-mllvm', '-polly-position=early', '-mllvm', '-polly-reschedule=0', '-mllvm', '-polly-pattern-matching-based-opts=0']
    if debuginfo:
        cmdline += ['-g', '-gcolumn-info'] # FIXME: loopnests.json overwritten if multiple files passed to clang
    if os.name == 'nt':
        cmdline += ['-l', r"C:\Users\meinersbur\build\llvm-project\release\lib\libomp.dll.lib"]
    else:
        cmdline += ['-fopenmp']
    cmdline += ['-mllvm', '-polly-omp-backend=LLVM']
    cmdline += ['-Werror=pass-failed']
    cmdline += extraflags
    cmdline += ['-o', outfile]
    return cmdline



def extract_loopnests(tempdir,ccargs,execargs):
    print("Extract loop nests...")

    extractloopnest = tempdir / 'base'
    extractloopnest.mkdir(parents=True,exist_ok=True)
    exefile = extractloopnest / ccargs.o.name

    cmdline = make_ccline(ccargs, outfile=exefile, extraflags=['-mllvm', '-polly-output-loopnest=loopnests.json'], debuginfo=True)
    invoke.diag(*cmdline, cwd=extractloopnest,onerror=invoke.Invoke.EXCEPTION)
    
    loopnestfile = extractloopnest / 'loopnests.json'
    root = read_json(files=[loopnestfile])

    print("Time the output...")
    p = invoke.diag(exefile, timeout=execargs.timeout, appendenv={'LD_LIBRARY_PATH': execargs.ld_library_path}, cwd=extractloopnest, onerror=invoke.Invoke.EXCEPTION)
    root.duration = p.walltime
    root.exppath = extractloopnest
    root.expnumber = 0
    root.depth=1
    return root


expnumber = 1
def run_experiment(tempdir:pathlib.Path, experiment: Experiment, ccargs,execargs,writedot:bool,root:Experiment):
    global expnumber
    expdir = tempdir / f"experiment{expnumber}"
    experiment.expnumber = expnumber
    logfile =  expdir / 'desc.txt'
    dotfile = expdir / 'graph.dot'
    expnumber +=1 
    expdir.mkdir(parents=True,exist_ok=True)
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
            rootloopnestexperiment  = rootloopnestexperiment.derived_from 

        first = None

        for loop in rootloopnestexperiment.loopnest.subloops_recursive():
            if loop.isroot:
                continue
            filename = mkpath(loop.filename).resolve()
            line = loop.line-1     # is one-based
            column = loop.column-1 # is one-based
            name = loop.name

            if (first == None) or (first > (line,column)):
                first = (line,column)

            contentlines = contents.get(filename)
            assert contentlines,"Loopnest's file not matched with intput file?!?"

            oldline = contentlines[line]    
            newline = oldline[:column] + f"\n#pragma clang loop id({name})\n" + oldline[column:] # FIXME: if multiple loops per line, ensure that later are replaced first
            contentlines[line] = newline

        oldline = contentlines[first[0]]
        newline = oldline[:first[1]] + '\n' + ''.join(s + '\n' for s in reversed(x.pragmalist)) + oldline[first[1]:] 
        contentlines[first[0]] = newline


    # Writeback files in new dir
    newccfiles = []
    for k,content in contents.items():
        filename = expdir / k.name
        createfile(filename, ''.join(content))
        newccfiles.append(filename)
        pass

    exefile = expdir / ccargs.o.name
    cmdline = make_ccline(ccargs, ccfiles=newccfiles, outfile = exefile)
    try:
        invoke.diag(*cmdline, cwd=expdir, onerror=invoke.Invoke.EXCEPTION, std_prefixed=expdir / 'cc.txt')
    except subprocess.CalledProcessError:
        # Compilation failed; illegal program
        experiment.duration = math.inf
        return

    try:
        p = invoke.diag(exefile, cwd=expdir, onerror=invoke.Invoke.EXCEPTION,timeout=execargs.timeout, std_prefixed=expdir / 'exec.txt', appendenv={'LD_LIBRARY_PATH': execargs.ld_library_path})
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
                        for line in as_dot(root):
                            print(line,file=f)




def parse_cc_cmdline(cmdline):
    cc = cmdline[0]
    ccflags = []
    ccfiles = []
    o = None

    i = 1
    while i < len(cmdline):
        arg = cmdline[i]
        nextarg = cmdline[i+1] if i +1 <len(cmdline) else None

        if arg == '-o' and nextarg !=None:
            o = mkpath(nextarg) 
            i+=2
            continue

        if arg.startswith('-'):
              ccflags.append(arg)
              i+=1
              continue
            
        fn = mkpath(arg)
        if not fn.suffix.lower() in {'.c', '.cxx', '.cpp'}:
              ccflags.append(arg)
              i+=1
              continue

        if not fn.is_file():
              ccflags.append(arg)
              i+=1
              continue

        ccfiles.append(fn)
        i+=1

    result = argparse.Namespace()
    result.cwd = pathlib.Path.cwd()
    result.cc = cc
    result.ccfiles = [f.resolve() for f in ccfiles]
    result.o = o
    result.ccflags = ccflags
    return result





class PriorityQueue:    
    class Item:
        def __init__(self,priority:float,item):
            self.priority= priority
            self.item = item

        def __eq__(self, other):
            return self.priority == other.priority
        def __ne__(self, other):
            return self.priority != other.priority
        def __gt__(self, other):
            return  self.priority > other.priority
        def __lt__(self, other):
            return self.priority < other.priority
        def __ge__(self, other):
            return self.priority >= other.priority
        def __le__(self, other):
            return self.priority <= other.priority

    def __init__(self,*args,key=lambda x:x):
        self.key = key
        self.elts = [PriorityQueue.Item(key(x),x) for x in args]
        heapq.heapify(self.elts)

    def push(self, item):
        heapq.heappush(self.elts, PriorityQueue.Item(self.key(item), item))

    def pop(self):
       return heapq.heappop(self.elts).item

    def top(self):
        return self.elts[0].item

    def replace(self,item):
        heapq.heapreplace(self.elts, PriorityQueue.Item(self.key(item), item))

    def update(self):
        self.replace(self.top())

    def __len__(self):
        return len(self.elts)

    def empty(self):
        return len(self.elts)==0


@subcommand("autotune")
def autotune(parser, args):
    if parser:
        add_boolean_argument(parser, 'keep')
        parser.add_argument('--ld-library-path', action='append')
        parser.add_argument('--outdir')
        parser.add_argument('--timeout', type=float, help="Max exec time in seconds; default is no timout")
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


        with contextlib.ExitStack() as stack:
            if args.keep:
                d = tempfile.mkdtemp(dir=outdir,prefix='mctree-')
            else:
                d = stack.enter_context(tempfile.TemporaryDirectory(dir=outdir,prefix='mctree-'))
            d = mkpath(d)
            
            bestfile = d / 'best.txt'
            #dotfile = d / 'graph.dot'
            csvfile =  d / 'experiments.csv'
            newbestcsvfile = d / 'newbest.csv'
            csvlog = csvfile.open('w+')
            newbestlog = newbestcsvfile.open('w+')




            root = extract_loopnests(d,ccargs= ccargs,execargs= execargs)
            print("Baseline is")
            print(root)
            print("")

            priorotyfunc = lambda x: -math.inf if x.duration is None else -x.duration.total_seconds()
            pq = PriorityQueue(root,key=priorotyfunc)
            bestsofar = root

            csvlog.write(f"{root.expnumber},{root.duration.total_seconds()},{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
            newbestlog.write(f"{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")


            while not pq.empty():
                item = pq.top()

                if item.duration == None:
                    run_experiment(d, item, ccargs=ccargs,execargs=execargs,writedot=maxdepth<=4,root=root)
                    if item.duration == math.inf:
                        # Invalid pragmas? Remove experiment entirely
                        print("Experiment failed")
                        pq.pop()
                        continue

                    pq.update()
                    if bestsofar.duration > item.duration:
                        print(f"New experiment better than old (old: {bestsofar.duration}, new: {item.duration})")
                        print(f"Path {item.exppath}\n")
                        print(item)
                        bestsofar = item
                        with bestfile.open('w+') as f:
                            f.write(f"Best experiment so far\n")
                            f.write(f"Time: {bestsofar.duration}\n")
                            f.write(f"Path: {bestsofar.exppath}\n\n")
                            for line in  bestsofar.to_lines():
                                f.write(line)
                                f.write('\n')
                        newbestlog.write(f"{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
                        newbestlog.flush()
                    csvlog.write(f"{item.expnumber},{item.duration.total_seconds()},{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
                    csvlog.flush()
                    continue

                if not item.has_expanded:
                    print(f"Selecting best experiment {item.duration} for expansion")
                    expand_searchtree(item, remaining_depth=1)
                    for child in item.derivatives:
                        pq.push(child)

                    maxdepth = item.depth+1



                if item.has_expanded and item.duration != None:
                    pq.pop()

            print("No more experiments!!?")





def main(argv: str) -> int:
    global tiling_enabled, threading_enabled, interchange_enabled
    parser = argparse.ArgumentParser(
        description="Loop transformation search tree proof-of-concept", allow_abbrev=False)

    parser.add_argument('--maxdepth', type=int, default=2)
    add_boolean_argument(parser, "--tiling", default=True)
    parser.add_argument('--tiling-sizes')
    add_boolean_argument(parser, "--threading", default=True)
    add_boolean_argument(parser, "--interchange", default=True)

    subparsers = parser.add_subparsers(dest='subcommand')
    for cmd, func in commands.items():
        subparser = subparsers.add_parser(cmd)
        func(parser=subparser, args=None)
    args = parser.parse_args(str(v) for v in argv[1:])

    tiling_enabled = args.tiling
    if  args.tiling_sizes!=None:
        Tiling.tilesizes = [int(s) for s in args.tiling_sizes.split(',')]
    threading_enabled = args.threading
    interchange_enabled = args.interchange

    cmdlet = commands.get(args.subcommand)
    if not cmdlet:
        die("No command?")
    return cmdlet(parser=None, args=args)


if __name__ == '__main__':
    if errcode := main(argv=sys.argv):
        exit(errcode)
