#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
import contextlib
import argparse
from mctree import *
import mctree.tool.invoke as invoke
from mctree.tool.support import *
import mctree 

# Decorator
commands = {}
def subcommand(name):
    def command_func(_func):
        global commands
        commands[name] = _func
        return _func
    return command_func


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
        mctree_autotune(ccline = args.ccline, exec_args=shcombine(arg=args.exec_arg,args=args.exec_args),polybench_time=args.polybench_time,ld_library_path=args.ld_library_path,timeout=args.timeout,outdir=args.outdir,keep=args.keep)


def mctree_autotune(ccline,exec_args,polybench_time,ld_library_path,timeout,outdir,keep):
        ccargs = parse_cc_cmdline(ccline)
        ccargs.polybench_time = polybench_time

        execopts = argparse.Namespace()

        execopts.ld_library_path = None
        if ld_library_path != None:
            execopts.ld_library_path = ':'.join(ld_library_path)

        execopts.timeout = None
        if timeout != None:
            execopts.timeout = datetime.timedelta(seconds=timeout)
        execopts.polybench_time = polybench_time

        execopts.args = exec_args

        outdir = mkpath(outdir)
        num_experiments = 0

        with contextlib.ExitStack() as stack:
            if keep:
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

            bestsofar = root
            csvlog.write(f"{root.expnumber},{root.duration.total_seconds()},{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
            newbestlog.write(f"{bestsofar.expnumber},{bestsofar.duration.total_seconds()}\n")
            def update_best(item):
                nonlocal bestsofar
                if item.duration == math.inf:
                        return 

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

            if ccs:
                ts = experiment_as_ccs(root)
                os = ccs.ObjectiveSpace(name = "ospace")
                v1 = ccs.NumericalHyperparameter(lower = float(0), upper = float('inf'))
                os.add_hyperparameters([v1])
                e1 = ccs.Variable(hyperparameter = v1)
                os.add_objectives( { e1: ccs.MINIMIZE } )
                
                t = ccs.RandomTreeTuner(name = "random search", tree_space = ts, objective_space = os)
                rootev = ccs.TreeEvaluation(objective_space = os, configuration = ccs.TreeConfiguration(tree_space = ts, position = []), values = [root.duration.total_seconds()] )
                t.tell([rootev])

                while True:
                    next = t.ask()[0]
                    item = next.node.user_data
                    print("Selected:", next.position)

                    if item.duration !=None:                        
                        print("Already evaluated:",item)
                        continue 

                    num_experiments += 1
                    run_experiment(d, item, ccargs=ccargs, execopts=execopts, 
                        writedot=False, 
                        dotfilter=None,
                        dotexpandfilter=None,
                        root=root)
                    if item.duration == math.inf:
                        print("Experiment failed")
                        ev = ccs.TreeEvaluation(objective_space = os, configuration = next, values = [math.inf] )
                        next.node.weight = 0
                        next.node.bias = 0
                    else:
                        ev = ccs.TreeEvaluation(objective_space = os, configuration = next, values = [item.duration.total_seconds()] )
                    t.tell([ev])
                    update_best(item)
            else:
                def priorotyfunc(x): 
                    return -math.inf if x.duration is None else x.duration.total_seconds()
                pq = PriorityQueue(root, key=priorotyfunc)
                closed = set()

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
                        update_best(item)
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
        root.nestexperiments.append(LoopNestExperiment(None, example, [],[], loopcounter=loopcounter))
        
        if ccs:
            ts = experiment_as_ccs(root)
            for line in as_dot_from_ccs(ts, loopneststructure=args.loopneststructure):
                print(line)
        else:
            for line in as_dot(root, loopneststructure=args.loopneststructure):
                print(line)
        return 0



@subcommand("jsonfile")
def jsonfile(parser, args):
    if parser:
        parser.add_argument('filename', nargs='+')
    if args:
        root = read_json(files=args.filename)

        if ccs:
            ts = experiment_as_ccs(root, max_depth=args.maxdepth)
            for line in as_dot_from_ccs(ts):
                print(line)
        else:
            for line in as_dot(root, max_depth=args.maxdepth):
                print(line)
        return 0




import mctree.ytoptgen as ytoptgen

@subcommand("ytopt-problem")
def ytopt(parser, args):
    if parser:
        parser.add_argument('filename', nargs='+')
        parser.add_argument('--outdir',type=pathlib.Path)
    if args:
        ytoptgen.gen_ytopt_problem(filename=args.filename,outdir=args.outdir, max_depth=args.maxdepth)



def main(argv: str) -> int:
    global transformers,ccs
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
    #add_boolean_argument(parser, "--unrolling-and-jam-full", default=True)
    parser.add_argument('--unrolling-and-jam-factors')
    parser.add_argument('--packing-arrays',action='append')
    add_boolean_argument(parser, "--fission", default=True)
    add_boolean_argument(parser, "--fusion", default=True)

    add_boolean_argument(parser, "--parametric", default=False)
    add_boolean_argument(parser, "--ccs", default=False)

    subparsers = parser.add_subparsers(dest='subcommand')
    for cmd, func in commands.items():
        subparser = subparsers.add_parser(cmd)
        func(parser=subparser, args=None)
    args = parser.parse_args(str(v) for v in argv[1:])

    if args.ccs:
        ccs = enable_ccs()

    set_maxdepth(args.maxdepth)


    if args.tiling:
        tilesizes = [4,16]
        if args.tiling_sizes != None:
            tilesizes = [int(s) for s in args.tiling_sizes.split(',')]
        if args.parametric:
            transformers.append(TilingParametric.get_factory(tilesizes))
        else:
            transformers.append(Tiling.get_factory(tilesizes))
    if args.threading:
        transformers.append(Threading.get_factory())
    if args.interchange:
        transformers.append(Interchange.get_factory())
    if args.reversal:
        transformers.append(Reversal.get_factory())
    if args.unrolling:
        if args.unrolling_full:
            transformers.append(UnrollingFull.get_factory())   

        factors = [2, 4, 8]
        if args.unrolling_factors != None:
            factors = [int(s) for s in args.unrolling_factors.split(',')]
        if args.parametric:
            transformers.append(UnrollingParametric.get_factory(factors))
        else:
            transformers.append(Unrolling.get_factory(factors))
    if args.unrolling_and_jam:
        factors = [2, 4, 8]
        if args.unrolling_and_jam_factors != None:
            factors = [int(s) for s in args.unrolling_and_jam_factors.split(',')]
        if args.parametric:
            transformers.append(UnrollingAndJam.get_factory(factors))
        else:
            transformers.append(UnrollingAndJamParametric.get_factory(factors))
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
