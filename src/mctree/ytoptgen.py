#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from mctree import *
import mctree 
import pathlib
from .tool.support import *

escaperules = { "'": r"\'", '\n': r'\n' }
def pyescape(s):
    if isinstance(s,str):
        return '"' + ''.join(escaperules.get(c) if c in escaperules else c for c in s) + '"'
    return str(s)


def pylist(l):
    return '[' + ', '.join(pyescape(e) for e in l) + ']'


p = 0
params = []
experiment_to_param = dict()
def param_for_experiment(experiment):
    global p, params, experiment_to_param
    if param := experiment_to_param.get(experiment):
        return param
    param = f"p{p}_loopnest"
    experiment_to_param[experiment] = param
    params.append(param)
    p += 1
    return param


# TODO: Consolidate with same functionality from run_experiment 
def prepare_cfiles(experiment, outdir):
    ccfiles = set()

    for x in experiment.nestexperiments:
        loops = set()
        rootloopnestexperiment = x
        while rootloopnestexperiment.derived_from != None:
            rootloopnestexperiment = rootloopnestexperiment.derived_from

        for loop in rootloopnestexperiment.loopnest.subloops_recursive():
            if  loop.filename :
                ccfiles.add( mkpath(loop.filename))


    contents = {}
    for f in ccfiles:
        contents[f.resolve()] = f.open('r').readlines()

    for x in experiment.nestexperiments:
        first = None
        rootloopnestexperiment = x
        while rootloopnestexperiment.derived_from != None:
            rootloopnestexperiment = rootloopnestexperiment.derived_from

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
        paramname = param_for_experiment(rootloopnestexperiment) 
        newline = oldline[:first[1]] + '\n#' + paramname+ '\n'  + oldline[first[1]:] 
        contentlines[first[0]] = newline

    # Writeback files in new dir
    newccfiles = []
    for k, content in contents.items():
        filename = outdir / k.name
        createfile(filename, ''.join(content))
        newccfiles.append(filename)
    return newccfiles




def gen_ytopt_problem(filename, outdir: pathlib.Path, max_depth):
    root = read_json(files=filename)

    outdir.mkdir(parents=True,exist_ok=True)
    output = outdir / 'ytopt.py'

    global params
    conditions = []
    conds = []


    

    newccfiles = prepare_cfiles(root, outdir)



    with output.open('w+') as f:
        f.write(r"""#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, math
import numpy as np
from autotune import TuningProblem
from autotune.space import *
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical
from plopper import *

cs = CS.ConfigurationSpace(seed=1234)
""")

        # TODO: One for each nestexperiment
        for experiment in root.derivatives_recursive(max_depth=max_depth):
            param = None
            assert len(experiment.nestexperiments)==1
            for cne in experiment.nestexperiments: 
                param = param_for_experiment(cne)

                eparams = cne.newparams
                for ep in eparams:
                    f.write(f"{ep.name} = CSH.CategoricalHyperparameter(name='{ep.name}', choices={ep.choices}, default_value={pyescape(ep.choices[0])})\n") 
                    params.append(ep.name)


            choices = []
            for c in experiment.derivatives_recursive(max_depth=1):
                if c is experiment :
                    continue
                for cne in c.nestexperiments:                
                    addedpragmas = '\n'.join(cne.newpragmas)
                    choice = addedpragmas
                    if experiment.depth < max_depth:
                        cparam = param_for_experiment(cne)
                        choice = f"#{cparam}\n{addedpragmas}"
                    choices.append(choice)
            
            f.write(f"{param} = CSH.CategoricalHyperparameter(name='{param}', choices={pylist(choices)}, default_value='')\n") 


        f.write(f"cs.add_hyperparameters({', '.join(params)})\n")
        f.write("\n")
        f.write(f'sourcefile = {pyescape(str(newccfiles))}\n') # TODO: More than one file

        f.write(r"""
input_space = cs
output_space = Space([Real(0.0, inf, name='time')])

dir_path = os.path.dirname(os.path.realpath(sourcefile))
obj = Plopper(sourcefile,dir_path)

x1=['p0','p1','p2','p3']
def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]]]
        print('CONFIG:',point)
        params = ["P0","P1","P2","P3"]
        result = obj.findRuntime(value, params)
        return result
    x = np.array([point[f'p{i}'] for i in range(len(point))])
    results = plopper_func(x)
    print('OUTPUT:%f',results)
    return results

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None)
""")







