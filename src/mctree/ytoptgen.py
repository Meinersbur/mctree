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
    return '[' + ', '.join(pyescape(e for e in l)) + ']'


def gen_ytopt_problem(filename, output: pathlib.Path, max_depth):
    root = read_json(files=filename)

   
    p = 0
    params = []
    experiment_to_param = dict()
    def param_for_experiment(experiment):
        nonlocal p, params, experiment_to_param
        if param := experiment_to_param.get(experiment):
            return param
        param = f"p{p}_loopnest"
        experiment_to_param[experiment] = param
        params.append(param)
        p += 1
        return param
    

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
                    choices.append(pyescape(choice))
            
            f.write(f"{param} = CSH.CategoricalHyperparameter(name='{param}', choices={pylist(choices)}, default_value='')\n") 


        f.write(f"cs.add_hyperparameters({', '.join(params)})\n")

        f.write(r"""
input_space = cs
output_space = Space([Real(0.0, inf, name='time')])
""")







