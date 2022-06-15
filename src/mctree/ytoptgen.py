#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from mctree import *
import pathlib
from .tool.support import *

def gen_ytopt_problem(filename, output: pathlib.Path, max_depth):
    root = read_json(files=filename)

   
    
    experiments = []
    for experiment in root.derivatives_recursive(max_depth=max_depth):
        #p = len(cs)
        #s = f'p{p} = CS.CategoricalHyperparameter(name="p{p}", choices=[])'
        #cs.append(s)
        experiments.append(experiment)

    
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
        params = ["p0"]
        choices = []
        for e in experiments:
            if e == root:
                choice = ""
            else:
                pragmas = []
                for nestex in e.nestexperiments:
                    pragmas.extend(nestex.pragmalist)
                choice = "\n".join(pragmas)
            choices.append(choice)
        #
        f.write(f"p0 = CSH.CategoricalHyperparameter(name='p0', choices={choices}, default_value='')\n")

        f.write(f"cs.add_hyperparameters({params})")

        f.write(r"""input_space = cs
output_space = Space([Real(0.0, inf, name='time')])
""")







