#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

script = Path(sys.argv[0]).absolute()
thisscript = Path(__file__)

sys.path.insert(0,str( (thisscript.parent.parent / 'src').absolute() ))
from mctree.__main__ import main



if errcode := main(argv=sys.argv):
    exit(errcode)
