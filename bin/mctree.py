#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from mctree import main

if errcode := main.main(argv=sys.argv):
    exit(errcode)
