#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configparser

from .graph import *
from .operations import *
from .session import *
from .train import *

config = configparser.ConfigParser()
config.read('worker.conf')
SERVER_HOST = config['DEFAULT']['SERVER_HOST']
SERVER_PORT = config['DEFAULT'].getint('SERVER_PORT')

# Create a default graph.
import builtins
DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()
DEFAULT_PS = builtins.DEFAULT_PS = WorkerClient(SERVER_HOST, SERVER_PORT)

