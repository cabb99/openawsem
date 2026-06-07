from pathlib import Path

__location__= Path(__file__).resolve().parent
from .DataPath import DataPath
from .myFunctions import *
from openawsem.memory.projects import create_fragment_memories, create_single_memory
from .create_debyeHuckel import generate_charge_array