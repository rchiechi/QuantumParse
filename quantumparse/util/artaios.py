#!/usr/bin/env python3

import sys
import os
import re
import asyncio
import stat
import shutil
import argparse
from collections import OrderedDict
from pathlib import Path

## Artaios binary
ARTAIOSBIN = os.getenv('ARTAIOSBIN', os.path.join(os.path.expanduser('~'),"source/artaios/bin/artaios"))
##


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parallel Artaios Transport Calculation')
    
    # Input file argument
    parser.add_argument('input_file', 
                        nargs='?', 
                        default='transport.in',
                        help='Input file for transport calculation (default: transport.in)')
    
    # Optional arguments can be added here
    parser.add_argument('-n', '--nthreads', 
                        type=int,
                        default=10,
                        help='Number of simltanous jobs to use (overrides automatic detection)')
    
    parser.add_argument('--artaios',
                        default=ARTAIOSBIN,
                        type=str, 
                        help='Artaios executable')

    # Parse arguments
    args = parser.parse_args()

    return args


async def run_command_in_directory(command, directory):
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=directory,
    )

    async def read_stream(stream, prefix):
        while line := await stream.readline():
            print(f"Thread {Path(directory).stem}: {line.decode().rstrip()}")

    await asyncio.gather(
        read_stream(process.stdout, "Stdout"),
        read_stream(process.stderr, "Stderr"),
    )

    returncode = await process.wait()  # Wait for the process to finish
    return returncode, directory


# $partitioning
#        totnbas  479
#        leftbas 0-0 #CHECK THIS!
#        centralbas 1-479 #CHECK THIS!
#        rightbas 0-0 #CHECK THIS!
#     $end
#     $energy_range
#       start  -8.0
#        end     -1.0
#        steps 200
#     $end
#     $system
#        nspin  1
#     $end
#     $electrodes
#        self_energy wbl
#        dos_s 0.036
#        fermi_level -5.0
#     $end
#     $general
#       do_transport
#       unit   eV
#       modelham
#       loewdin_central
#       qcprog gen
#     $end
# 
#     $subsystem
#       print_molden T
#       print_diag_central T
#       do_diag_central T
#       moldeninfile C10.molden.input
#     $end


class InputFile:
    
    allowed_fields = ('partitioning',
                      'energy_range',
                      'system',
                      'electrodes',
                      'general')
    
    def __init__(self, input_path):
        self.ranged_input_lines = []
        self.energy_range = {"start": None,
                             "end": None,
                             "steps": None}
        self.parse_input_file(input_path)
        
    def parse_input_file(self, input_path):
        infield = ''
        input_lines = {}
        for field in self.allowed_fields:
            input_lines[field] = {}
        with input_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] == '#':
                    continue
                if line in [f"${key}" for key in self.allowed_fields]:
                    infield = line.replace('$', '')
                    continue
                elif line == "$end":
                    infield = ''
                    continue
                parts = line.split()
                if infield:
                    if len(parts) > 1:
                        key = parts[0]
                        if parts[1].isdigit():
                            val = int(parts[1])
                        else:
                            try:
                                val = float(parts[1])
                            except ValueError:
                                val = parts[1]
                        input_lines[infield][key] = val
                        if infield == 'energy_range':
                            self.energy_range[key] = val
                    else:
                        input_lines[infield][line] = ''
        self.input_lines = input_lines

    def _emit_input_file(self):
        in_energy_range = False
        input_lines = []
        for field, line in self.input_lines.items():
            input_lines.append(f"${field}")
            if field == 'energy_range':
                for key, val in self.energy_range.items():
                    input_lines.append(f"{key} {val}")
            else:
                for key, val in line.items():
                    input_lines.append(f"{key} {val}")
            input_lines.append("$end")
        return input_lines
    
    @property
    def energy_start(self):
        return self.energy_range["start"]
    @energy_start.setter
    def energy_start(self, val:float):
        self.energy_range["start"] = val
    @property
    def energy_end(self):
        return self.energy_range["end"]
    @energy_end.setter
    def energy_end(self, val:float):
        self.energy_range["end"] = val
    @property
    def energy_steps(self):
        return self.energy_range["steps"]
    @energy_steps.setter
    def energy_steps(self, val:float):
        self.energy_range["steps"] = val
    @property
    def filecontents(self):
        return "\n".join(self.input_lines)
    @property
    def ranged_inputs(self):
        return [ "\n".join(lines) for lines in self.ranged_input_lines ]
    @property
    def parsed(self):
        return bool(len(self.ranged_inputs))

    def distribute_energy_ranges(self, nthreads):
        """
        Distributes an energy range into sets for parallel processing.
    
        Args:
            energy_start (float): The starting energy value.
            energy_end (float): The ending energy value.
            steps (int): The total number of steps.
            nthreads (int): The number of threads.
    
        Returns:
            tuple: A tuple containing two lists, energy_starts and energy_ends.
        """
        if self.energy_steps % nthreads != 0:
            raise ValueError("Steps must be divisible by nthreads.")
    
        steps_per_thread = self.energy_steps // nthreads
        energy_range = self.energy_end - self.energy_start
        energy_increment = energy_range / self.energy_steps
        thread_energy_increment = steps_per_thread * energy_increment
    
        energy_starts = []
        energy_ends = []
        
        
        self.energy_steps = steps_per_thread
        current_start = self.energy_start
        end_energy = self.energy_end
        for i in range(nthreads -1):
            self.energy_start = round(current_start, 2)
            current_end = self.energy_start + thread_energy_increment
            self.energy_end = round(current_end, 2)
            current_start = current_end + energy_increment
            self.ranged_input_lines.append(self._emit_input_file())
            print(f"Thread {i+1:02d}: {self.energy_start} -> {self.energy_end}")

        self.energy_start = current_start
        self.energy_end = end_energy
        self.ranged_input_lines.append(self._emit_input_file())
        print(f"Thread {nthreads:02d}: {self.energy_start} -> {self.energy_end}")
        print(f"{self.energy_steps} steps / thread")
        print(f"Increments: {thread_energy_increment}")
        print(f"{thread_energy_increment * self.energy_steps}")
        if len(self.ranged_input_lines) != nthreads:
            raise ValueError(f"Invalid number of steps generated: {len(self.ranged_inputs)} != {nthreads}")


######### Main () ############################

async def main():
    
    args = parse_arguments()
    
    if not args.input_file:
        print("No input file provided")
        return
    
    if not Path(args.artaios).exists():
        print(f"{args.artaios} does not exist.")
        return
    
    inputfile = InputFile(Path(args.input_file))
    
    while inputfile.energy_steps % args.nthreads:
        print(args.nthreads % inputfile.energy_steps)
        args.nthreads -= 1
    if args.nthreads > 1:
        print(f"Dividing {inputfile.energy_steps} steps into {args.nthreads} threads.")
    else:
        print(f"Not enough energy steps to parallelize.")
        return

    
    inputfile.distribute_energy_ranges(args.nthreads)
    tempdir = Path('artaios_parallel')
    supfiles = [Path('hamiltonian.1'), Path('overlap')]
    exec_dirs = []

    try:
        tempdir.mkdir()
        for i, ranged_input in enumerate(inputfile.ranged_inputs):
            exec_dirs.append(Path(tempdir / str(i).zfill(2)))
            exec_dirs[-1].mkdir()
            for f in supfiles:
                shutil.copy(f, exec_dirs[-1] / f)
            Path(exec_dirs[-1] / "transport.in").write_text(ranged_input)
                
    except FileExistsError:
        print(f"{tempdir} already exists, clean it up before running {sys.argv[0]}")
        return
    except Exception as e:
        print(f"Unhandled error while creating {tempdir}: {e}")
        return

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run_command_in_directory([args.artaios, "transport.in"], directory)) for directory in exec_dirs]

    for task in tasks:
        returncode, directory = task.result()
        print(f"Return Code: {returncode} from {directory}")
