#!/usr/bin/env python3
import os
import argparse


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", action='store', help='.inp file(s) sumbit to the queue', nargs='+')
    parser.add_argument("-ca", action='store_true', default=False,
                        help='Copy all of the files in the current dir')
    parser.add_argument("-cs", action='store_true', default=False,
                        help="Copy all files from the scratch directory")
    parser.add_argument("-notrashtmp", action='store_true', default=False,
                        help="Don't trash the temporary files that may be generated")
    parser.add_argument("-np", type=int, default=0, help="Override the number of cores specified in the input file")
    parser.add_argument("-gbw", action='store_true', default=False,
                        help="Copy corresponding gbw file to scratch dir")

    return parser.parse_args()


def print_bash_script(inp_filename, ca, cs, notrashtmp, np, gbw):

    sh_filename = str(inp_filename.replace('.inp', '.sh'))
    gbw_filename = str(inp_filename.replace('.inp', '_old.gbw'))

    keyword_line = ''
    with open(inp_filename, 'r') as inp_file:
        for line in inp_file:
            if line.startswith('!'):
                keyword_line = line

    nprocs = 1

    for item in keyword_line.split():
        if item.startswith("PAL"):
            nprocs = item[3:]

    if np != 0:
        nprocs = np

    with open(sh_filename, 'w') as bash_script:
        print('#!/bin/bash', file=bash_script)
        print('#', file=bash_script)
        print('#$ -cwd', file=bash_script)
        print('#', file=bash_script)
        print('#$ -pe smp ' + str(nprocs), file=bash_script)
        print('#$ -l s_rt=360:00:00', file=bash_script)
        print('#', file=bash_script)
        print('module load mpi/openmpi3-x86_64', file=bash_script)
        print('export PATH=/usr/local/orca_4_1_1_linux_x86-64:$PATH', file=bash_script)
        print('export NBOEXE=/usr/local/nbo7/bin/nbo7.i4.exe', file=bash_script)
        print('export ORIG=$PWD', file=bash_script)
        print('export SCR=$TMPDIR', file=bash_script)
        print('mkdir -p $SCR', file=bash_script)
        if ca:
            print('cp * $SCR', file=bash_script)
        else:
            print('cp ' + inp_filename + ' $SCR', file=bash_script)
        if gbw:
            print('cp ' + gbw_filename + ' $SCR', file=bash_script)
        print('#', file=bash_script)
        print('cd $SCR', file=bash_script)
        print(('/usr/local/orca_4_1_1_linux_x86-64/orca ' +
              inp_filename + ' > $ORIG/' + inp_filename.replace('.inp', '.out')), file=bash_script)
        if notrashtmp:
            pass
        else:
            print('rm -f *.tmp', file=bash_script)
        if cs:
            print('cp * $ORIG', file=bash_script)
        else:
            print('cp *.xyz *.gbw *.qro *.nbo *.47 $ORIG', file=bash_script)
        print('cd / ', file=bash_script)
        print('rm -Rf $SCR', file=bash_script)
        print('cd $ORIG', file=bash_script)
        print('rm *.sh.*', file=bash_script)

    return sh_filename


def run_qsub(script_filename):

    os.system('qsub ' + script_filename)

    return 0


if __name__ == '__main__':

    args = get_args()
    for filename in args.filenames:
        sh_filename = print_bash_script(filename, args.ca, args.cs, args.notrashtmp, args.np, args.gbw)
        run_qsub(sh_filename)