#!/bin/bash
source activate cdms2up
f2py --fcompiler=gfortran --f90flags="-fopenmp" -lgomp -c -m ctp cluster_toolkit_parallel.f90 only: clus_sig_p clus_sig_p_ncl
cp ctp.so ../../../ClimTools/
