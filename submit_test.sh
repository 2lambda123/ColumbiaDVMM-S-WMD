#!/bin/sh
#submit_test.sh
#Torque script to run Matlab program

#Torque directives
#PBS -N swmd
#PBS -W group_list=yetidvmm
#PBS -l nodes=1,walltime=01:30:00,mem=16000mb
#PBS -M xz2437@columbia.edu
#PBS -m abe
#PBS -V

#set output and error directories (SSCC example here)
#PBS -o localhost:/vega/dvmm/users/xz2437/log/
#PBS -e localhost:/vega/dvmm/users/xz2437/log/

#Command to execute Matlab code
matlab -nosplash -nodisplay -nodesktop -r "swmd_distance" > ./log/matoutfile

#Command below is to execute Matlab code for Job Array (Example 4) so that each part writes own output
#matlab -nosplash -nodisplay -nodesktop -r "simPoissGLM($LAMBDA)" > matoutfile.$PBS_ARRAYID

#End of script
