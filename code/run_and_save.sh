#!/bin/bash
   /home/xuzhang/MATLAB/bin/matlab -c /home/xuzhang/tool/Mathworks_Matlab_R2015a_Linux/fixr2015arel/Standalone.lic -nodisplay -nojvm -nosplash -nodesktop -r "try, run('swmd_distance_python.m'), catch, exit(1), end, exit(0);"
