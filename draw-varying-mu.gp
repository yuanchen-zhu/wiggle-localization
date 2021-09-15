set logscale x
set style data linespoints
set grid ytics mytics xtics mxtics
set key on
set xlabel "$\mu$"
set ylabel "$\sigma_p$"
set xrange [1:16]

#set terminal mp color latex solid
#set output "plot-g-rabbit.mp"

plot 'g-None-md-7-N-100-D-1-deltaRatio-10-epsilon-0.0001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-None-md-7-N-100-D-1-deltaRatio-10-epsilon-0.001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-None-md-7-N-100-D-1-deltaRatio-20-epsilon-0.0001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-None-md-7-N-100-D-1-deltaRatio-20-epsilon-0.001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-None-md-7-N-100-D-2-deltaRatio-10-epsilon-0.0001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-None-md-7-N-100-D-2-deltaRatio-10-epsilon-0.001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-None-md-7-N-100-D-2-deltaRatio-20-epsilon-0.0001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-None-md-7-N-100-D-2-deltaRatio-20-epsilon-0.001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-space.xml?10-2-md-9-N-100-D-1-deltaRatio-10-epsilon-0.0001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-space.xml?10-2-md-9-N-100-D-1-deltaRatio-10-epsilon-0.001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-space.xml?10-2-md-9-N-100-D-1-deltaRatio-20-epsilon-0.0001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-space.xml?10-2-md-9-N-100-D-1-deltaRatio-20-epsilon-0.001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-space.xml?10-2-md-9-N-100-D-3-deltaRatio-10-epsilon-0.0001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-space.xml?10-2-md-9-N-100-D-3-deltaRatio-10-epsilon-0.001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-space.xml?10-2-md-9-N-100-D-3-deltaRatio-20-epsilon-0.0001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1", \
    'g-space.xml?10-2-md-9-N-100-D-3-deltaRatio-20-epsilon-0.001-varying-mu.plot' using 1:2 t "isotropic, 0.1% noise, N=100, D=1"
