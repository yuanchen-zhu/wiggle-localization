set terminal mp color solid latex magnification 1.3
set output "varying.mp"

set logscale x
set logscale y
set style data linespoints
set grid ytics mytics xtics mxtics
set key on
set size ratio 0.7

set xlabel "$\\mu$"
#set ylabel "$\\sigma_p$"
set yrange [:]


# plot for varying mu

set xrange [1:40]
set yrange [:1]
load 'varying-mu-p.gp'
#set ylabel "$\\sigma_d$"
load 'varying-mu-d.gp'

# plot for varying N
set xrange [:200]
set yrange [:10]
set xlabel "$N$"
#set ylabel "$\\sigma_p$"
load 'varying-N-p.gp'
#set ylabel "$\\sigma_d$"
load 'varying-N-d.gp'

# plot for varying perturb
set xrange [5:]
set yrange [:1]
set xlabel "$\\frac{\\delta}{\\epsilon R}$"
#set ylabel "$\\sigma_p$"
set xrange [:]
load 'varying-delta-p.gp'
#set ylabel "$\\sigma_d$"
set yrange [:]
load 'varying-delta-d.gp'

# plot for varying D
set xrange [2:10]
set yrange [:1]
set xlabel "$D$"
#set ylabel "$\\sigma_p$"
unset logscale x
set xrange[:]
set yrange[:]
load 'varying-D-p.gp'

#set ylabel "$\\sigma_d$"
load 'varying-D-d.gp'
