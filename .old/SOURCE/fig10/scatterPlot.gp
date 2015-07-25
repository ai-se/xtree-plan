#  scatter_plot_input.txt
#
#  Input to GNUPLOT, causing it to read SCATTER_PLOT.TXT and
#  create a "scatter plot" of the data, that is, dots representing
#  each point.
#
#  Choose the output device.
#
# set term png medium
set terminal postscript eps color colortext
#
#  Name the output file.
#
set output "scatter_plot.eps"
#
#  Set the plot title.
#
set title "Ivy"
#
#  Get grid lines.
#
set grid
#
#  Set axis labels.
#
set xlabel " X "
set ylabel " Y "
unset tics
#
# Set xrange and yrange
#
# set xrange [-600:1500]
set yrange [-3:]
#
#  The following command forces X and Y axes to appear the same size.
#
set size ratio 1
#
#  Timestamp the plot.
#
# set timestamp
#
#  Plot the data using X and Y ranges [0,1],
#  using the data in 'scatter_plot.txt',
#  marking the data with points only (a scatter plot)
#  using line type 3 (blue)
#  and point type 4 (open square)
#
set key right bottom Left title 'Legend' box 3

plot 'Testing' with points lt 1 lc rgb '#dd0000', 'Planned' with points lt 1 lc rgb '#00dd00', 'Training' with points lt 1 lc rgb '#777777' 
#
#  Terminate.
#
quit
