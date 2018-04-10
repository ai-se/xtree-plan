#  automobile_input.txt
#
set term png
set output "ant.png"
#
#  Fields in each record are separated by commas.
#
set datafile separator ","

set title "Ant"
set xlabel "% Overlap with XTREE's recommendations"
set ylabel "Bugs Reduced"
set grid
plot 'ant/ant_1.csv' using 1:2
quit