#! /bin/tcsh
#BSUB -q day 
#BSUB -n 4 
#BSUB -o ./out.%J 
#BSUB -e ./err.%J
for VAR in ant, camel, forrest, ivy, jedit, pbeans, log4j, synapse, velocity, xalan, xerces 
do
  ./save $VAR
done 
