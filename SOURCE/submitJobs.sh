#! /bin/bash
#BSUB -W 6000
#BSUB -n 4 
#BSUB -o ./out.%J 
#BSUB -e ./err.%J
rm err/*
rm out/*
for VAR in "ant" "camel" "forrest" "ivy" "jedit"  "poi" "lucene" "pbeans" "log4j" "synapse" "velocity" "xalan" "xerces"
do
  bsub -W 6000 -n 4 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J tcsh save "$VAR" $RANDOM
done
