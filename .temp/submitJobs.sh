#! /bin/bash
#BSUB -W 6000
#BSUB -n 4 
#BSUB -o ./out.%J 
#BSUB -e ./err.%J
rm err/*
rm out/*
for VAR in "Apache" "BDBJ" "BDBC" "LLVM" "X264" "SQL"
do
  bsub -W 100 -n 4 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J tcsh save_conf $VAR $RANDOM
done

for VAR in "ant" "ivy" "lucene" "jedit" "poi"
do
    bsub -W 100 -n 4 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J tcsh save $VAR $RANDOM
done
