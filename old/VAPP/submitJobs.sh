#! /bin/bash
#BSUB -W 6000
#BSUB -n 4 
#BSUB -o ./out.%J 
#BSUB -e ./err.%J
rm err/*
rm out/*
for VAR in 'Apache' 'SQL' 'BDBC' 'BDBJ' 'X264' 'LLVM'
  do
  for doWhat in 'Accuracy' 'AUC' 'Median'
    do
    bsub -W 6000 -n 4 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J tcsh save "$VAR" "$doWhat" $RANDOM
  done

done
