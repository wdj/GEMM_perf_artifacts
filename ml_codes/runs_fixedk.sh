#!/bin/bash
#SBATCH -A stf006
#SBATCH -N 1
#SBATCH -t 48:00:00

# NOTE: these runs are done on andes.olcf.ornl.gov

# (cd logs; env method=dtree  sbatch -J dtree1  ../runs_fixedk.sh)
# (cd logs; env method=etree  sbatch -J etree1  ../runs_fixedk.sh)
# (cd logs; env method=gboost sbatch -J gboost1 ../runs_fixedk.sh)
# (cd logs; env method=ranfor sbatch -J ranfor1 ../runs_fixedk.sh)
# (cd logs; env method=nnet   sbatch -J nnet1   ../runs_fixedk.sh)
# (cd logs; env method=gproc  sbatch -J gproc1  ../runs_fixedk.sh)
# (cd logs; env method=linreg sbatch -J linreg1 ../runs_fixedk.sh)
# (cd logs; env method=pasag  sbatch -J pasag1  ../runs_fixedk.sh)
# (cd logs; env method=ridge  sbatch -J ridge1  ../runs_fixedk.sh)
# (cd logs; env method=sgd    sbatch -J sgd1    ../runs_fixedk.sh)

cd /gpfs/alpine/gen006/proj-shared/$USER/andes
outdir=outs_fixedk
mkdir -p $outdir

module load python

for stub2 in "" "_nop" "_nos" "_nops" ; do

  stub=runs1c-peak$stub2          j=4; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method $j $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs1c-bsd$stub2           j=4; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method $j $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs1c-spock4.2$stub2      j=8; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method $j $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs1c-iris$stub2          j=8; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method $j $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt

done

