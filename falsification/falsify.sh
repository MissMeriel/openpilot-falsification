#!/bin/bash
imageset=102
eps=10
for imageset in 104 102 199 314 005 680 597 475 448 390
do
	outfile=dnnf_test_eps"$eps"_imageset"$imageset"_pm10percent-forTable2.log
	touch $outfile
	echo epsilon "$eps"
	for m in {1..3}
	do
		for i in {1..10}
		do
			echo "imageset $imageset epsilon "$eps" property $m attempt $i"
			echo "imageset $imageset epsilon "$eps" property $m attempt $i" >> $outfile
			
			dnnf properties/supercombo-property0"$m".py --network N ../models/supercombo.onnx --n_starts 100 --save-violation  counterexamples/counterexample_imageset0"$imageset"_prop"$m"_"$i"_test_eps"$eps"-perc10.npy --prop.image dataset_imgs/imageset0"$imageset".npy --prop.epsilon "$eps" >> $outfile
			echo counterexamples/counterexample_imageset0"$imageset"_prop"$m"_"$i"_test_eps"$eps".npy
		done
	done
done
