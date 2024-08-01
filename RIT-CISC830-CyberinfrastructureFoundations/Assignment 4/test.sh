correct_cnt=0

g++ valid.cc -o valid

for i in {1..10}
do
	printf "\nworking on case ${i}:\n"
	test_data=sample${i}.in
	correct_file=sample${i}.out
	timeout 120s bash compile.sh
	CUDA_VISIBLE_DEVICES=7,8 time timeout 60s taskset -c 1-8 bash run.sh ${test_data} output_file${i}
	./valid ${test_data} output_file${i}
	res=$?
	if [ "${res}" -eq "0" ]; then
		correct_cnt=$((correct_cnt+1))
	else
		echo "incorrect on case ${i}"
	fi
done
printf "correct: ${correct_cnt}\n"
