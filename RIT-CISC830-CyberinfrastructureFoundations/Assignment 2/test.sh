correct_cnt=0

for i in {1..2}
do
	printf "\nworking on case ${i}:\n"
	test_data=sample${i}.in
	correct_file=sample${i}.out
	timeout 120s bash compile.sh
	time timeout 60s taskset -c 1-8 bash run.sh ${test_data} output_file
	a=`diff -y --suppress-common-lines output_file ${correct_file} | wc -l`
	b=`cat ${correct_file} | wc -l`
	if [ $(($a * 2)) -le $b ]
		correct_cnt=$((correct_cnt+1))
	else
		echo "incorrect on case ${i}"
	fi
done
printf "correct: ${correct_cnt}\n"
