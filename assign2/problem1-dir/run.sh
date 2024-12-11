perf c2c record -F 22000 -u -- ./reference.out 5 $1
perf c2c report -NN -i perf.data --stdio > ./perf-report_reference.out
echo "Reference Report Generated"
rm perf.data
perf c2c record -F 22000 -u -- ./padded.out 5 $1
perf c2c report -NN -i perf.data --stdio > ./perf-report_padded.out
rm perf.data
echo "Padded Report Generated"
perf c2c record -F 22000 -u -- ./improved.out 5 $1
perf c2c report -NN -i perf.data --stdio > ./perf-report_improved.out
rm perf.data
echo "Improved Report Generated"