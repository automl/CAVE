#!/bin/sh

echo "CSV_SEPARATE"
cave examples/csv_separate/* --ta_exec_dir examples/csv_separate --output output/examples-csv_separate_whitetest --verbose WARNING
echo "CSV_ALLINONE"
cave examples/csv_allinone/* --ta_exec_dir examples/csv_allinone --output output/examples-csv_allinone_whitetest --verbose WARNING
echo "SMAC2"
cave examples/smac2/ --ta_exec_dir examples/smac2/smac-output/aclib/state-run1/ --output output/examples-smac2_whitetest --verbose WARNING
echo "SMAC3"
cave examples/smac3/example_output/run* --ta_exec_dir examples/smac3 --output output/examples-smac3_whitetest --verbose WARNING
echo "BOHB"
cave examples/bohb --output output/examples-bohb_whitetest --verbose WARNING
