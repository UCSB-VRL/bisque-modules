"""
 BQMOD CONFIGURATION:

 bqmod set -n "NPHSegmentationMetric"
 bqmod set -a "VB"
 bqmod set -d "Performs metric computation on (correctly) segmented brain scan files."
 bqmod inputs --image -n "Input Scans"
 bqmod outputs --image -n "Output Metric"
 bqmod summary
"""

import csv
import pathlib
import os
import metric
import sys
# import logging

def run_module(input_path_dict, output_folder_path):
    scans_path = pathlib.Path(input_path_dict['Input Scans'])
    result_path = output_folder_path / pathlib.Path("results.csv")
    exists = result_path.exists()


    with open(result_path, mode='a+') as result_file:
        # result_writer = csv.writer(result_file, delimiter=',')
        result = metric.compute_metric(scans_path)

        result_writer = csv.DictWriter(result_file, fieldnames=result.keys())

        if not exists: result_writer.writeheader()

        result_writer.writerow(result) # column names

    output_paths_dict = {'Output Metric': str(result_path)}

    return output_paths_dict

def main():
    input_path_dict = {'Input Scans': sys.argv[1]}
    output_folder_path = pathlib.Path(sys.argv[2])

    output_paths_dict = run_module(input_path_dict, output_folder_path)

    print(output_paths_dict)

if __name__ == "__main__":
    main()
