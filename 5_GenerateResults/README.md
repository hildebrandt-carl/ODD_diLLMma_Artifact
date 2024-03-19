```bash
python3 RQ1_Human_Inspection.py --dataset_directory "../1_Datasets/Data"
```

```bash
python3 RQ1_Inspect_Images_In_ODD.py --annotator Human --description_filter Both --dataset External_Jutah --resize_display "(640,480)" --filter_human_verified_odd --dataset_directory "../1_Datasets/Data"
```

```bash
python3 RQ2_ODD_Dimension_Comparison.py --description_filter Both --dataset_directory "../1_Datasets/Data"
```

```bash
python3 RQ2_ODD_Vector_Comparison.py --description_filter Both --dataset_directory "../1_Datasets/Data"
```

```bash
python3 S2_Dimension_Compliance.py --annotator Human --description_filter Both --dataset_directory "../1_Datasets/Data"
```

```bash
python3 S2_Pass_Fail_Compliance.py --annotator Human --dataset_directory "../1_Datasets/Data"
```


```bash
python3 Compute_Total_Video_Time.py --dataset_directory "../1_Datasets/Data"
```

```bash
python3 Failure_Count.py --annotator Vicuna_Plus --description_filter Both --dataset_directory "../1_Datasets/Data"
```