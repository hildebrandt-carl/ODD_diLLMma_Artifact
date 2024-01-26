# Generate Results
This code processes the caption and produces the results

## RQ1 - Failures within the ODD

The first command you can run is the code used to generate a bar graph comparing the how many tests lie in the ODD, outside the ODD, and were misclassified compared to the baseline.

For example the following code will compare each of the LLM's to the human baseline for each of the baselines

```bash
$ python3 RQ1c_failures_ODD_compare.py --dataset OpenPilot_2016 --size 200 --llm_model "vicuna, llama, chat_gpt, human" --baseline human --show_plot
$ python3 RQ1c_failures_ODD_compare.py --dataset OpenPilot_2k19 --size 200 --llm_model "vicuna, llama, chat_gpt, human" --baseline human --show_plot
$ python3 RQ1c_failures_ODD_compare.py --dataset External_jutah --size 200 --llm_model "vicuna, llama, chat_gpt, human" --baseline human --show_plot
```

This code reads the data from the database and generates the following results. 

![RQ1 base results](../Misc/rq1c.png)

Using these results we computed what is show in the paper. To generate the graphs shown in the paper you can run the following commands:

```bash
$ python3 RQ1a_human_comparison.py
$ python3 RQ1b_time_comparison.py
```

This will result in the following graphs:

![RQ1 paper results](../Misc/rq1a_b.png)

The failure samples can then be found in the folder: `3_GenerateResults/results` where each of the failures are sorted by dataset followed by LLM.
```bash
$ tree .
...
├── External_jutah
│   ├── chat_gpt
│   │   ├── fail_0017.png
│   │   ├── ...
│   │   └── fail_0093.png
│   ├── human
│   │   ├── fail_0001.png
│   │   ├── ...
│   │   └── fail_0098.png
│   ├── llama
│   │   ├── fail_0046.png
│   │   ├── ...
│   │   └── fail_0091.png
│   └── vicuna
│       └── ...
├── OpenPilot_2k19
│   ├── chat_gpt
│   │   └── ...
│   ├── human
│   │   └── ...
│   ├── llama
│   │   └── ...
│   └── vicuna
│   │   └── ...
└── OpenPilot_2016
    ├── chat_gpt
    │   └── ...
    ├── human
    │   └── ...
    ├── llama
    │   └── ...
    └── vicuna
        └── ...
```

This is where the images from the paper were selected:

![RQ1 selected examples](../Misc/rq1_samples.png)



## RQ2 - Compliance and Violation of ODD

The next research question looked at comparing datasets. Specifically we wanted to compare the specific ODD semantics for both the passing portion of input, and the failing input. To view the passing portion you can use the following commands:

```bash
$ python3 RQ2a_dataset_compare.py --dataset OpenPilot_2016 --size 200 --llm_model "vicuna, llama, chat_gpt, human" --pass_fail pass --show_plot
$ python3 RQ2a_dataset_compare.py --dataset OpenPilot_2k19 --size 200 --llm_model "vicuna, llama, chat_gpt, human" --pass_fail pass --show_plot
$ python3 RQ2a_dataset_compare.py --dataset External_jutah --size 200 --llm_model "vicuna, llama, chat_gpt, human" --pass_fail pass --show_plot
```

This will result in the following images:

![RQ2 passing ODD semantics](../Misc/rq2a_pass.png)

To view the failing portion you can use the following command:

```bash
$ python3 RQ2a_dataset_compare.py --dataset OpenPilot_2016 --size 200 --llm_model "vicuna, llama, chat_gpt, human" --pass_fail fail --show_plot
$ python3 RQ2a_dataset_compare.py --dataset OpenPilot_2k19 --size 200 --llm_model "vicuna, llama, chat_gpt, human" --pass_fail fail --show_plot
$ python3 RQ2a_dataset_compare.py --dataset External_jutah --size 200 --llm_model "vicuna, llama, chat_gpt, human" --pass_fail fail --show_plot
```

This will result in the following images:

![RQ2 failing ODD semantics](../Misc/rq2a_fail.png)

We also presented the area overlap in the paper. To get those values you can use the following bash script:

```bash
$ python3 RQ2b_area_calculations.py
```

The results will be presented in the terminal as follows:

```bash
Model: Vicuna agreement with human
	comma.ai 2016: 20.30%
	comma.ai 2k19: 28.90%
	JUtah: 27.40%
	Average: 25.53%
Model: Llama2 agreement with human
	comma.ai 2016: 43.80%
	comma.ai 2k19: 26.30%
	JUtah: 59.30%
	Average: 43.13%
Model: ChatGPT-4V agreement with human
	comma.ai 2016: 83.30%
	comma.ai 2k19: 88.10%
	JUtah: 87.10%
	Average: 86.17%
```

## RQ3 - Grouping inputs by ODD semantics

The final research question looked at how effectively each of the inputs could be grouped based on their ODD semantics into consistent passing and failing clusters. _Running this command is expensive and took around an hour to compute on a 128 core PC._ However you can compute a cheap approximate using fewer `clustering_iterations` which is the number of times it repeats the clustering before determining a trend using the average of all results. Running this example took 5 minutes on an 8 core `i7-1185G7`.

```bash
python3 RQ3_ODD_clustering.py --llm_models "vicuna, llama, chat_gpt, human" --size 200 --max_clusters 100 --clustering_iterations 3 --dataset OpenPilot_2016 --show_plot
```

This will result in the following results for `comma.ai 2016` dataset, which we compare to the results in the paper:

![RQ3 approximate comparison](../Misc/rq3_approximate.png)

To generate all the images in the paper you can use the following command:

```bash
$ python3 RQ3_ODD_clustering.py --llm_models "vicuna, llama, chat_gpt, human" --size 200 --max_clusters 100 --clustering_iterations 25 --dataset OpenPilot_2016 --show_plot
$ python3 RQ3_ODD_clustering.py --llm_models "vicuna, llama, chat_gpt, human" --size 200 --max_clusters 100 --clustering_iterations 25 --dataset OpenPilot_2k19 --show_plot
$ python3 RQ3_ODD_clustering.py --llm_models "vicuna, llama, chat_gpt, human" --size 200 --max_clusters 100 --clustering_iterations 25 --dataset External_jutah --show_plot
```

This will result in the images shown in the paper:

![RQ3 final paper results](../Misc/rq3_paper.png)