# Processing Dataset

This code is used to process the dataset.

## Processing Original Data

### OpenPilot_2k19

To process this data you can use the `OpenPilot_2k19/Process_OpenPilot_2k19.sh` script. This script does the following:
* Unzips all the `Chunk_x.zip` files in `0_OriginalData`
* Finds all the `*.hevc` video files
* Moves all `*.hevc` videos files into `1_ProcessData` and renames them to have unique names
* Deletes all uncompressed folders

To run the script you can use
```bash
./Process_OpenPilot_2k19.sh
```

### External_jutah

To process this data you can use the `External_jutah/Process_OpenPilot_utah.sh` script. This script does the following:
* Reads each of the video files
* Compresses them to have the correct framerate and name

To run the script you can use
```bash
./Process_OpenPilot_utah.sh
```

### OpenPilot_2016

To process this data you can use the `OpenPilot_2016/Process_OpenPilot_2016.sh` script. This script does the following:
* Extracts only the imgFormat folder from the zip
* Takes each of the images from that folder and creates a video at 20 FPS which is what the images where generated at
* Slows that video down to the correct framerate and name
* Deletes all the left over files

To run the script you can use
```bash
./Process_OpenPilot_2016.sh
```

## Generating Pass Fail

The first step is to compute the number of passing and failing tests. To do that you can use the `3_identify_pass_fail.py`. You can use this by running the following:

```bash
$ python3 3_identify_pass_fail.py --dataset "OpenPilot_2k19"
>>> Worst error: 1006.9 degrees
>>> Total unknowns: 1356959
>>> Total passing: 464215
>>> Total failing: 3937

$ python3 3_identify_pass_fail.py --dataset "External_jutah"
>>> Worst error: 884.0 degrees
>>> Total unknowns: 1149813
>>> Total passing: 464712
>>> Total failing: 17342

$ python3 3_identify_pass_fail.py --dataset "OpenPilot_2016"
>>> Worst error: 940.1 degrees
>>> Total unknowns: 340069
>>> Total passing: 41263
>>> Total failing: 10514
```

This code is set to look in the `../0_Datasets` for datasets. It will loop through all the `*.h5` files in `2_SteeringData`, and will output a set of text files to `3_PassFail`. These text files will be in the format:

```
Failures: 0
Passing: 466
Unknowns: 5534
-------------------
Frame 0) (s1:-99.0) (s2:-99.0) (e:-99.0): Unknown
Frame 1) (s1:-99.0) (s2:-99.0) (e:-99.0): Unknown
Frame 2) (s1:-99.0) (s2:-99.0) (e:-99.0): Unknown
...
Frame 778) (s1:-4.0) (s2:-3.3) (e:0.8): Pass
Frame 779) (s1:-4.0) (s2:-2.9) (e:1.1): Unknown
Frame 780) (s1:-3.9) (s2:-3.0) (e:0.9): Pass
Frame 781) (s1:-3.6) (s2:-3.0) (e:0.6): Pass
Frame 782) (s1:-3.6) (s2:-2.6) (e:1.0): Pass
```

This format lists the number of failures, passing cases, and unknowns. Unknowns are due to frame mismatches, or when the error is between the pass and fail range.

The default for passing and failures are:
* Pass -> Error less than 1 degree
* Failure -> Error greater than 45 degree

## Selecting Data

The next step is to select X passing and failing input. To do that you can use `4_select_data.py`. You can use this by running the following:
```bash
python3 4_select_data.py --dataset "OpenPilot_2k19"
python3 4_select_data.py --dataset "External_jutah"
python3 4_select_data.py --dataset "OpenPilot_2016"
```

This code is set to look in the `../0_Datasets` for datasets. It will loop through all the `*.txt` files in `3_PassFail`, and will output a set of images to the `4_SelectedData`. These images will be from the videos in the `1_ProcessedData` folder. They will be labeled as either pass or fail.

They are selected by evenly distributing pass and fails from each video.