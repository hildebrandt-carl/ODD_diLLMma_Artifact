OPENPILOT_CONTROL_RATE = 5
VIDEO_FPS = 15


ODD = {"Poor Visibility": "Does this image have poor visibility (heavy rain, snow, fog, etc.) or weather conditions that may interfere with sensor operation?",
       "Image Obstructed": "Was the camera that took this image obstructed including by excessive paint or adhesive products (such as wraps, stickers, rubber coating, etc.), covered or damaged by mud, ice, snow, etc?",
       "Sharp Curve": "Is the road we are driving on a sharp curve?",
       "On-off Ramp": "Is the road we are driving on an on-off ramp?",
       "Intersection": "Is the road we are driving on an intersection?",
       "Restricted Lane": "Does the road in this image have restricted lanes?",
       "Construction": "Does the road in this image have construction zones?",
       "Highly Banked": "Is the road we are driving on highly banked?",
       "Bright Light": "Does this image have bright light (due to oncoming headlights, direct sunlight, etc.)?",
       "Narrow Road": "Is the road we are driving on narrow or winding?",
       "Hilly Road": "Is the road we are driving on a hill?"}

ANNOTATOR_NAMING = {"ChatGPT_Base": "ChatGPT-4V",
                    "Vicuna_Base": "Vicuna",
                    "Vicuna_Plus": "Vicuna+",
                    "Human": "Human"}

ANNOTATOR_COLOR = {"ChatGPT_Base": "C0",
                   "Vicuna_Base": "C1",
                   "Vicuna_Plus": "C2",
                   "Human": "C3"}

ANNOTATOR_LINES = {"ChatGPT_Base": "solid",
                   "Vicuna_Base": "solid",
                   "Vicuna_Plus": "solid",
                   "Human": "solid"}

DATASET_NAMING = {"OpenPilot_2016": "comma.ai 2016",
                  "OpenPilot_2k19": "comma.ai 2k19",
                  "External_Jutah": "External JUtah"}

DATASET_SHAPE = {"OpenPilot_2016": "o",
                 "OpenPilot_2k19": "s",
                 "External_Jutah": "^"}

DATASET_COLOR = {"OpenPilot_2016": "C6",
                 "OpenPilot_2k19": "C5",
                 "External_Jutah": "C7"}

DATASET_ORDER = {"OpenPilot_2016": "1",
                 "OpenPilot_2k19": "2",
                 "External_Jutah": "3"}