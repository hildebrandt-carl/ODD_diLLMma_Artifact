import numpy as np


comma_ai = np.array([[ 41.,  70.,   2.,   1.,   0.,   3.,   5.,  42.,   0.,   0.],
                    [ 49.,  66.,  35.,  22.,  48.,  42.,  33.,  42.,  18.,  44.],
                    [ 92.,  93.,  83.,  63.,  63.,  93.,  79.,  90.,  93.,  59.],
                    [ 85.,  98., 100., 100.,  92.,  98.,  99.,  90., 100.,  99.]])

comma_2k19 = np.array([[ 70.,  72.,  10.,  14.,   8.,  14.,   8.,  57.,   1.,   7.],
                    [ 49.,  52.,  20.,   6.,  17.,  24.,  20.,  30.,   5.,  12.],
                    [ 92.,  89.,  97.,  83.,  47.,  93.,  95.,  97., 100.,  72.],
                    [ 98.,  99.,  98.,  99.,  87., 100., 100.,  98.,  94.,  99.]])

utah = np.array([[ 30.,  53.,  16.,  15.,   7.,   8.,   9.,  36.,   1.,  21.],
                    [ 91.,  85.,  75.,  44.,   3.,  55.,  49.,  44.,  27.,  42.],
                    [ 99., 100.,  96.,  96.,  15.,  53.,  68.,  98., 100.,  90.],
                    [100., 100., 100., 100.,  54.,  86.,  95.,  91.,  96., 100.]])

model_names = {
    0: 'Vicuna',
    1: 'Llama2',
    2: 'ChatGPT-4V',
}

dataset_names = {
    0: 'comma.ai 2016',
    1: 'comma.ai 2k19',
    2: 'JUtah',
}

for model in range(3):
    model_name = model_names[model]
    dataset_agreement = []
    print(f'Model: {model_name} agreement with human')
    for dataset_num, dataset in enumerate([comma_ai, comma_2k19, utah]):
        dataset_name = dataset_names[dataset_num]
        human_output = dataset[-1]
        model_output = dataset[model]
        agreement_amounts = []
        for dimension in range(len(comma_ai[model])):
            positive = min(human_output[dimension], model_output[dimension])
            negative = min((100-human_output[dimension], (100-model_output[dimension])))
            agreement = positive + negative
            agreement_amounts.append(agreement)
        avg_agreement = np.mean(agreement_amounts)
        dataset_agreement.append(avg_agreement)
        print(f'\t{dataset_name}: {avg_agreement:0.2f}%')
    print(f'\tAverage: {np.mean(dataset_agreement):0.2f}%')