import itertools

from Common.description_loader import DescriptionLoader


def main():
    datasets = ['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016']
    llms = ['ChatGPT', 'Llama', 'Vicuna']

    for dataset, llm in itertools.product(datasets, llms):
        folder = f'./1_Datasets/Data/{dataset}/5_Descriptions/{llm}_Base/'
        desc_loader = DescriptionLoader(folder)
        neg_count = 0
        for i in range(len(desc_loader.coverage_vector)):
            if -1 in desc_loader.coverage_vector[i]:
                neg_count += 1
        print(f'{dataset} {llm}: {neg_count} / {len(desc_loader.coverage_vector)} ({neg_count*100 / len(desc_loader.coverage_vector):0.2f}%)')


if __name__ == '__main__':
    main()