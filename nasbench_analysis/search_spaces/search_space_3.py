import matplotlib.pyplot as plt

from nasbench_analysis.search_spaces.search_space import SearchSpace
from nasbench_analysis.utils import NasbenchWrapper


def analysis():
    search_space_3 = SearchSpace(num_parents_per_node={'0': 0, '1': 1, '2': 1, '3': 1, '4': 2, '5': 2, '6': 2},
                                 search_space_number=3, num_intermediate_nodes=5)

    # Load NASBench
    nasbench = NasbenchWrapper('/home/siemsj/projects/darts_weight_sharing_analysis/nasbench_full.tfrecord')

    test_error = []
    valid_error = []

    search_space_creator = search_space_3.create_search_space(with_loose_ends=False, upscale=False)
    for adjacency_matrix, ops, model_spec in search_space_creator:
        # Query NASBench
        data = nasbench.query(model_spec)
        for item in data:
            test_error.append(1 - item['test_accuracy'])
            valid_error.append(1 - item['validation_accuracy'])

    print('Number of architectures', len(test_error) / len(data))

    plt.figure()
    plt.title(
        'Distribution of test error in search space (no. architectures {})'.format(int(len(test_error) / len(data))))
    plt.hist(test_error, bins=800)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('Test error')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, 0.3)
    plt.savefig('nasbench_analysis/search_spaces/export/search_space_3/test_error_distribution.pdf', dpi=600)
    plt.show()

    plt.figure()
    plt.title('Distribution of validation error in search space (no. architectures {})'.format(
        int(len(valid_error) / len(data))))
    plt.hist(valid_error, bins=800)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('Validation error')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, 0.3)
    plt.savefig('nasbench_analysis/search_spaces/export/search_space_3/valid_error_distribution.pdf', dpi=600)
    plt.show()

    print('min test_error', min(test_error), 'min valid_error', min(valid_error))


if __name__ == '__main__':
    analysis()
