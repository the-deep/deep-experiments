from sklearn.neighbors import NearestNeighbors
import random
import numpy as np


class MLSOL:
    def __init__(self, perc_gen_instances=0.3, k=5, random_seed: int = 0):
        random.seed(random_seed)
        self.perc_gen_instances = perc_gen_instances
        self.k = k

    def get_Cij(self, sample_label, neighbours_label_values):
        num_common_labels = np.count_nonzero(neighbours_label_values == sample_label)
        k = len(neighbours_label_values)
        return (k - num_common_labels) / k

    def get_C(self, y, indices):
        C = np.zeros(y.shape)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                C[i, j] = self.get_Cij(y[i, j], y[indices[i, 1:], j])
        return C

    def get_weight_per_example(self, y, C, min_class_per_label):
        w = np.zeros(y.shape[0])
        sum_of_non_out_minority_examples_per_example = np.zeros(y.shape[1])
        for j in range(y.shape[1]):
            for i in range(y.shape[0]):
                if y[i, j] == min_class_per_label[j] and C[i, j] < 1:
                    sum_of_non_out_minority_examples_per_example[j] += C[i, j]

        for i in range(y.shape[0]):
            sum = 0
            for j in range(y.shape[1]):
                if y[i, j] == min_class_per_label[j] and C[i, j] < 1:
                    sum += C[i, j] / sum_of_non_out_minority_examples_per_example[j]
            w[i] = sum

        return w

    def get_min_class_per_label(self, y):
        # this needs to be improved to handle different class values, not just ones and zeros
        min_per_label = np.zeros(y.shape[1])
        for i in range(y.shape[1]):
            class_1 = np.count_nonzero(y[:, i] == 1)
            class_0 = np.count_nonzero(y[:, i] == 0)
            if class_1 >= class_0:
                min_per_label[i] = 0
            else:
                min_per_label[i] = 1
        return min_per_label

    def get_type_matrix(self, y, C, min_class_per_label, neighbour_indices):
        T = np.empty(C.shape, dtype="U2")
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if y[i, j] == min_class_per_label[j]:
                    if C[i, j] < 0.3:
                        T[i, j] = "SA"
                    elif C[i, j] < 0.7:
                        T[i, j] = "BD"
                    elif C[i, j] < 1:
                        T[i, j] = "RR"
                    else:
                        T[i, j] = "BD"
                else:
                    T[i, j] = "MJ"

        has_changed = True
        while has_changed:
            has_changed = False
            for i in range(C.shape[0]):
                for j in range(C.shape[1]):
                    if T[i, j] == "RR":
                        for m in neighbour_indices[i, 1:]:
                            if T[m, j] == "SF" or T[m, j] == "BD":
                                T[i, j] = "BD"
                                has_changed = True
                                break

        return T

    def get_seed_instance(self, w):
        seed_index = 0
        limit = random.random() * sum(w)
        temp_sum = 0
        for i in range(len(w)):
            temp_sum += w[i]
            if limit <= temp_sum:
                seed_index = i
                break
        return seed_index

    def generate_instance(self, x_seed, y_seed, T_seed, x_reference, y_reference, T_reference):
        x_synthetic = np.zeros(len(x_seed))
        y_synthetic = np.zeros(len(y_seed))
        for i in range(len(x_seed)):
            x_synthetic[i] = x_seed[i] + (random.random() * (x_reference[i] - x_seed[i]))
        dist_seed = np.linalg.norm(x_synthetic - x_seed)
        dist_reference = np.linalg.norm(x_synthetic - x_reference)
        cd = dist_seed / (dist_seed - dist_reference)
        theta = 0
        for j in range(len(y_seed)):
            if y_seed[j] == y_reference[j]:
                y_synthetic[j] = y_seed[j]
            else:
                if T_seed[j] == "MJ":
                    # swap the two indices seed and reference
                    x_seed, x_reference, y_seed, y_reference, T_seed, T_reference = (
                        x_reference,
                        x_seed,
                        y_reference,
                        y_seed,
                        T_reference,
                        T_seed,
                    )
                    cd = 1 - cd

                if T_seed[j] == "SF":
                    theta = 0.5
                elif T_seed[j] == "BD":
                    theta = 0.75
                elif T_seed[j] == "RR":
                    theta = 1.00005
                elif T_seed[j] == "OT":
                    theta = -0.00005

                if cd <= theta:
                    y_synthetic[j] = y_seed[j]
                else:
                    y_synthetic[j] = y_reference[j]

        return x_synthetic, y_synthetic

    def fit_resample(self, X, y):

        # find every label's minority class
        min_class_per_label = self.get_min_class_per_label(y)

        # number of instances to generate
        gen_num = int(X.shape[0] * self.perc_gen_instances)

        X_synthesized, y_synthesized = np.zeros((gen_num, X.shape[1])), np.zeros(
            (gen_num, y.shape[1])
        )

        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm="brute", metric="euclidean").fit(
            X
        )
        distances, indices = nbrs.kneighbors(X)

        C = self.get_C(y, indices)
        w = self.get_weight_per_example(y, C, min_class_per_label)
        T = self.get_type_matrix(y, C, min_class_per_label, indices)
        counter = 0

        for i in range(gen_num):
            # select a seed instance from your initial dataset based on the weights vector w
            seed_index = self.get_seed_instance(w)
            reference_index = indices[seed_index][random.randint(1, self.k)]
            # i am adding the synthesized samples in reverse inside the dataset
            X_synthesized[gen_num - 1], y_synthesized[gen_num - 1] = self.generate_instance(
                X[seed_index, :],
                y[seed_index, :],
                T[seed_index, :],
                X[reference_index, :],
                y[reference_index, :],
                T[reference_index, :],
            )
            # print('sample '+str(counter)+' / '+str(init_size)+' created')
            gen_num -= 1
            counter += 1

        X_aug = np.concatenate((X, np.flip(X_synthesized, axis=0)), 0)
        y_aug = np.concatenate((y, np.flip(y_synthesized, axis=0)), 0)

        return X_aug, y_aug
