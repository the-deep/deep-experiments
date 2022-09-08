import hdbscan
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


class HDBSCAN_MODEL:
    def __init__(self, embeddings):
        self.hdbscan_model = hdbscan.HDBSCAN(
            gen_min_span_tree=True
        ).fit(embeddings)

        self.params_dist = {
            "min_cluster_size": [
                10, 15, 20, 25, 30, 35, 40, 45, 50,
                55, 60, 65, 70, 75, 80, 100, 150, 200
            ],
            "min_samples": [1, 5, 10, 15, 20, 30],
            "cluster_selection_method": ["eom", "leaf"],
            "metric": ["euclidean", "manhattan"]
        }

        self.embeddings = embeddings
        self.iter_search = 30
        self.seed = 1234

    def calc_score(self):
        score = make_scorer(
            hdbscan.validity.validity_index,
            greater_is_better=True
        )

        self.random_search = RandomizedSearchCV(
            self.hdbscan_model,
            param_distributions=self.params_dist,
            n_iter=self.iter_search,
            scoring=score,
            random_state=self.seed
        )
        self.random_search.fit(self.embeddings)
        # Return dbcv score
        return self.random_search.best_estimator_.relative_validity_

    def get_best_params(self):
        return self.random_search.best_params_
