import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import List
from collections import defaultdict
import networkx as nx
from typing import List, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")
import hdbscan
import umap.umap_ as umap

from .utils import preprocess_sentences, build_graph, get_n_words


class ReportsGenerator:
    def __init__(
        self,
        summarization_model_name: str = "sshleifer/distilbart-cnn-12-6",
        sentence_embedding_model_name: str = "sentence-transformers/all-distilroberta-v1",
    ):

        self.summarization_model = pipeline(
            "summarization",
            model=summarization_model_name,
            tokenizer=summarization_model_name,
            # device=self.device,
        )
        self.sentence_transformer = SentenceTransformer(
            sentence_embedding_model_name,  # device=self.device
        )

    def _get_sentences_embeddings(self, original_sentences):
        """
        get all tweets embeddings, one embedding per sentence
        """
        cleaned_text = preprocess_sentences(original_sentences)
        return self.sentence_transformer.encode(cleaned_text)

    def _get_hdbscan_clusters(self, embeddings):
        """
        1 - Get embeddings of tweets
        2 - Data reduction algorithm: UMAP if we have too many sentences to cluster
        3 - HDBscan clustering
        """
        n_rows = embeddings.shape[0]
        if n_rows <= 10:
            return np.ones(n_rows)
        else:
            # if too many sentence, apply dimentianality reduction to embedings
            if n_rows > 100:
                reduced_embeddings = umap.UMAP(
                    n_neighbors=7, n_components=15, metric="cosine"
                ).fit_transform(embeddings)
            else:
                reduced_embeddings = embeddings

            # Hdbscan
            cluster = hdbscan.HDBSCAN(
                min_cluster_size=10, metric="euclidean", cluster_selection_method="eom"
            ).fit(reduced_embeddings)

            return cluster.labels_

    def _summarize_one_cluster(
        self,
        original_sentences_one_cluster: List[str],
        embeddings_one_cluster,
    ) -> str:
        """
        Get summary for each cluster
        1 - Compute cosine similarity
        2 - Build undirected graph based on similarity matrix between excerpts
        3 - Get top n reelvant sentences using the pagerank algorithm
        4 - Generate summary of kept sentences
        """
        if type(embeddings_one_cluster) is list:
            embeddings_one_cluster = np.stack(embeddings_one_cluster)

        cosine_similarity_matrix = cosine_similarity(
            embeddings_one_cluster, embeddings_one_cluster
        )

        # get pagerank score for each sentence
        try:
            graph = build_graph(cosine_similarity_matrix)
            pageranks = nx.pagerank(graph)
            scores = np.array(list(pageranks.values()))

        except Exception:  # no ranking if pagerank algorithm doesn't converge
            scores = np.ones(len(embeddings_one_cluster))

        # keep 15 sentences with highest scores
        n_max_kept_sentences = 15
        top_n_sentence_ids = np.argsort(scores)[::-1][:n_max_kept_sentences]
        ranked_sentences = " ".join(
            [original_sentences_one_cluster[id_tmp] for id_tmp in (top_n_sentence_ids)]
        )

        # summarize selected sentences
        summarized_entries = self.summarization_model(
            ranked_sentences, min_length=5, max_length=128, truncation=True
        )[0]["summary_text"]

        return summarized_entries

    def _multiclusters_summarization(
        self,
        entries_as_sentences: List[str],
        entries_embeddings,
        cluster_labels: List[int],
    ) -> str:
        dict_grouped_excerpts = {
            cluster_id: defaultdict(list) for cluster_id in list(set(cluster_labels))
        }
        n_sentences = len(entries_as_sentences)

        # Group sentences, embeddings into respective clusters.
        for i in range(n_sentences):
            cluster_i_label = cluster_labels[i]
            dict_grouped_excerpts[cluster_i_label]["sentences"].append(
                entries_as_sentences[i]
            )
            dict_grouped_excerpts[cluster_i_label]["embeddings"].append(
                entries_embeddings[i]
            )

        # summarize each cluster.
        summarized_entries_per_cluster = []
        for one_cluster_specifics in dict_grouped_excerpts.values():
            n_sentences_one_cluster = len(one_cluster_specifics["sentences"])
            if n_sentences_one_cluster > 10:
                summarized_entries_per_cluster.append(
                    self._summarize_one_cluster(
                        one_cluster_specifics["sentences"],
                        one_cluster_specifics["embeddings"],
                    )
                )

        return " ".join(summarized_entries_per_cluster)

    def _summarization_iteration(self, entries_as_str: str) -> str:

        # Get embeddings
        entries_as_sentences = sent_tokenize(entries_as_str)
        entries_embeddings = self._get_sentences_embeddings(entries_as_sentences)

        # Get clusters
        cluster_labels = self._get_hdbscan_clusters(entries_embeddings)
        n_clusters = len(list(set(cluster_labels)))

        if n_clusters == 1:
            summarized_entries = self._summarize_one_cluster(
                entries_as_sentences, entries_embeddings
            )

        else:
            summarized_entries = self._multiclusters_summarization(
                entries_as_sentences=entries_as_sentences,
                entries_embeddings=entries_embeddings,
                cluster_labels=cluster_labels,
            )

        return summarized_entries

    def __call__(
        self,
        entries: Union[str, List[str]],
        max_iterations: int = 3,
        max_summary_length: int = 256,
    ) -> str:
        """
        Args:
            - entries: text to be summarized, either as a form of a list of sentences or paragraph.
            - max_iterations: int: maximum number of iterations to be performed while summarizing
            - max_summary_length: maximum length of the summary
        """

        assert (
            max_iterations is not int
        ), "'max_iterations' parameter must be an integer."
        assert max_iterations >= 1, "'max_iterations' parameter must >= 1."

        if type(entries) is list:
            entries_as_str = " ".join([str(one_entry) for one_entry in entries])
        elif type(entries) is str:
            entries_as_str = entries
        else:
            AssertionError(
                "argument 'entries' must be one of the types [str, List[str]]"
            )

        if get_n_words(entries_as_str) < max_summary_length:
            warnings.warn(
                "Warning... The length of the original text is smaller than the maximum summary length, setting 'max_iterations' parameter to 1."
            )
            max_iterations = 1

        summarized_text = self._summarization_iteration(entries_as_str)
        n_iterations = 1

        while (
            get_n_words(summarized_text) > max_summary_length
            and n_iterations < max_iterations
        ):
            summarized_text = self._summarization_iteration(summarized_text)
            n_iterations += 1

        if (
            n_iterations == max_iterations
            and get_n_words(summarized_text) > max_summary_length
        ):
            warnings.warn(
                "Warning... Maximum number of iterations reached but summarized text length is still longer than the max_length, returning the long summarized version."
            )

        return summarized_text
