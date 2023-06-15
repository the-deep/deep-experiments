def _apply_explainability(
    test_df: pd.DataFrame,
    logged_models: Dict[str, torch.nn.Module],
    output_data_dir: str,
):
    start_explainer = time.process_time()

    if test_df.shape[0] < 200:
        explainability_df = test_df.copy()
    else:
        explainability_df = pd.concat([train_val_df, test_df])[
            ["entry_id", "lang", "excerpt", "target"]
        ].copy()
    explainability_df = explainability_df[explainability_df.lang == "en"]
    n_explainability_entries = explainability_df.shape[0]

    total_explained_labels = 0

    cls_explainer = MultiLabelClassificationExplainer(
        logged_models["backbone"]  # .to(torch.device("cpu")),
    )

    interpretability_results = defaultdict(list)

    for i in range(n_explainability_entries):
        # each 200 sentences, log to mlflow the the sentence number and save the predictions
        if i % 200 == 0:
            mlflow.log_metric("zz_iter_number", i)
            with open(
                Path(output_data_dir) / "explainability_results.pickle",
                "wb",
            ) as f:
                dill.dump(interpretability_results, f)

        row_i = explainability_df.iloc[i]
        one_sentence = row_i["excerpt"]
        one_entry_id = row_i["entry_id"]
        groundtruth_one_row = custom_eval(row_i["target"])
        groundtruth_one_row = [
            item
            for item in groundtruth_one_row
            if "first_level_tags->pillars" not in item
        ]

        if len(groundtruth_one_row) > 0:
            attributions_one_entry = cls_explainer(one_sentence, groundtruth_one_row)
            total_explained_labels += len(groundtruth_one_row)
            for label_name, sentence in attributions_one_entry.items():
                interpretability_results[label_name].append(
                    {"entry_id": one_entry_id, "sentence": sentence}
                )

    end_explainer = time.process_time()
    # save time taken
    time_for_interpretability_per_sentence = np.round(
        (end_explainer - start_explainer) / test_df.shape[0], 2
    )

    mlflow.log_metric(
        "z_explainability_time_per_sentence",
        time_for_interpretability_per_sentence,
    )

    time_for_interpretability_per_label = np.round(
        (end_explainer - start_explainer) / total_explained_labels, 2
    )

    mlflow.log_metric(
        "z_explainability_time_per_label", time_for_interpretability_per_label
    )


def _get_predictions_unlabled(
    train_val_data_labeled: pd.DataFrame,
    train_val_data_non_labeled: pd.DataFrame,
    test_data_labeled: pd.DataFrame,
    relabling_min_ratio: float,
    tags_with_same_projects: List[str],
):
    estimated_prop_false_negatives = _proportion_false_negatives(
        train_val_data_labeled, train_val_data_non_labeled
    )

    mlflow.log_params(
        {
            f"_prop_false_negatives_{_clean_str_for_logging(one_tag)}": round(
                estimated_prop_false_negatives, 3
            )
            for one_tag in tags_with_same_projects
        }
    )

    if estimated_prop_false_negatives > relabling_min_ratio:
        # classification model name
        classification_transformer_name = (
            f"model_{_clean_str_for_logging('_'.join(tags_with_same_projects))}"
        )

        transformer_model, test_set_results = train_test(
            train_val_data_labeled,
            test_data_labeled,
            classification_transformer_name,
        )
        # predictions on unlabeled train val df
        final_predictions_unlabled_train_val = (
            transformer_model.generate_test_predictions(
                train_val_data_non_labeled.excerpt, apply_postprocessing=False
            )
        )

        return final_predictions_unlabled_train_val, test_set_results

    else:
        n_non_labeled = train_val_data_non_labeled.shape[0]
        final_predictions_unlabled_train_val = [[] for _ in range(n_non_labeled)]
        test_set_results = {}
        relabeled = False

    mlflow.log_params(
        {
            f"relabled_{_clean_str_for_logging(one_tag)}": relabeled
            for one_tag in tags_with_same_projects
            if relabeled
        }
    )

    return final_predictions_unlabled_train_val, relabeled, test_set_results


"""###### non sector groups
        for tags_with_same_projects in non_sector_groups:

            projects_list_one_same_tags_set = projects_list_per_tag[
                tags_with_same_projects[0]
            ]

            (
                train_val_data_labeled,
                train_val_data_non_labeled,
                test_data_labeled,
            ) = _get_labled_unlabled_data(
                train_val_df,
                test_df,
                projects_list_one_same_tags_set,
                tags_with_same_projects,
            )

            # classification model name
            classification_transformer_name = (
                f"model_{_clean_str_for_logging('_'.join(tags_with_same_projects))}"
            )

            (
                final_predictions_unlabled_train_val,
                relabeled,
                test_set_results,
            ) = _get_predictions_unlabled(
                train_val_data_labeled,
                train_val_data_non_labeled,
                test_data_labeled,
                args.relabling_min_ratio,
                tags_with_same_projects,
            )
            if relabeled:
                well_labeled_examples.extend(tags_with_same_projects)
                final_results.update(test_set_results)

            # update with labels
            train_val_final_labels = _update_final_labels_dict(
                train_val_data_labeled,
                final_predictions_unlabled_train_val,
                train_val_data_non_labeled,
                train_val_final_labels,
            )
            
        final_labels_df = pd.DataFrame(
            list(
                zip(
                    list(train_val_final_labels.keys()),
                    list(train_val_final_labels.values()),
                )
            ),
            columns=["entry_id", "target"],
        )

        # save predictions df
        final_labels_df.to_csv(
            Path(args.output_data_dir) / "final_labels_df.csv", index=None
        )

        train_val_df = pd.merge(
            left=train_val_df.drop(columns=["target"]),
            right=final_labels_df,
            on="entry_id",
        )
"""
