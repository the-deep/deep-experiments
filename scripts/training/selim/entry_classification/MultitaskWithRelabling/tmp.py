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
