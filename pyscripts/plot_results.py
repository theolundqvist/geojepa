import re
from collections import defaultdict
import argparse
import glob
import os
import json
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import yaml  # Import yaml for reading config.yaml files
import numpy as np  # For median and MAE computation
import rootutils
from tqdm import tqdm


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def parse_results_file(args, file_path, cutoff):
    """
    Parse the results.json file and extract relevant data.
    If "cheat" key is missing, it will default to False instead of failing.
    Also reads config.yaml in the same directory to extract additional flags.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

            # Ensure required fields exist and set defaults where necessary
            if "cheat" not in data:
                data["cheat"] = True  # Default to False if "cheat" is missing

            if not all(key in data for key in ["model", "dataset", "date", "time"]):
                print("key error")
                raise ValueError(f"Missing required field in {file_path}")

            # Parse the time as a datetime object for comparison
            date = datetime.strptime(data["date"], "%Y-%m-%d")

            if date < cutoff["hard"]:
                return None

            time = datetime.strptime(data["time"], "%H:%M:%S")
            # Extract learning rate and weight decay values
            lr = data.get("lr", None)
            lr_min = data.get("lr_min", None)
            weight_decay = data.get("weight_decay", None)

            # Read config.yaml
            config_file = os.path.join(os.path.dirname(file_path), "config.yaml")
            # Add "-I", "-T", or "-TI" to the model name based on the flags
            suffix = ""

            # if tokenize_geometry or tokenize_tags or tokenize_images:
            #     suffix = '-'
            # if tokenize_geometry:
            #     suffix += 'G'
            # if tokenize_tags:
            #     suffix += 'T'
            # if tokenize_images:
            #     suffix += 'I'
            # if tokenize_images_finetuned:
            #     suffix += 'F'
            def parse_embedded_name(name):
                # if "sklearn" in name:
                #     data["cheat"] = False
                suffix = ""
                for n in ["vitb16", "scalemae", "resnet", "efficientnet"]:
                    if n in name:
                        data["model"] = n
                        data["cheat"] = False
                for n in ["tag_ae", "tagformer_ae", "tagformer_mae", "tagformer_lmae"]:
                    if n in name:
                        data["model"] = n
                        data["cheat"] = True
                if "tag_avg" in name:
                    data["model"] = "tag_avg"
                    data["cheat"] = "masked" not in name

                pattern = r"geojepa"
                if re.search(pattern, name, re.IGNORECASE):
                    data["model"] = "geojepa_probe"
                    data["cheat"] = "masked" not in name
                    pattern = r"geojepa(_m[0-9]+)*_([gtif]+)"
                    match = re.search(pattern, name, re.IGNORECASE)
                    if match:
                        suf = match.group(2).lower()
                        if "g" in suf:
                            suffix += "G"
                        if "t" in suf:
                            suffix += "T"
                        if "i" in suf:
                            suffix += "I"
                if "pool" in name:
                    data["pool"] = True
                elif "_svr" in name:
                    suffix += "-SVR"
                elif "_svc" in name:
                    suffix += "-SVC"
                elif "_mlpc" in name:
                    suffix += "-MLP"
                elif "_mlpr" in name:
                    suffix += "-MLP"
                elif "_linr" in name:
                    suffix += "-LIN"
                elif "_linc" in name:
                    suffix += "-LIN"
                return suffix

            if os.path.exists(config_file):
                with open(config_file, "r") as cf:
                    config_data = yaml.safe_load(cf)
                    backbone = {}
                    try:
                        backbone = config_data["model"]["backbone"]
                    except:
                        pass
                    try:
                        if (
                            len(backbone) != 0
                            and "EmbeddingLookup" in backbone["_target_"]
                        ):
                            dir_name = backbone["dir"].rstrip("/").split("/")[-2]
                            suffix = parse_embedded_name(dir_name)
                    except Exception:
                        pass
                    if backbone.get("tokenizer", None) is not None:
                        if backbone["tokenizer"]["tokenize_geometry"]:
                            suffix += "G"
                        if backbone["tokenizer"]["tokenize_tags"]:
                            suffix += "T"
                        if backbone["tokenizer"]["tokenize_images"]:
                            suffix += "I"
                    if data["model"] == "tagformer":
                        head = config_data["model"]["head"]
                        if head.get("use_semantic_encoding", False):
                            suffix += "S"
                        if head.get("use_positional_encoding", False):
                            suffix += "P"
            else:
                suffix = parse_embedded_name(data["model"])
            if suffix != "":
                suffix = "-" + suffix

            # Now adjust the model name according to the rules
            model_name = data["model"]

            # Apply the model name mapping
            model_name_mappings = {
                "geojepa_probe": "(M) GeoJEPA",
                "cheat_geojepa_probe": "GeoJEPA",
                "scalemae": "ScaleMAE",
                "resnet": "ResNet",
                "vitb16": "ViT-B/16",
                "efficientnet": "EfficientNet",
                "ag-I-medium-600s": "AG-4M-600s",
                "ag-I-medium-900s": "AG-4M-900s",
                "ag-I-high-600s": "AG-97M-600s",
                "ag-I-high-900s": "AG-97M-900s",
                "ag-I-best-3600s": "AG-197M-3600s",
                "tagformer": "(M) Tagformer",
                "cheat_tagformer": "Tagformer",
                "tagformer_lmae": "(M) TagformerLMAE",
                "cheat_tagformer_lmae": "TagformerLMAE",
                "cheat_tagae": "TagAE",
                "tag_ae": "(M) TagAE",
                "cheat_tag_ae": "TagAE",
                "tag_avg": "(M) TagPool",
                "cheat_tag_avg": "TagPool",
                "tagcountencoder": "(M) TagCountMLP",
                "cheat_tagcountencoder": "TagCountMLP",
                "cheat_gpt-4o": "GPT-4o",
                "cheat_gpt-4o-mini": "GPT-4o-mini",
                "cheat_dummy": "Dummy",
            }

            if "-I" in suffix:
                data["cheat"] = True

            if data.get("pool", False) and suffix != "":
                suffix += "_pool"

            # If cheat is True, prefix "cheat_"
            if data["cheat"]:
                model_name = f"cheat_{model_name}"

            basic_model_name = model_name
            if not args.original_names:
                model_name = model_name_mappings.get(model_name, model_name)
            if args.dir_name and args.dir_name in basic_model_name:
                model_name = Path(file_path).parent.name
                remove = [
                    "_mlpr",
                    "_mlpc",
                    "_linr",
                    "_linc",
                    "_traffic_signals",
                    "_masked",
                    "_car_bridge",
                    "_bridge",
                    "_max_speed",
                    "_building_count",
                ]
                data["cheat"] = "masked" not in model_name
                if "masked" in model_name and not data["cheat"]:
                    model_name = "(M) " + model_name
                for r in remove:
                    model_name = model_name.replace(r, "")
                if "embedding_lookup" in model_name:
                    return None

            model_name += suffix
            if args.date is not None and (
                args.date == "all" or args.date.lower() in model_name.lower()
            ):
                date = datetime.strptime(data["date"], "%Y-%m-%d")
                model_name = date.strftime("%d/%m") + " " + model_name

            def get_sizes():
                sizes = {}
                for size in ["huge", "tiny", "large", "small"]:
                    if size in data:
                        k = size
                        if size == "tiny":
                            if (
                                datetime.strptime(data["date"], "%Y-%m-%d")
                                < cutoff["tiny"]
                            ):
                                k = "tiny_old"
                            else:
                                k = "tiny"
                        mae = data[size].get("test/mae", data[size].get("mae", None))
                        if str(mae) != "nan":
                            sizes[k] = mae
                        else:
                            print(
                                f"\nNaN MAE score for {model_name}:{data['dataset']}:{size} \nin {file_path}"
                            )
                return sizes

            if data["dataset"] == "pretraining":
                return None

            if date > cutoff["soft"]:
                model_name += "*"

            model_name = model_name.replace("--", "-")

            return {
                "model": model_name,
                "dataset": data["dataset"],
                "config": Path(file_path).parent / "hydra/config.yaml",
                "cheat": data["cheat"],
                "training_time": data.get("training_time", None),
                "date": date,
                "time": time,  # Store time as datetime object
                "lr": lr,
                "lr_min": lr_min,
                "weight_decay": weight_decay,
                "data_sizes": get_sizes(),
            }
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in {file_path}: {e}")
    except KeyError as e:
        print(f"Missing key {e} in {file_path}")
    except ValueError as e:
        print(f"Value error in {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error parsing {file_path}: {e}")
    return None


def organize_results_by_dataset(results):
    """
    Organize results by dataset, data size, and model.
    """
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for result in results:
        dataset = result["dataset"]
        model = result["model"]
        date = result["date"]

        for data_size, mae_score in result["data_sizes"].items():
            if mae_score is None:
                continue  # Skip if MAE score is missing

            # Append the MAE score to the list for this model, dataset, and data size
            organized[dataset][data_size][model].append(
                {
                    "date": date,
                    "mae_score": mae_score,
                    "lr": result["lr"],
                    "lr_min": result["lr_min"],
                    "weight_decay": result["weight_decay"],
                    "config": result["config"],
                    "training_time": result["training_time"],
                }
            )

    return organized


def compute_baseline_maes(organized_data, data_dir):
    """
    Compute the baseline MAE for each dataset and data size by predicting the median label.
    """
    baseline_maes = {}
    for dataset in organized_data.keys():
        for data_size in organized_data[dataset].keys():
            labels_file = os.path.join(
                data_dir, data_size, "tasks", dataset, "labels.txt"
            )
            if os.path.exists(labels_file):
                with open(labels_file, "r") as f:
                    labels = [
                        float(line.strip().split(":")[-1]) for line in f if line.strip()
                    ]
                if labels:
                    median_label = np.median(labels)
                    mae = np.mean([abs(label - median_label) for label in labels])
                    print(f"Baseline MAE for {dataset} ({data_size}): {mae}")
                    baseline_maes[(dataset, data_size)] = mae
                else:
                    print(f"No labels found in {labels_file}")
                    baseline_maes[(dataset, data_size)] = None
            else:
                print(f"Labels file not found: {labels_file}")
                baseline_maes[(dataset, data_size)] = None
    return baseline_maes


def plot_results(organized_data, baseline_maes, output_dir):
    """
    Create box plots for each dataset and data size, showing the MAE scores for all models,
    and display a single table per dataset and data size showing model names, dates, and MAE scores.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset, data_sizes in organized_data.items():
        for data_size, models in data_sizes.items():
            labels = []
            mae_scores_list = []

            for model, runs in models.items():
                labels.append(model)
                # Extract MAE scores from all runs for this model
                mae_scores = [float(run["mae_score"]) for run in runs]
                mae_scores_list.append(mae_scores)

            # Sort the data by the median of MAE scores
            sorted_data = sorted(
                zip(labels, mae_scores_list), key=lambda x: np.min(x[1])
            )
            labels, mae_scores_list = zip(*sorted_data)

            # Plotting box plot for MAE scores
            fig, ax = plt.subplots(figsize=(10, 6))
            box = ax.boxplot(mae_scores_list, tick_labels=labels, patch_artist=True)

            # Customize box plot appearance
            colors = ["lightblue"] * len(labels)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)

            ax.set_xlabel("Models")
            ax.set_ylabel("Test MAE")
            ax.set_title(f"Test MAE Box Plot for Dataset: {dataset} ({data_size})")
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            # Apply conditional formatting to x-tick labels
            for tick_label, model, mae_scores in zip(
                ax.get_xticklabels(), labels, mae_scores_list
            ):
                if "JEPA" in model:
                    # if np.median(mae_scores) == min_median_mae:
                    tick_label.set_fontweight("bold")
                # Underline the label if the minimum MAE is below a threshold
                # if np.min(mae_scores) < 0.5:  # Example threshold
                #     tick_label.set_style('italic')  # Matplotlib doesn't support underline directly

            plt.tight_layout()

            # Add baseline MAE as a horizontal line
            baseline_mae = baseline_maes.get((dataset, data_size))
            if baseline_mae is not None:
                ax.axhline(
                    baseline_mae, color="red", linestyle="--", label="Baseline MAE"
                )
                ax.legend()

            # Save box plot with MAE scores for the dataset and data size
            dataset_safe = dataset.replace("/", "_")
            os.makedirs(Path(output_dir) / data_size, exist_ok=True)
            output_file = os.path.join(
                output_dir, f"{data_size}/{dataset_safe}_boxplot.png"
            )
            plt.savefig(output_file)
            print(f"Box plot saved to {output_file} with {len(labels)} models.")
            plt.close()

            # Now, create a single table for the dataset and data size
            fig, ax = plt.subplots(
                figsize=(10, 3)
            )  # Adjust size to fit the table better
            ax.axis("off")  # Hide the axes, only show the table

            # # Create the table
            # table_data = []
            # for i in range(len(labels)):
            #     model = labels[i]
            #     mae_scores = mae_scores_list[i]
            #     dates = [run["date"].strftime('%Y-%m-%d') for run in models[model]]
            #     # Combine dates and MAE scores for display
            #     for date, mae_score in zip(dates, mae_scores):
            #         table_data.append((model, date, mae_score))
            # # Add baseline MAE to the table
            # table_data.append(("Baseline", "-", baseline_mae if baseline_mae is not None else "N/A"))
            #
            # # Create the table
            # table = ax.table(cellText=table_data, colLabels=["Model", "Date", "MAE Score"], loc="center",
            #                  cellLoc="center")
            # table.auto_set_font_size(False)
            # table.set_fontsize(10)
            # table.scale(1, 2)  # Adjust table scale for better readability
            #
            # # Save the table plot
            # table_output_file = os.path.join(output_dir, f"{data_size}/{dataset_safe}_table.png")
            # plt.savefig(table_output_file)
            # print(f"Table plot saved to {table_output_file}.")
            # plt.close()


def save_best_mae_to_csv(organized_data, output_file):
    rows = []
    for dataset, data_sizes in organized_data.items():
        for data_size in data_sizes:
            for model, runs in data_sizes[data_size].items():
                # Find the run with the best (minimum) MAE score
                best_run = min(runs, key=lambda x: float(x["mae_score"]))
                training_time = best_run.get("training_time", None)
                rows.append(
                    {
                        "size": data_size,
                        "dataset": dataset,
                        "model": model,
                        "best_mae": best_run["mae_score"],
                        "training_time": ""
                        if not training_time
                        else f"{training_time // 60:.0f}m {training_time % 60:.0f}s",
                        "lr": best_run["lr"],
                        "lr_min": best_run["lr_min"],
                        "weight_decay": best_run["weight_decay"],
                        "date": best_run["date"].strftime("%Y-%m-%d"),
                        "config": best_run["config"],
                    }
                )

    # Sort rows by dataset and then by best MAE score within each dataset
    rows.sort(key=lambda x: (x["size"], x["dataset"], float(x["best_mae"])))

    # Save to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Best MAE scores for 'huge' data size saved to {output_file}")


def process_files(args, cutoff):
    """
    Process all results.json files and organize results by dataset, data size, and model.
    """
    result_files = glob.glob(
        os.path.join(args.directory, "**", "results.json"), recursive=True
    )
    print(f"Found {len(result_files)} results.json files.")

    all_results = []
    print("Parsing result files...")
    for file in tqdm(result_files):
        result = parse_results_file(args, file, cutoff)
        if result:
            all_results.append(result)

    if not all_results:
        print("No results found.")
        return {}, {}

    # Organize results by dataset, data size, and model
    organized_data = organize_results_by_dataset(all_results)

    # Compute baseline MAEs
    baseline_maes = compute_baseline_maes(organized_data, args.data_dir)

    return organized_data, baseline_maes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse results.json files and plot MAE scores."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="remote-logs",
        help="Root directory to search for results.json files (default: remote-logs)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory for the plots",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tiles",
        help="Root directory of the data (default: data/tiles)",
    )
    parser.add_argument("--tiny-cutoff", type=str, default="2024-12-03")
    parser.add_argument("--hard-cutoff", type=str, default="2024-12-18")
    parser.add_argument("--jepa-cutoff", type=str, default="2024-12-18")
    parser.add_argument(
        "--soft-cutoff", type=str, default=datetime.today().strftime("%Y-%m-%d")
    )
    parser.add_argument(
        "--date",
        type=str,
        dest="date",
        help="Display date in the model name, specify the model to show date for",
    )
    parser.add_argument(
        "--original-names", action="store_true", help="Display original model names"
    )
    parser.add_argument(
        "--dir-name",
        type=str,
        default=None,
        help="Display dir name as model name, if arg is included in model name",
    )

    args = parser.parse_args()
    tiny_cutoff = datetime.strptime(args.tiny_cutoff, "%Y-%m-%d")
    soft_cutoff = datetime.strptime(args.soft_cutoff, "%Y-%m-%d")
    hard_cutoff = datetime.strptime(args.hard_cutoff, "%Y-%m-%d")
    cutoff = {
        "tiny": tiny_cutoff,
        "soft": soft_cutoff,
        "hard": hard_cutoff,
        "jepa": datetime.strptime(args.jepa_cutoff, "%Y-%m-%d"),
    }

    print(f"Searching in directory: {args.directory}")
    organized_data, baseline_maes = process_files(args, cutoff)
    if organized_data:
        output = args.output + "/" + datetime.today().strftime("%Y-%m-%d")
        plot_results(organized_data, baseline_maes, output)
        csv_output = os.path.join(output, "best_mae_scores.csv")
        save_best_mae_to_csv(organized_data, csv_output)
        os.system(f"open -a finder {output}")
    else:
        print("No data found in the specified directory.")
