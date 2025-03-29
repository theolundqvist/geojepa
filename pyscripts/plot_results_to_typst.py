import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime


def parse_arguments():
    date = datetime.today().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Process MAE scores.")
    parser.add_argument(
        "-i",
        "--filename",
        type=str,
        default=f"results/{date}/best_mae_scores.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="Include only models containing this substring (case-insensitive).",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Exclude models containing this substring (case-insensitive).",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Specify a model to use as reference for percentage comparison.",
    )
    parser.add_argument(
        "--ranks",
        action="store_true",
        help="Show two columns at the end with the number of first and second places.",
    )
    parser.add_argument(
        "--bold",
        action="store_true",
        help="Best model in bold, second best underlined.",
    )
    parser.add_argument(
        "--underline", action="store_true", help="Best model underlined."
    )
    parser.add_argument(
        "--sort", action="store_true", help="Sort models by average rank."
    )
    parser.add_argument(
        "--harmonic", action="store_true", help="Print harmonic mean of ratios."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="huge",
        help="The size of the dataset, huge/small/tiny",
    )
    parser.add_argument(
        "--precision", type=int, default=4, help="Number of decimal places to display."
    )
    args = parser.parse_args()
    args.reference = args.ref
    print("Using file: ", args.filename)
    return args


def read_csv(filename):
    data = []
    with open(filename, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert 'best_mae' to float for comparison
            try:
                row["best_mae"] = float(row["best_mae"])
                row["task"] = row["dataset"]
                row["dataset"] = row["size"]
            except ValueError:
                row["best_mae"] = math.inf  # Assign infinity if conversion fails
            data.append(row)
    return data


def filter_data(data, include=None, exclude=None, size="huge"):
    filtered = []
    for row in data:
        model = row.get("model", "").lower()
        if row.get("size", "") != size:
            continue
        if include:
            include_fail = True
            for inc in include.split(","):
                if inc.lower() in model:
                    include_fail = False
                    break
            if include_fail:
                continue

        if exclude:
            should_exclude = False
            for exc in exclude.split(","):
                if exc.lower() in model:
                    should_exclude = True
                    break
            if should_exclude:
                continue
        filtered.append(row)
    return filtered


def get_unique_values(data, key):
    return sorted({row[key] for row in data})


def find_top_two_mae(data, tasks):
    # Initialize dictionaries to hold the top two MAEs per task
    top_two_mae = {task: [math.inf, math.inf] for task in tasks}
    for row in data:
        task = row.get("task")
        if task not in tasks:
            continue
        mae = row.get("best_mae", math.inf)
        if mae < top_two_mae[task][0]:
            # New best MAE found
            top_two_mae[task][1] = top_two_mae[task][0]
            top_two_mae[task][0] = mae
        elif top_two_mae[task][0] < mae < top_two_mae[task][1]:
            # New second best MAE found
            top_two_mae[task][1] = mae
    return top_two_mae


def build_model_mae_map(data, models, tasks):
    model_mae = {model: {task: None for task in tasks} for model in models}
    for row in data:
        model = row.get("model")
        task = row.get("task")
        mae = row.get("best_mae")
        model_mae[model][task] = mae
    return model_mae


def count_first_second(model_mae_map, top_two_mae, tasks):
    first_counts = defaultdict(int)
    second_counts = defaultdict(int)

    for task in tasks:
        best_mae, second_best_mae = top_two_mae.get(task, [math.inf, math.inf])
        for model, mae_dict in model_mae_map.items():
            mae = mae_dict.get(task)
            if mae is not None:
                if math.isclose(mae, best_mae, rel_tol=1e-9):
                    first_counts[model] += 1
                elif math.isclose(mae, second_best_mae, rel_tol=1e-9):
                    second_counts[model] += 1
    return dict(first_counts), dict(second_counts)


def format_header(args, tasks, include_ranks=False, model_width=25, task_width=15):
    datasetfmt = {
        "building_count": "Buildings",
        "max_speed": "Max Speed",
        "traffic_signals": "Traffic Signals",
        "bridge": "Bridge",
        "car_bridge": "Car Bridge",
    }
    header = (
        f"{'[*Model*],'.ljust(model_width)}"
        + ",".join(f"[*{datasetfmt[task]}*]".center(task_width) for task in tasks)
        + ","
    )
    if args.harmonic:
        header += f"{'[*Harm.*]'}".center(10) + ","

    if include_ranks:
        header += f"{'[_First_]'}".center(10) + f",{'[_Second_]'},".center(10)
    return header


def format_row(
    args,
    model,
    task_mae,
    top_two_mae,
    tasks,
    first_counts,
    second_counts,
    reference_mae=None,
    percent_mode=False,
    underline=False,
    bold=False,
    include_ranks=False,
    harmonic_ratios={},
    task_width=15,
):
    escaped_model = model.replace("*", r"\*")
    row = f"[{escaped_model}]".ljust(25) + ","
    for task in tasks:
        mae = task_mae.get(task)
        best_mae, second_best_mae = top_two_mae.get(task, [math.inf, math.inf])
        if percent_mode and reference_mae:
            ref_mae = reference_mae.get(task)
            if ref_mae is None or ref_mae == math.inf:
                # Reference MAE not available; cannot compute percentage
                cell = "[N/A],"
            elif model == reference_mae.get("model"):
                # Reference model: display actual MAE
                if mae is not None and mae != math.inf:
                    cell = f"[{round(mae, args.precision)}],"
                    if bold and math.isclose(mae, best_mae, rel_tol=1e-9):
                        cell = f"[*{round(mae, args.precision)}*],"
                    elif bold and math.isclose(mae, second_best_mae, rel_tol=1e-9):
                        cell = f"underline[{round(mae, args.precision)}],"
                    if underline and math.isclose(mae, best_mae, rel_tol=1e-9):
                        cell = f"underline[{round(mae, args.precision)}],"
                else:
                    cell = "[],"
            else:
                if mae is not None and ref_mae != 0:
                    percent = ((mae - ref_mae) / ref_mae) * 100
                    if math.isclose(mae, ref_mae, rel_tol=1e-9):
                        cell = "[0.00%],"
                    else:
                        text = ("+" if percent > 0 else "-") + f"{round(percent, 1)}%"
                        color = "rgb(0,150,0)" if percent < 0 else "rgb(150,0,0)"
                        cell = f"text(fill: {color},[{text}]),"
                        if bold and math.isclose(mae, best_mae, rel_tol=1e-9):
                            cell = f"text(fill: {color},[*{text}*]),"
                        elif bold and math.isclose(mae, second_best_mae, rel_tol=1e-9):
                            cell = f"text(fill: {color},underline[{text}]),"
                        if underline and math.isclose(mae, best_mae, rel_tol=1e-9):
                            cell = f"text(fill: {color},underline[{text}]),"
                else:
                    cell = "[N/A],"
        else:
            # Original mode: display actual MAE with formatting
            if mae is not None and mae != math.inf:
                cell = f"[{round(mae, args.precision)}],"
                if bold and math.isclose(mae, best_mae, rel_tol=1e-9):
                    cell = f"[*{round(mae, args.precision)}*],"
                elif bold and math.isclose(mae, second_best_mae, rel_tol=1e-9):
                    cell = f"underline[{round(mae, args.precision)}],"
                if underline and math.isclose(mae, best_mae, rel_tol=1e-9):
                    cell = f"underline[{round(mae, args.precision)}],"
            else:
                cell = "[],"
        row += f"{cell.center(task_width)}"
    if args.harmonic:
        row += f"[{round(harmonic_ratios.get(model, math.inf), args.precision)}],"
    if include_ranks:
        row += (
            f"[{str(first_counts.get(model, ''))}],"
            f"[{str(second_counts.get(model, ''))}],"
        )
    return row




def compute_harmonic_mean_ratios(model_mae_map, models, tasks, reference):
    """
    For each task, find the best (lowest) MAE. Then compute the ratio
    mae(model, task) / best(task) for each model and take the harmonic
    mean of those ratios across tasks.

    Returns a dict: { model_name: harmonic_mean_of_ratios }
    Lower values indicate overall better performance.
    """
    # 1. Find best MAE per task
    ref_maes = {}
    for task in tasks:
        best_mae = math.inf
        for model in models:
            mae = model_mae_map[model].get(task, math.inf)
            if mae is None:
                mae = math.inf
            print(mae, best_mae, model, task)
            if mae < best_mae:
                best_mae = mae
        ref_maes[task] = best_mae

    # if reference:
    #     ref_maes = model_mae_map[reference]

    # 2. Compute ratio = mae / best_mae for each (model, task)
    #    Then compute the harmonic mean across tasks
    hm_ratios = {}
    for model in models:
        reciprocals_sum = 0.0
        valid_count = 0
        for task in tasks:
            mae = model_mae_map[model].get(task, math.inf)
            if mae is None:
                mae = math.inf
            best_mae = ref_maes[task]
            if mae != math.inf and best_mae != math.inf and best_mae != 0:
                ratio = best_mae / mae  # <= 1.0 if "best_mae" truly is the minimum
                reciprocals_sum += 1.0 / ratio
                valid_count += 1

        if valid_count > 0:
            # Harmonic mean of the ratios = number_of_tasks / sum_of(1/ratio_i)
            hm_ratios[model] = valid_count / reciprocals_sum
        else:
            hm_ratios[model] = math.inf  # Or 0 if you prefer

    return hm_ratios


def compute_average_rank(model_mae_map, models, tasks):
    # For each dataset, rank models by MAE (lower is better)
    task_ranks = {task: {} for task in tasks}
    for task in tasks:
        # Get all models' MAE for this dataset
        def orelse(x, otherwise):
            return x if x is not None else otherwise

        scores = [
            (model, orelse(model_mae_map[model].get(task, math.inf), math.inf))
            for model in models
        ]
        # Sort models by MAE ascending
        scores.sort(key=lambda x: x[1])
        # Assign ranks (1 for best, etc.)
        for rank, (model, _) in enumerate(scores, start=1):
            task_ranks[task][model] = rank

    # Compute average rank for each model
    average_ranks = {}
    for model in models:
        ranks = [task_ranks[ds].get(model, len(models) + 1) for ds in tasks]
        average_ranks[model] = sum(ranks) / len(ranks)
    return average_ranks


def main():
    args = parse_arguments()
    data = read_csv(args.filename)
    filtered_data = filter_data(data, args.include, args.exclude, args.dataset)

    tasks = ["building_count", "max_speed", "traffic_signals", "bridge", "car_bridge"]
    models = get_unique_values(filtered_data, "model")

    top_two_mae = find_top_two_mae(filtered_data, tasks)
    model_mae_map = build_model_mae_map(filtered_data, models, tasks)

    # Count first and second place finishes
    first_counts, second_counts = count_first_second(model_mae_map, top_two_mae, tasks)

    percent_mode = False
    reference_model = None
    reference_mae = {}

    if args.reference:
        # Find models matching the percent substring (case-insensitive)
        matching_models = [
            model for model in models if args.reference.lower() in model.lower()
        ]
        if len(matching_models) == 0:
            print(f"Multiple models found matching the substring '{args.reference}':")
            for i, model in enumerate(models):
                print(f"{i} - {model}")
            try:
                i = int(input("Please select the reference model by number: "))
                reference_model = models[i]
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                return
        if len(matching_models) > 1:
            print(f"Multiple models found matching the substring '{args.reference}':")
            for i, model in enumerate(matching_models):
                print(f"{i} - {model}")
            try:
                i = int(input("Please select the reference model by number: "))
                reference_model = matching_models[i]
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                return
        else:
            reference_model = matching_models[0]
        percent_mode = True
        # Extract reference MAE per dataset
        reference_mae = model_mae_map.get(reference_model, {})
        # Add 'model' key to reference_mae for comparison in format_row
        reference_mae["model"] = reference_model

    # Prepare Header
    header = format_header(
        args, tasks, include_ranks=args.ranks, model_width=25, task_width=15
    )

    output_lines = []
    output_lines.append(header)

    harmonic_ratios = compute_harmonic_mean_ratios(
        model_mae_map, models, tasks, reference_model
    )  # compute_average_rank(model_mae_map, models, tasks)
    if args.sort:
        # Compute the mean std values for each model

        models = sorted(
            models,
            key=lambda model: (harmonic_ratios.get(model, math.inf)),
            reverse=True,
        )
    for model in models:
        row = format_row(
            model=model,
            args=args,
            task_mae=model_mae_map.get(model, {}),
            top_two_mae=top_two_mae,
            tasks=tasks,
            first_counts=first_counts,
            second_counts=second_counts,
            reference_mae=reference_mae,
            percent_mode=percent_mode,
            bold=args.bold,
            underline=args.underline,
            include_ranks=args.ranks,
            harmonic_ratios=harmonic_ratios,
            task_width=15,
        )
        output_lines.append(row)

    output_text = "\n\n".join(output_lines)
    print(output_text)
    copy_to_clipboard(output_text)
    print("The output has been copied to the clipboard.")


def copy_to_clipboard(text):
    import pyperclip

    pyperclip.copy(text)


if __name__ == "__main__":
    main()
