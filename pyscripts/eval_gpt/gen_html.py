import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import argparse
import sys
import os


def get_parent_tile(x, y, current_z=16, parent_z=14):
    assert current_z > parent_z
    # zoom 16 -> zoom 14
    zoom_diff = current_z - parent_z
    # Convert coordinates
    parent_x = x // (2**zoom_diff)
    parent_y = y // (2**zoom_diff)
    return (parent_x, parent_y)


def gen_html(df, name, task):
    image_dir = f"../../data/tiles/tiny/tasks/{task}/smaller_images"
    rows = []

    for _, row in df.iterrows():
        file_name = row["file_name"]
        z, x, y = file_name.split("_")
        image_path = f"{image_dir}/{file_name}.webp"  # Adjust extension if needed

        p_x, p_y = get_parent_tile(int(x), int(y))

        parent_tile = f"14_{p_x}_{p_y}"

        # Convert image to base64
        try:
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                img_tag = f'<img src="data:image/png;base64,{img_base64}" style="height:30vh;">'
        except Exception as e:
            img_tag = f"<p>Image not found or could not be loaded: {e}</p>"

        # Create a table row
        rows.append(f"""
            <tr>
                <td>{img_tag}</td>
                <td>{row["predicted"]}</td>
                <td>{row["expected"]}</td>
                <td>{row["Reasoning"]}</td>
                <td>{file_name}\n{parent_tile}</td>
            </tr>
        """)

    # Create the HTML structure
    html_content = f"""
    <html>
    <head>
        <title>DataFrame with Images</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .scrollable-table {{
                overflow-y: auto;
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="scrollable-table">
            <table>
                <tr>
                    <th>Image</th>
                    <th>Predicted</th>
                    <th>Expected</th>
                    <th>Reasoning</th>
                    <th>File Name</th>
                </tr>
                {"".join(rows)}
            </table>
        </div>
    </body>
    </html>
    """

    # Save the HTML to a file
    with open(f"{task}/{name}.html", "w") as f:
        f.write(html_content)


def main(root):
    if not os.path.isdir(root):
        print(f"Directory {root} does not exist.")
        sys.exit(1)

    for directory in os.listdir(root):
        if os.path.isdir(directory):
            task = directory
            print(task)

            # Iterate over all files in the specified directory
            for filename in os.listdir(directory):
                if filename.endswith(".pkl"):  # Process only .pkl files
                    name = os.path.splitext(filename)[0]
                    print(f"\n{filename}")
                    # Load the DataFrame from the pickle file
                    p = os.path.join(directory, filename)
                    df_loaded = pd.read_pickle(p)

                    gen_html(df_loaded, name, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="visualizer", description="Print GPT results")
    parser.add_argument(
        "--root",
        default=".",
        type=str,
        help="Path to the directory containing PBF files",
    )
    args = parser.parse_args()
    main(args.root)
