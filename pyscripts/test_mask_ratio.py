import numpy as np
import torch
from rootutils import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

num_targets = 4
min_context = 0.15
min_context_2 = 0.10


@torch.no_grad()
def main():
    for ratio in np.linspace(0.1, 0.5, 9):
        minc = 999999999
        maxc = 0
        sum = 0
        minc_with_min = 999999999
        maxc_with_min = 0
        sum_with_min = 0
        minc_with_min_2 = 999999999
        maxc_with_min_2 = 0
        sum_with_min_2 = 0
        for i in range(100000):
            tokens = torch.ones((100,), dtype=torch.bool)
            for t in range(num_targets):
                rand = torch.randperm(100, dtype=torch.long)[: int(100 * ratio)]
                tokens[rand] = False
            c_ratio = tokens[tokens].size(0) / 100

            c_ratio_with_min = max(min_context, c_ratio)
            minc_with_min = min(minc_with_min, c_ratio_with_min)
            maxc_with_min = max(maxc_with_min, c_ratio_with_min)
            sum_with_min += c_ratio_with_min

            c_ratio_with_min_2 = max(min_context_2, c_ratio)
            minc_with_min_2 = min(minc_with_min_2, c_ratio_with_min_2)
            maxc_with_min_2 = max(maxc_with_min_2, c_ratio_with_min_2)
            sum_with_min_2 += c_ratio_with_min_2

            minc = min(minc, c_ratio)
            maxc = max(maxc, c_ratio)
            sum += c_ratio
        print(f"\n\nRatio {ratio:.2f}")
        print(f"min {minc}, max {maxc}, avg {sum / 100000:.2f}")
        print(
            f"with min_context {min_context:.2f}, min {minc_with_min}, max {maxc_with_min}, avg {sum_with_min / 100000:.2f}"
        )
        print(
            f"with min_context {min_context_2:.2f}, min {minc_with_min_2}, max {maxc_with_min_2}, avg {sum_with_min_2 / 100000:.2f}"
        )


if __name__ == "__main__":
    main()
