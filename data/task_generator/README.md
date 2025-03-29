
### How to run

```bash
builds/linux_x64/task_generator -i ../tiles/sthlm/unprocessed -o ../tiles/sthlm/tasks
```

#### How to Use the Modular Removal Strategy:

1. **Define a New Removal Strategy**:
    - At the beginning of the `main` function (or any configuration section), define a new `FeatureRemovalStrategy` with your desired criteria.
    - **Example: Removing Speed Limits**
        ```cpp
        FeatureRemovalStrategy speed_limit_removal = {
            [](const unprocessed::Feature& feature) -> bool {
                auto it = feature.tags().find("maxspeed");
                if (it != feature.tags().end()) {
                    return true;
                }
                return false;
            }
        };
        ```

2. **Switching Strategies**:
    - To switch between different removal strategies, simply pass the desired `FeatureRemovalStrategy` instance to the `process_directory` function.
    - **Example**:
        ```cpp
        // To remove traffic signals
        process_directory(input_directory, output_directory, traffic_signal_removal);

        // To remove speed limits
        // process_directory(input_directory, output_directory, speed_limit_removal);
        ```

3. **Combining Multiple Strategies**:
    - If you wish to apply multiple removal criteria simultaneously, you can define a combined strategy.
    - **Example**:
        ```cpp
        FeatureRemovalStrategy combined_removal = {
            [&](const unprocessed::Feature& feature) -> bool {
                // Remove traffic signals
                auto it1 = feature.tags().find("traffic_signals");
                if (it1 != feature.tags().end() && it1->second == "traffic_signals" && feature.geometry().points_size() == 1) {
                    return true;
                }

                // Remove speed limits
                auto it2 = feature.tags().find("maxspeed");
                if (it2 != feature.tags().end()) {
                    return true;
                }

                return false;
            }
        };

        process_directory(input_directory, output_directory, combined_removal);