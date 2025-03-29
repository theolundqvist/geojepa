#include <fstream>
#include <iostream>
#include <filesystem>
#include <mutex>
#include <atomic>
#include <set>
#include <utility>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <functional>
#include <memory> // For smart pointers

#include "proto/unprocessed_tile_group.pb.h"
#include "proto/processed_tile_group.pb.h"
#include "indicators.hpp"
#include "cxxopts.h"
#include "BS_thread_pool.hpp"

#include "generator.h"

namespace fs = std::filesystem;
using namespace indicators;


// Global mutex for write operations (shared among all strategies)
std::mutex write_mutex;

// Set to keep track of output files and prevent duplicates (shared among all strategies)
std::set<std::string> output_files;

// Main function
int main(int argc, char *argv[]) {
    // Parse command-line arguments
    std::string input_directory;
    std::string output_directory;
    int status = parse_args(input_directory, output_directory, argc, argv);
    if (status != -1) return status;

    // Define multiple feature removal strategies
    std::vector<std::unique_ptr<FeatureHandlingStrategy>> strategies;
    strategies.emplace_back(std::make_unique<FeatureHandlingStrategy>(
            "pretraining",
            [](unprocessed::Feature *feature) -> strategyResult {
                return {false, 1};
            },
            [](const std::vector<float> &labels) -> float {
                return std::accumulate(labels.begin(), labels.end(), 0.0f);
            },
            StrategyConfig{StrategyConfig::TaskType::REGRESSION, 0}
    ));
    strategies.emplace_back(std::make_unique<FeatureHandlingStrategy>(
            "traffic_signals",
            [](unprocessed::Feature *feature) -> strategyResult {
                auto tags = feature->mutable_tags();
                auto is_point = feature->geometry().points_size() == 1;
                bool to_remove{false};
                for (auto it = tags->begin(); it != tags->end();) {
                    const auto &tag = *it;
                    if (utils::match_substring(tag.first, "traffic_signals")
                        || utils::match_substring(tag.second, "traffic_signals")
                        || utils::match_substring(tag.first, "crossing:signals")
                            ) {
                        it = tags->erase(it);
                        if (is_point) to_remove = true;
                    } else {
                        ++it;
                    }
                }
                return {to_remove, to_remove ? 1.0f : 0.0f};
            },
            [](std::vector<float> floats) -> float {
                return std::accumulate(floats.begin(), floats.end(), 0.0f);
            },
            StrategyConfig{StrategyConfig::TaskType::REGRESSION, 0}
    ));
    // Add more strategies here as needed
    strategies.emplace_back(std::make_unique<FeatureHandlingStrategy>(
            "bridge",
            [](unprocessed::Feature *feature) -> strategyResult {
                auto tags = feature->mutable_tags();
                bool to_remove{false};
                float has_bridge{0.0f};
                // if (tags->empty()) printf("Number of points in empty tags: %d\n", feature->geometry().points_size());
                for (auto it = tags->begin(); it != tags->end();) {
                    const auto &tag = *it;
                    if (utils::match_substring(tag.first, "bridge")
                    || utils::match_substring(tag.second, "bridge")) {
                        has_bridge = 1.0f;
                        // printf("Remove tag: %s=%s at %f, %f\n", tag.first.c_str(), tag.second.c_str(), feature->geometry().points()[0].lon(), feature->geometry().points()[0].lat());
                        it = tags->erase(it);
                    } else if ( utils::match_substring(tag.first, "layer")){
                        it = tags->erase(it);
                    } else {
                        ++it;
                    }
                }
                if (tags->empty() && has_bridge > 0.5f) {
                    // printf("REMOVING FEATURE that has bridge=%f AT %f:%f\n", has_bridge, feature->geometry().points()[0].lon(), feature->geometry().points()[0].lat());
                    to_remove = true;
                }
                return {to_remove, has_bridge};
            },
            [](std::vector<float> floats) -> float {
                return std::accumulate(floats.begin(), floats.end(), 0.0f) > 0.5f ? 1.0f : 0.0f;
            },
            StrategyConfig{StrategyConfig::TaskType::REGRESSION, 0}
    ));

    strategies.emplace_back(std::make_unique<FeatureHandlingStrategy>(
            "car_bridge",
            [](unprocessed::Feature *feature) -> strategyResult {
                auto tags = feature->mutable_tags();
                bool to_remove{false};
                float has_bridge{0.0f};
                bool is_car_road{false};
                // if (tags->empty()) printf("Number of points in empty tags: %d\n", feature->geometry().points_size());
                for (auto it = tags->begin(); it != tags->end();) {
                    const auto &tag = *it;
                    if (utils::match_substring(tag.first, "bridge")
                        || utils::match_substring(tag.second, "bridge")
                        ) {
                        has_bridge = 1.0f;
                        // printf("Remove tag: %s=%s at %f, %f\n", tag.first.c_str(), tag.second.c_str(), feature->geometry().points()[0].lon(), feature->geometry().points()[0].lat());
                        it = tags->erase(it);
                    } else if ( utils::match_substring(tag.first, "layer")){
                        it = tags->erase(it);
                    } else {
                        if (utils::match_substring(tag.first, "highway") && (
                                utils::match_substring(tag.second, "primary")
                                || utils::match_substring(tag.second, "secondary")
                                || utils::match_substring(tag.second, "tertiary")
                                || utils::match_substring(tag.second, "motorway")
                                || utils::match_substring(tag.second, "motorway_link")
                                || utils::match_substring(tag.second, "unclassified")
                                || utils::match_substring(tag.second, "service")
                        )) {
                            is_car_road = true;
                        }
                        ++it;
                    }
                }
                if (tags->empty() && has_bridge > 0.5f) {
                    // printf("REMOVING FEATURE that has bridge=%f AT %f:%f\n", has_bridge, feature->geometry().points()[0].lon(), feature->geometry().points()[0].lat());
                    to_remove = true;
                }
                return {is_car_road ? to_remove : false, is_car_road ? has_bridge : 0.0f};
            },
            [](std::vector<float> floats) -> float {
                return std::accumulate(floats.begin(), floats.end(), 0.0f) > 0.5f ? 1.0f : 0.0f;
            },
            StrategyConfig{StrategyConfig::TaskType::REGRESSION, 0}
    ));

    strategies.emplace_back(std::make_unique<FeatureHandlingStrategy>(
            "building_count",
            [](unprocessed::Feature *feature) -> strategyResult {
                auto tags = feature->mutable_tags();
                bool to_remove{false};
                for (auto it = tags->begin(); it != tags->end();) {
                    const auto &tag = *it;
                    if (utils::match_substring(tag.first, "building")) {
                        it = tags->erase(it);
                        to_remove = true;
                    } else {
                        ++it;
                    }
                }
                return {to_remove, to_remove ? 1.0f : 0.0f};
            },
            [](std::vector<float> floats) -> float {
                return std::accumulate(floats.begin(), floats.end(), 0.0f);
            },
            StrategyConfig{StrategyConfig::TaskType::REGRESSION, 0}
    ));

    strategies.emplace_back(std::make_unique<FeatureHandlingStrategy>(
            "max_speed",
            [](unprocessed::Feature *feature) -> strategyResult {
                auto tags = feature->mutable_tags();
                bool to_remove{false};
                float max_speed{-100.0f};
                for (auto it = tags->begin(); it != tags->end();) {
                    const auto &tag = *it;
                    if (utils::match_substring(tag.first, "maxspeed")) {
                        auto parsed_speed = utils::parse_speed(tag.second);
                        max_speed = std::max(parsed_speed, max_speed);
                        it = tags->erase(it);
                    } else {
                        if (utils::match_substring(tag.first, "highway")) {
                            if (max_speed < 0.0f) {
                                max_speed = 0.0f;
                            }
                        }
                        ++it;
                    }
                }

                return {to_remove, max_speed};
            },
            [](const std::vector<float> &floats) -> float {
                auto max_it = std::max_element(floats.begin(), floats.end());
                if (max_it == floats.end()) {
                    return -100.0f;
                }
                return *max_it;
            },
            StrategyConfig{StrategyConfig::TaskType::REGRESSION, 0}
    ));

    // strategies.emplace_back(std::make_unique<FeatureHandlingStrategy>(
    //         "measure roofs",
    //         [](unprocessed::Feature *feature) -> strategyResult {
    //             auto to_measure{false};
    //             float feature_area{0.0f};
    //             auto is_polygon = feature->geometry().is_closed();
    //             for (const auto& tag : feature->mutable_tags()) {
    //                 if (tag.first == "building") {
    //                     to_measure = true;
    //                     break;
    //                 }
    //             }
    //             if (to_measure) {
    //                 feature_area = feature->geometry().
    //             }
    //             return {false, feature_area};
    //         },
    //         [](std::vector<float> floats) -> float {
    //             return std::accumulate(floats.begin(), floats.end(), 0.0f);
    //         },
    //         StrategyConfig{StrategyConfig::TaskType::REGRESSION, 0}
    // ));

    // Create the main output directory
    try {
        if (fs::exists(output_directory) && fs::is_directory(output_directory)) {
            fs::remove_all(output_directory);
        }
        fs::create_directories(output_directory);
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error handling directory: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Process each strategy
    for (auto &strategy_ptr: strategies) {
        FeatureHandlingStrategy &strategy = *strategy_ptr;
        std::cout << "Processing strategy: " << strategy.name << std::endl;
        std::string strategy_output_dir = output_directory + strategy.name + "/";

        // Create strategy-specific output directory
        try {
            fs::create_directories(strategy_output_dir);
        } catch (const fs::filesystem_error &e) {
            std::cerr << "Error creating strategy directory: " << e.what() << std::endl;
            continue; // Skip this strategy if directory creation fails
        }

        // Open strategy-specific logfile
        std::string logfile_path = strategy_output_dir + "/labels.txt";
        std::ofstream logfile_stream(logfile_path, std::ios::binary);
        if (!logfile_stream) {
            std::cerr << "Failed to open logfile: " << logfile_path << std::endl;
            continue; // Skip this strategy if logfile cannot be opened
        }

        // Open strategy-specific configfile
        std::string config_filepath = strategy_output_dir + "/config.yaml";
        std::ofstream configfile(config_filepath, std::ios::binary);
        if (!configfile) {
            std::cerr << "Failed to open configfile: " << config_filepath << std::endl;
            continue; // Skip this strategy if logfile cannot be opened
        }

        // Process the directory with the current strategy
        try {
            process_directory(input_directory, strategy_output_dir, logfile_stream, configfile, strategy);
        } catch (const std::exception &e) {
            std::cerr << "Processing failed for strategy " << strategy.name << ": " << e.what() << std::endl;
            continue; // Continue with next strategy if processing fails
        }
    }

    // Optionally, display a summary of all strategies
    std::cout << "All strategies processed." << std::endl;

    return EXIT_SUCCESS;
}

// Parses command-line arguments and populates input and output directories
int parse_args(std::string &input_directory, std::string &output_directory, int argc, char *argv[]) {
    cxxopts::Options options("osm_task_generator",
                             "Remove specific features from OSM pbf files and store the results.");
    options.add_options()
            ("i,input-directory", "Directory containing unprocessed OSM pbf files", cxxopts::value<std::string>())
            ("o,output-directory", "Output directory for tiles and log", cxxopts::value<std::string>())
            ("c,config-directory", "Directory containing configuration files",
             cxxopts::value<std::string>()->default_value(""))
            ("h,help", "Print help", cxxopts::value<bool>()->default_value("false"));

    options.parse_positional({"input-directory"});
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result["help"].as<bool>()) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    if (!result.count("input-directory") || !result.count("output-directory")) {
        std::cerr << "Both input and output directories are required." << std::endl;
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    input_directory = result["input-directory"].as<std::string>();
    output_directory = result["output-directory"].as<std::string>() + "/";
    return -1; // Indicates successful parsing
}

// Reads a TileGroup from a file
void read_file(const std::string &input_file, unprocessed::TileGroup &tile_group) {
    std::ifstream file(input_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + input_file);
    }
    if (!tile_group.ParseFromIstream(&file)) {
        throw std::runtime_error("Failed to parse tile group from " + input_file);
    }
    file.close();
}

// Writes a TileGroup to a file
void write_file(const std::string &output_file, const unprocessed::TileGroup &tile_group) {
    std::lock_guard<std::mutex> lock(write_mutex);
    if (output_files.find(output_file) != output_files.end()) {
        throw std::runtime_error("Output file " + output_file + " already exists");
    }
    output_files.insert(output_file);

    std::ofstream file(output_file, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file " + output_file);
    }
    if (!tile_group.SerializeToOstream(&file)) {
        throw std::runtime_error("Failed to write tile group to " + output_file);
    }
}


std::pair<int, int> get_parent_tile(int x, int y) {
    const int zoom_diff = 16 - 14;
    const int divisor = 1 << zoom_diff;

    int px = x / divisor;
    int py = y / divisor;

    return std::make_pair(px, py);
}

// Processes a single TileGroup: counts and removes features based on the strategy
void process_tile_group(
        unprocessed::TileGroup &tile_group,
        std::ofstream &logfile,
        FeatureHandlingStrategy &strategy
) {
    int num_out_of_bound_tiles_removed{0};
    // Remove tiles that are not part of the current tile group
    auto tiles = tile_group.mutable_tiles();
    for (auto it = tiles->begin(); it != tiles->end();) {
        auto pc = get_parent_tile(it->x(), it->y());
        if (pc.first != tile_group.x() || pc.second != tile_group.y()) {
            it = tiles->erase(it);
            ++num_out_of_bound_tiles_removed;
        } else {
            ++it;
        }
    }
//    if (num_out_of_bound_tiles_removed > 0) {
//        printf("Removed %d out of bound tiles\n", num_out_of_bound_tiles_removed);
//    }

    for (auto &tile: *tile_group.mutable_tiles()) {
        size_t num_removed_in_tile = 0;
        std::vector<float> values_to_add{};
        // if(tile.x() == 10495 && tile.y() == 25298) {
        //     printf("Processing tile %d:%d\n", tile.x(), tile.y());
        //     for(const auto& feat: tile.features()) {
        //         for(const auto& tag: feat.tags()) {
        //             printf("%s=%s\n", tag.first.c_str(), tag.second.c_str());
        //         }
        //     }
        //     for(const auto& feat: tile.features()) {
        //         printf("Feature at %f, %f\n", feat.geometry().points()[0].lon(), feat.geometry().points()[0].lat());
        //     }
        // }

        // Access features directly
        auto features = tile.mutable_features();
        int feats_before_remove = features->size();
        int to_remove = 0;
        int fid = 0;
        // printf("\n");
        // for (const auto& group: tile.groups()) {
        //     for (const auto& index: group.feature_indices()) {
        //         printf("%d,", index);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
        for (auto feat_it = features->begin(); feat_it != features->end();) {
            const unprocessed::Feature &feature = *feat_it;

            auto strat_result = strategy.should_remove(
                    tile.mutable_features(static_cast<int>(feat_it - features->begin())));

            // Determine if the feature should be removed based on the strategy
            if (strat_result.should_remove) {
                // printf("remove feature %d of %d\n", static_cast<int>(feat_it - features->begin())+to_remove, features->size());
                ++to_remove;
                // Remove the feature
                feat_it = features->erase(feat_it);

                // make sure to remove the feature from all groups, and decrement the indices of all features after the removed one
                for (int i = 0; i < tile.groups_size(); ++i) {
                    auto group = tile.mutable_groups(i);
                    auto indices = group->mutable_feature_indices();
                    for (auto it = indices->begin(); it != indices->end();) {
                        if (*it == fid) {
                            it = indices->erase(it);
                        } else if (*it > fid) {
                            group->set_feature_indices(static_cast<int>(it - indices->begin()), *it - 1);
                            //(*it)--;
                            ++it;
                        } else {
                            ++it;
                        }
                    }

                }
            } else {
                ++feat_it;
            }
            values_to_add.push_back(strat_result.value_to_add);
            fid++;
        }
        // printf("\n");
        // for (const auto& group: tile.groups()) {
        //     for (const auto& index: group.feature_indices()) {
        //         printf("%d,", index);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
        // printf("before: %d\nto_remove: %d\nafter: %d\n", feats_before_remove, to_remove, features->size());
        const auto total_value = strategy.accumulate(values_to_add);

        // Update tile and file counters
        strategy.tiles_processed.fetch_add(1, std::memory_order_relaxed);

        // Thread-safe logging
        {
            std::lock_guard<std::mutex> lock(strategy.logfile_mutex);
            logfile << tile.zoom() << "_" << tile.x() << "_" << tile.y() << ":" << total_value << "\n";
        }
        //TODO REMOVE THIS LATER
        // tile.clear_groups();
        //TODO ENDS
        for (const auto &group: tile.groups()) {
            for (const auto &index: group.feature_indices()) {
                if (index >= tile.features_size()) {
                    printf("\nFeature index out of bounds [%d,%d]\n", index, tile.features_size());
                    throw std::runtime_error("Feature index out of bounds");
                }
            }
        }
    }
}

// Processes a single file: reads, processes, and writes TileGroup
void count_features(const std::string &file_path, const std::string &output_directory, std::ofstream &logfile_stream,
                    FeatureHandlingStrategy &strategy) {
    unprocessed::TileGroup tile_group;
    read_file(file_path, tile_group);

    process_tile_group(tile_group, logfile_stream, strategy);

    const std::string output_file = output_directory +
                                    std::to_string(tile_group.zoom()) + "_" +
                                    std::to_string(tile_group.x()) + "_" +
                                    std::to_string(tile_group.y()) + ".pbf";

    write_file(output_file, tile_group);

    // Update global file counter
    strategy.files_processed.fetch_add(1, std::memory_order_relaxed);
}

// Processes all files in the input directory for a given strategy
void process_directory(const std::string &input_directory, const std::string &output_directory,
                       std::ofstream &logfile_stream, std::ofstream &configfile_stream,
                       FeatureHandlingStrategy &strategy) {
    using namespace indicators;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Collect all .pbf files
    std::vector<fs::directory_entry> files;
    for (const auto &entry: fs::recursive_directory_iterator(input_directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pbf") {
            files.emplace_back(entry);
        } else {
            // std::cerr << "Skipping non-pbf file: " << entry.path().filename().string() << std::endl;
        }
    }

    const size_t num_files = files.size();

    if (num_files == 0) {
        std::cout << "No .pbf files found in the input directory." << std::endl;
        return;
    }

    // Setup progress bar
    BlockProgressBar progress_bar{
            option::BarWidth{50},
            option::Start{"["},
            option::End{"]"},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
            option::MaxProgress{num_files}
    };

    BS::thread_pool pool;
    std::atomic<size_t> num_processed{0};
    std::mutex progress_mutex;

    // Function to update progress bar
    auto update_progress = [&]() {
        size_t processed = num_processed.fetch_add(1, std::memory_order_relaxed) + 1;
        {
            std::lock_guard<std::mutex> lock(progress_mutex);
            progress_bar.set_option(option::PostfixText{
                    std::to_string(processed) + "/" + std::to_string(num_files)
            });
            progress_bar.tick();
        }
    };

    // Process each file
    const bool parallel = true;
    for (const auto &file: files) {
        auto task = [&]() {
            try {
                count_features(file.path().string(), output_directory,
                               logfile_stream, strategy);
            } catch (const std::exception &e) {
                std::cerr << "\nError processing file " << file.path() << ": " << e.what() << std::endl;
            }
            update_progress();
        };
        if (parallel) {
            auto res = pool.submit_task(task);
        } else {
            task();
        }
    }

    pool.wait();
    progress_bar.mark_as_completed();

    // Write to configFile
    auto predictionType = strategy.config.predictionType;
    configfile_stream << StrategyConfig::taskTypeToString(predictionType) << std::endl;
    if (predictionType != StrategyConfig::REGRESSION) {
        configfile_stream << ", " << strategy.config.predictionType << std::endl;
    }
    configfile_stream.close();


    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Display strategy-specific statistics
    std::cout << "\nStrategy: " << strategy.name
              // << "\nFeatures removed: " << strategy.features_removed.load()
              << "\nTiles processed: " << strategy.tiles_processed.load()
              << "\nFiles processed: " << strategy.files_processed.load()
              << "\nTotal time: " << total_duration << " ms\n" << std::endl;
}

void print_contents(const unprocessed::TileGroup &tile_group) {
    for (auto &tile: tile_group.tiles()) {
        int num_signals{0};

        for (auto &feat: tile.features()) {  // Mutable access to features
            // Erase matching tags using a for-each loop and an iterator
            for (auto &tag: feat.tags()) {
                if (tag.first == "traffic_signals" ||
                    tag.first == "highway" && tag.second == "traffic_signals") {
                    ++num_signals;
                    break;
                }
            }
        }
        printf("%d, %d Num tiles: %d\n", tile.x(), tile.y(), num_signals);
    }
}
