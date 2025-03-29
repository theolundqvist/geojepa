#include <utility>
#include "utils.h"

//
// Created by Ludvig Delvret on 2024-10-30.
//

#ifndef GENERATOR_H
#define GENERATOR_H

struct StrategyConfig {
    enum TaskType {
        CLASSIFICATION,
        REGRESSION
    };

    TaskType predictionType;
    int num_choices;

    // Constructor
    StrategyConfig(const TaskType predictionType, const int num_choices)
        : predictionType(predictionType), num_choices(num_choices) {}

    // Static function to map TaskType enum to string
    static std::string taskTypeToString(const TaskType type) {
        switch (type) {
            case CLASSIFICATION: return "CLASSIFICATION";
            case REGRESSION:     return "REGRESSION";
            default:             return "UNKNOWN";
        }
    }
};

struct strategyResult {
    bool should_remove{false};
    float value_to_add{0.0};
};

// Structure to encapsulate feature removal logic and statistics
struct FeatureHandlingStrategy {
    std::string name; // Strategy name
    std::function<strategyResult(unprocessed::Feature *)> should_remove; // Removal predicate
    std::function<float(std::vector<float> floats_to_add)> accumulate;
    StrategyConfig config;

    // Statistics counters
    std::atomic<size_t> tiles_processed{0};
    std::atomic<size_t> files_processed{0};
    std::mutex logfile_mutex{};

    // Constructor
    FeatureHandlingStrategy(std::string n, std::function<strategyResult(unprocessed::Feature *)> remove_func,
    std::function<float(std::vector<float> floats_to_add)> accumulate, const StrategyConfig config)
            : name(std::move(n)), should_remove{std::move(remove_func)}, accumulate{std::move(accumulate)}, config{config} {}
};

// Function declarations
void read_file(const std::string &input_file, unprocessed::TileGroup &tile_group);

void write_file(const std::string &output_file, const unprocessed::TileGroup &tile_group);

void process_tile_group(unprocessed::TileGroup &tile_group, std::ofstream &logfile, FeatureHandlingStrategy &strategy);

void count_features(const std::string &file_path, const std::string &output_directory, std::ofstream &logfile,
                    FeatureHandlingStrategy &strategy);

void process_directory(const std::string &input_directory, const std::string &output_directory, std::ofstream &logfile_stream,
                        std::ofstream &configfile_stream, FeatureHandlingStrategy &strategy);

int parse_args(std::string &input_directory, std::string &output_directory, int argc, char *argv[]);

void print_contents(const unprocessed::TileGroup &tile_group);


#endif //GENERATOR_H
