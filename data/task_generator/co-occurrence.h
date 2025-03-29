#pragma once
//
// Created by Ludvig Delvret on 2024-11-21.
//

#ifndef CO_OCCURENCE_H

#include "proto/unprocessed_tile_group.pb.h"
#include "proto/processed_tile_group.pb.h"
#include <iostream>
#include "cxxopts.h"
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "indicators.hpp"
using namespace indicators;
namespace fs = std::filesystem;

struct stats {
    std::map<std::string, int> chosen_tag_counts;
    int num_tiles{};
    int num_features{};
};

using occurrenceMap = std::map<std::pair<std::string, std::string>, int>;

int parse_args(std::string &input_directory, std::string &output_directory, std::string &csv_file, int argc, char *argv[]);

void read_file(const std::string &input_file, processed::TileGroup &tile_group);

template <typename Map>
std::vector<std::pair<typename Map::key_type, typename Map::mapped_type>>
sort_by_occurrences(const Map& cooccurrence_map);

std::pair<stats, std::pair<occurrenceMap, occurrenceMap>> count_occurences(const processed::TileGroup& group);

void add_to_vector(const std::pair<occurrenceMap, occurrenceMap>& map_pair);

std::unordered_map<std::string, std::vector<std::string>> readCSV(const std::string& filename);

std::pair<std::string, std::string> handle_tags(const std::string& key, const std::string& value);

occurrenceMap combine_maps(const std::vector<occurrenceMap>& vec_of_maps);

void combine_stats(const stats& stat);
#define CO_OCCURENCE_H

#endif //CO_OCCURENCE_H
