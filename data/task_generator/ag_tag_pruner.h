#pragma once
//
// Created by Ludvig Delvret on 2024-12-05.
//

#ifndef AG_TAG_PRUNER_H
#define AG_TAG_PRUNER_H

#include "proto/unprocessed_tile_group.pb.h"
#include "proto/processed_tile_group.pb.h"
#include <iostream>
#include "cxxopts.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <boost/geometry/strategies/buffer.hpp>

#include "indicators.hpp"
using namespace indicators;
namespace fs = std::filesystem;

struct buffers {
    std::string tags;
    std::string images;
    std::string tags_images;
};

using subtile = std::string;
using occurrenceMap = std::map<std::string, int>;
using keyval_map = std::map<std::string, occurrenceMap>;

int parse_args(std::string &input_directory, std::string &output_directory, std::string &csv_string, float &p, int argc, char *argv[]);

void read_file(const std::string &input_file, processed::TileGroup &tile_group);

std::pair<std::vector<int>, std::map<std::string, int>> count_occurrences(const processed::TileGroup& group);

std::pair<keyval_map, std::map<std::string, std::vector<std::string>>>  readCSV(const std::string& filename);

void add_to_buffer(const buffers& subtile_vector);

std::pair<std::string, std::string> handle_tags(const std::string& key, const std::string& value);

std::pair<std::string, std::string> setup_tags(const std::string &key, const std::string &value);

occurrenceMap combine_maps(const std::vector<occurrenceMap>& vec_of_maps);

void print_results(const std::string& output_directory);

void gen_new_ag(const std::string& output_directory, const std::string &input_file);

#endif //AG_TAG_PRUNER_H
