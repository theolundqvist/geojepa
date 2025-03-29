#pragma once
//
// Created by Ludvig Delvret on 2024-11-26.
//

#ifndef AG_CSV_GENERATOR_H
#define AG_CSV_GENERATOR_H

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

buffers count_occurrences(const processed::TileGroup& group);

keyval_map readCSV(const std::string& filename);

void add_to_buffer(const buffers& subtile_vector);

void setup_tags(std::string &key, std::string &value);

std::pair<std::string, std::string> handle_tags(const std::string& key, const std::string& value);

occurrenceMap combine_maps(const std::vector<occurrenceMap>& vec_of_maps);

#endif //AG_CSV_GENERATOR_H
