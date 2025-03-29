//
// Created by Ludvig Delvret on 2024-11-21.
//

#include "co-occurrence.h"

#include <BS_thread_pool.hpp>

std::mutex vector_mutex;
std::mutex stats_mutex;

// Maps to store co-occurrence counts
occurrenceMap feature_tag_cooccurrence; // Tag pairs in the same feature
occurrenceMap tile_tag_cooccurrence;    // Tag pairs in the same tile
std::unordered_map<std::string, std::vector<std::string>> existing_key_vals;

std::vector<std::string> chosen_tags{"bridge", "traffic_signals", "maxspeed", "building"};

stats total_stats;

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    std::string input_directory;
    std::string output_directory;
    std::string csv_file;
    int status = parse_args(input_directory, output_directory, csv_file, argc, argv);
    if (status != -1) return status;

    // Create the main output directory
    try {
        if (!fs::exists(output_directory) || !fs::is_directory(output_directory)) {
            fs::create_directories(output_directory);
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error handling directory: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    existing_key_vals = readCSV(csv_file);

    // Collect all .pbf files
    std::vector<processed::TileGroup> tile_groups;
    for (const auto &entry: fs::recursive_directory_iterator(input_directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pbf") {
            processed::TileGroup tile_group;
            read_file(entry.path().string(), tile_group);
            tile_groups.emplace_back(tile_group);
        } else {
            // std::cerr << "Skipping non-pbf file: " << entry.path().filename().string() << std::endl;
        }
    }

    // Setup progress bar
    BlockProgressBar progress_bar{
            option::BarWidth{50},
            option::Start{"["},
            option::End{"]"},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
            option::MaxProgress{tile_groups.size()}
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
                    std::to_string(processed) + "/" + std::to_string(tile_groups.size())
            });
            progress_bar.tick();
        }
    };

    bool parallel = true;
    for (const auto& group : tile_groups) {
        auto task = [&]() {
            try {
                auto [stat, vectors] = count_occurences(group);
                add_to_vector(vectors);
                combine_stats(stat);
            } catch (const std::exception &e) {
                std::cerr << "\nError processing file " << group.zoom() << "_" << group.x() << "_" << group.y() << ": " << e.what() << std::endl;
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

    // Setup progress bar
    BlockProgressBar feature_stream_bar{
            option::BarWidth{50},
            option::Start{"["},
            option::End{"]"},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::ForegroundColor{Color::grey},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
            option::MaxProgress{feature_tag_cooccurrence.size()/1000}
    };

    std::ostringstream feat_buffer;
    int progress = 0;
    for (auto &[tag_pair, count] : feature_tag_cooccurrence) {
        for (auto &c:tag_pair.first) {
            if ( c != '\n') {
                feat_buffer << c;
            } else {
                feat_buffer << " ";
            }
        }
        feat_buffer << ",";
        for (auto &c:tag_pair.second) {
            if ( c != '\n') {
                feat_buffer << c;
            } else {
                feat_buffer << " ";
            }
        }
        feat_buffer << ",";
        feat_buffer << count << std::endl;
        if (progress % 1000 == 0) {
            feature_stream_bar.tick();
            feature_stream_bar.set_option(option::PostfixText{
                    std::to_string(progress) + "/" + std::to_string(feature_tag_cooccurrence.size())
            });
        }
        ++progress;
    }
    std::ofstream feat_outstream(output_directory + "/feat_cooccurrence.csv");
    if (feat_outstream.is_open()) {
        feat_outstream << feat_buffer.str();
    } else {
        std::cerr << "Error opening file for writing.\n";
    }

    feature_stream_bar.mark_as_completed();

    // Setup progress bar
    BlockProgressBar tile_stream_bar{
            option::BarWidth{50},
            option::Start{"["},
            option::End{"]"},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::ForegroundColor{Color::grey},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
            option::MaxProgress{tile_tag_cooccurrence.size()/1000}
    };

    std::stringstream tile_buffer;
    progress = 0;
    for (auto &[tag_pair, count] : tile_tag_cooccurrence) {
        for (auto &c:tag_pair.first) {
            if ( c != '\n') {
                tile_buffer << c;
            } else {
                tile_buffer << " ";
            }
        }
        tile_buffer << ",";
        for (auto &c:tag_pair.second) {
            if ( c != '\n') {
                tile_buffer << c;
            } else {
                tile_buffer << " ";
            }
        }
        tile_buffer << ",";
        tile_buffer << count << std::endl;

        if (progress%1000 == 0) {
            tile_stream_bar.tick();
            tile_stream_bar.set_option(option::PostfixText{
                    std::to_string(progress) + "/" + std::to_string(tile_tag_cooccurrence.size())
            });
        }
        ++progress;
    }
    std::ofstream tile_outstream(output_directory + "/tile_cooccurrence.csv");
    if (tile_outstream.is_open()) {
        tile_outstream << tile_buffer.str();
    } else {
        std::cerr << "Error opening file for writing.\n";
    }
    tile_stream_bar.mark_as_completed();

    printf("Total number of tiles: %d\n", total_stats.num_tiles);
    printf("Total number of features: %d\n", total_stats.num_features);
    printf("Task tag counts: \n");
    for (auto &[key, val]: total_stats.chosen_tag_counts) {
        printf("%s: %d\n", key.c_str(), val);
    }

    feat_outstream.close();
    tile_outstream.close();
}

// Reads a TileGroup from a file
void read_file(const std::string &input_file, processed::TileGroup &tile_group) {
    std::ifstream file(input_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + input_file);
    }
    if (!tile_group.ParsePartialFromIstream(&file)) {
        throw std::runtime_error("Failed to parse tile group from " + input_file);
    }
    file.close();
}

// Parses command-line arguments and populates input and output directories
int parse_args(std::string &input_directory, std::string &output_directory, std::string &csv_file, int argc, char *argv[]) {
    cxxopts::Options options("Tag co-occurences",
                             "Identify what other tags commonly occur alongside specific tags");
    options.add_options()
            ("i,input-directory", "Directory containing processed OSM pbf files", cxxopts::value<std::string>())
            ("o,output-directory", "Output directory for stat files", cxxopts::value<std::string>())
            ("c,csv_file", "Directory containing features.csv files", cxxopts::value<std::string>())
            //  cxxopts::value<std::string>()->default_value(""))
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

    if (!result.count("input-directory") || !result.count("output-directory") || !result.count("csv_file")) {
        std::cerr << "Both input and output directories, as well as csv_file are required." << std::endl;
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    input_directory = result["input-directory"].as<std::string>();
    output_directory = result["output-directory"].as<std::string>() + "/";
    csv_file = result["csv_file"].as<std::string>();
    return -1; // Indicates successful parsing
}

std::unordered_map<std::string, std::vector<std::string>> readCSV(const std::string& filename) {
    std::unordered_map<std::string, std::vector<std::string>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string key, value;
    char ch;
    bool parsingKey = true;

    while (file.get(ch)) {
        if (ch == ';' && parsingKey) {
            parsingKey = false; // Found the key-value separator
        } else if (ch == '\n') {
            // End of line: Remove the last character from value and store the pair
            if (!key.empty()) {
                if (!value.empty()) {
                    value.pop_back(); // Remove the last character
                    value.pop_back(); // Remove the last character
                }
                data[key].push_back(value); // Add the value to the vector for this key
            }
            key.clear();
            value.clear();
            parsingKey = true; // Reset for the next line
        } else {
            // Append to key or value based on context
            if (parsingKey) {
                key += ch;
            } else {
                value += ch;
            }
        }
    }

    // Add the last key-value pair if the file doesn't end with a newline
    if (!key.empty()) {
        if (!value.empty()) {
            value.pop_back(); // Remove the last character
        }
        data[key].push_back(value); // Add the value to the vector for this key
    }

    file.close();
    return data;
}

// Transfer map data into a sortable vector and sort by count
template <typename Map>
std::vector<std::pair<typename Map::key_type, typename Map::mapped_type>>
sort_by_occurrences(const Map& cooccurrence_map) {
    // Create a vector of pairs from the map
    std::vector<std::pair<typename Map::key_type, typename Map::mapped_type>> sorted_data(
        cooccurrence_map.begin(), cooccurrence_map.end()
    );

    // Sort the vector by the second element (occurrence count) in descending order
    std::sort(sorted_data.begin(), sorted_data.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second; // Descending order
        }
    );

    return sorted_data;
}

std::vector<std::string> split(const std::string& str, const char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(str); // Create a stringstream from the input string

    // Read each token separated by the delimiter
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token); // Add the token to the vector
    }
    return tokens;
}

std::string join(const std::vector<std::string>& vec, const size_t start, const size_t end,  const char delimiter) {
    std::ostringstream joined;

    // Loop through the specified subvector range
    for (size_t i = start; i < end; ++i) {
        // Add the string to the result
        joined << vec[i];

        // Add the delimiter if it's not the last element
        if (i < end - 1) {
            joined << delimiter;
        }
    }

    return joined.str();
}

std::pair<std::string, std::string> handle_tags(const std::string& key, const std::string& value) {
    if (existing_key_vals.find(key) != existing_key_vals.end()) {
        if (std::find(existing_key_vals[key].begin(), existing_key_vals[key].end(), value) != existing_key_vals[key].end()) {
                // Key: Val pair is found, no action necessary
                return {key, value};
        }
        if (std::find(existing_key_vals[key].begin(), existing_key_vals[key].end(), "") != existing_key_vals[key].end()) {
            // Key exists, but not matching value. Empty value does exist
            return {key, ""};
        }
        // Key: "" doesn't exist, but there is a matching key, and we have a value
        return {key, "nonexistant"};
        // printf("Key:value %s; %s not found\n", key.c_str(), value.c_str());
    } else {
        const auto key_tokens = split(key, ':');
        for (size_t i{key_tokens.size() - 1}; i > 0; --i ) {
            auto key_string = join(key_tokens, 0, i, ':');
            if (existing_key_vals.find(key_string + ":*") != existing_key_vals.end()) {
                return {key_string, ""};
            }
            if (existing_key_vals.find(key_string) != existing_key_vals.end()) {
                // printf("Key %s found, value: %s, previous: %s\n", key_string.c_str(), value.c_str(), key.c_str());
                return {key_string, ":nonexistant subkey"};
            }
        }
        // printf("Key %s not found, value: %s\n", key.c_str(), value.c_str());
    }
    return {"",""};
}

std::pair<stats, std::pair<occurrenceMap, occurrenceMap>> count_occurences(const processed::TileGroup& group){
    // Maps to store co-occurrence counts
    std::map<std::pair<std::string, std::string>, int> feature_tag_cooccurrence; // Tag pairs in the same feature
    std::map<std::pair<std::string, std::string>, int> tile_tag_cooccurrence;    // Tag pairs in the same tile
    stats local_stats{};
    for (const auto& tile : group.tiles()) {
        std::set<std::string> tile_tags; // Unique tag pairs for this tile

        ++local_stats.num_tiles;

        for (const auto& feat : tile.features()) {
            ++local_stats.num_features;
            std::vector<std::string> feature_tags;

            // Form explicit key-value tag pairs
            const auto& tags = feat.tags();
            for (size_t i = 0; i + 1 < tags.size(); i += 2) {
                const auto &[key, val] = handle_tags(tags[static_cast<int>(i)], tags[static_cast<int>(i) + 1]);
                if (key == "") continue;
                if (std::find(chosen_tags.begin(), chosen_tags.end(), key) != chosen_tags.end()) {
                    ++total_stats.chosen_tag_counts[key];
                }
                feature_tags.emplace_back(key + ":" + val);
            }

            // Count co-occurrences within the feature
            for (size_t i = 0; i < feature_tags.size(); i++) {
                for (size_t j = i + 1; j < feature_tags.size(); j++) {
                    auto tag_pair = std::minmax(feature_tags[i], feature_tags[j]); // Ensure consistent ordering
                    feature_tag_cooccurrence[tag_pair]++;
                }
            }

            // Add feature tags to the tile's tag set
            tile_tags.insert(feature_tags.begin(), feature_tags.end());
        }

        // Count co-occurrences within the tile
        std::vector<std::string> tile_tag_vector(tile_tags.begin(), tile_tags.end());
        for (size_t i = 0; i < tile_tag_vector.size(); i++) {
            for (size_t j = i + 1; j < tile_tag_vector.size(); j++) {
                auto tag_pair = std::minmax(tile_tag_vector[i], tile_tag_vector[j]); // Ensure consistent ordering
                tile_tag_cooccurrence[tag_pair]++;
            }
        }
    }
    return std::make_pair(local_stats, std::make_pair(feature_tag_cooccurrence, tile_tag_cooccurrence));
}

void add_to_vector(const std::pair<occurrenceMap, occurrenceMap>& map_pair) {
    std::lock_guard<std::mutex> lock(vector_mutex);
    for (const auto&[key, value] : map_pair.first) {
        // Add the value to the result map, creating the key if it doesn't exist
        feature_tag_cooccurrence[key] += value;
    }
    for (const auto&[key, value] : map_pair.second) {
        // Add the value to the result map, creating the key if it doesn't exist
        tile_tag_cooccurrence[key] += value;
    }
}

void combine_stats(const stats& stat) {
    std::lock_guard<std::mutex> lock(stats_mutex);
    for (const auto &[fst, snd]: stat.chosen_tag_counts) {
        total_stats.chosen_tag_counts[fst]  += snd;
    }
    total_stats.num_features += stat.num_features;
    total_stats.num_tiles += stat.num_tiles;
}

occurrenceMap combine_maps(const std::vector<occurrenceMap>& vec_of_maps) {
    occurrenceMap result_map{};
    // Iterate over each map in the vector
    for (const auto& single_map : vec_of_maps) {
        // Iterate over each key-value pair in the current map
        for (const auto&[key, value] : single_map) {
            // Add the value to the result map, creating the key if it doesn't exist
            result_map[key] += value;
        }
    }
    return result_map;
}


