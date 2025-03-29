//
// Created by Ludvig Delvret on 2024-11-26.
//

#include "ag_csv_generator.h"

#include <BS_thread_pool.hpp>
#include <boost/algorithm/string/predicate.hpp>

std::mutex buffer_mutex;


keyval_map existing_key_vals;
size_t existing_key_vals_size{};
std::map<std::string, float> labels{};
std::vector<int> counts{};
std::stringstream tags_buffer;
std::stringstream images_buffer;
std::stringstream tags_images_buffer;

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    std::string input_directory;
    std::string output_directory;
    std::string csv_file;
    float p;
    int status = parse_args(input_directory, output_directory, csv_file, p, argc, argv);
    if (status != -1) return status;

    std::ifstream labels_file(input_directory + "/labels.txt");

    if (!labels_file) {
        std::cerr << "Error opening labels file!" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(labels_file, line)) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            float value = std::stof(line.substr(pos + 1));
            labels[key] = value;
        }
    }

    std::vector<std::string> task_division{"train", "test", "val"};
    existing_key_vals = readCSV(csv_file);

    bool print{false};
    for (auto &[k, v] : existing_key_vals) {
        if (print) printf("key: %s", k.c_str());
        for (auto &[str, num] : v) {
            ++existing_key_vals_size;
            if (print) printf("\t%s", str.c_str());
        }
        if (print) printf("\n");
    }
    if (print) printf("existing_key_vals_size: %lu\n", existing_key_vals_size);
    // return 1;

    for (const auto &div: task_division) {
        auto div_output_directory = output_directory + "/" + div;
        auto div_input_directory = input_directory + "/" + div;
        // Create the main output directory
        try {
            if (!fs::exists(div_output_directory) || !fs::is_directory(div_output_directory)) {
                fs::create_directories(div_output_directory);
            }
        } catch (const fs::filesystem_error &e) {
            std::cerr << "Error handling directory: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        tags_buffer = std::stringstream();
        images_buffer = std::stringstream();
        tags_images_buffer = std::stringstream();

        // Collect all .pbf files
        std::vector<processed::TileGroup> tile_groups;
        for (const auto &entry: fs::recursive_directory_iterator(div_input_directory)) {
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
                option::ForegroundColor{Color::unspecified},
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


        // // Setup progress bar
        // BlockProgressBar subtiles_stream_bar{
        //     option::BarWidth{50},
        //     option::Start{"["},
        //     option::End{"]"},
        //     option::ShowElapsedTime{true},
        //     option::ShowRemainingTime{true},
        //     option::ForegroundColor{Color::unspecified},
        //     option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
        //     option::MaxProgress{all_subtiles_map.size()/100}
        // };

        int progress = 0;

        tags_buffer << "subtile,label";
        images_buffer << "subtile,label,image";
        tags_images_buffer << "subtile,label,image";

        for (size_t i = 0; i < existing_key_vals_size; i++) {
            tags_buffer << "," << i;
            tags_images_buffer << "," << i;
        }

        tags_buffer << std::endl;
        images_buffer << std::endl;
        tags_images_buffer << std::endl;

        bool parallel = true;
        for (const auto &group: tile_groups) {
            auto task = [&]() {
                try {
                    auto subtile_buffers = count_occurrences(group);
                    add_to_buffer(subtile_buffers);
                } catch (const std::exception &e) {
                    std::cerr << "\nError processing file " << group.zoom() << "_" << group.x() << "_" << group.y()
                              << ": " << e.what() << std::endl;
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

        std::ofstream tags_stream(div_output_directory + "/ag_tags_tiling.csv");
        if (tags_stream.is_open()) {
            tags_stream << tags_buffer.str();
        } else {
            std::cerr << "Error opening file for writing.\n";
        }
        tags_stream.close();
        std::ofstream images_stream(div_output_directory + "/ag_images_tiling.csv");
        if (images_stream.is_open()) {
            images_stream << images_buffer.str();
        } else {
            std::cerr << "Error opening file for writing.\n";
        }
        images_stream.close();

        std::ofstream tags_images_stream(div_output_directory + "/ag_tags_images_tiling.csv");
        if (tags_images_stream.is_open()) {
            tags_images_stream << tags_images_buffer.str();
        } else {
            std::cerr << "Error opening file for writing.\n";
        }
        tags_images_stream.close();
    }
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
int
parse_args(std::string &input_directory, std::string &output_directory, std::string &csv_file, float &p, int argc, char *argv[]) {
    cxxopts::Options options("Tag co-occurences",
                             "Identify what other tags commonly occur alongside specific tags");
    options.add_options()
            ("i,input-directory", "Directory containing processed OSM pbf files", cxxopts::value<std::string>())
            ("o,output-directory", "Output directory for stat files", cxxopts::value<std::string>())
            ("c,csv_file", "Directory containing features.csv files", cxxopts::value<std::string>())
            //  cxxopts::value<std::string>()->default_value(""))
            ("h,help", "Print help", cxxopts::value<bool>()->default_value("false"))
            ("p", "specify least common fraction of columns to remove", cxxopts::value<float>()->default_value("0.1"));

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
    p = result["p"].as<float>();
    return -1; // Indicates successful parsing
}

keyval_map readCSV(const std::string &filename) {
    keyval_map data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string key, value;
    char ch;
    bool parsingKey = true;
    int tag_id{0};

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
                setup_tags(key, value);
                data[key][value] = tag_id; // Add the value to the vector for this key
                ++tag_id;
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
        data[key][value] = 0; // Add the value to the vector for this key
    }

    file.close();
    return data;
}

void setup_tags(std::string &key,std::string &value) {
    std::replace(key.begin(), key.end(), '\n', ' ');
    std::replace(value.begin(), value.end(), '\n', ' ');
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
    // auto print = key.rfind("address", 0) == 0;
    auto print = key.find("water") != std::string::npos;
    print = false;
    if (print) printf("IN: %s = %s\n", key.c_str(), value.c_str());
    if (print) printf("key_vals size: %lu\n", existing_key_vals_size);

    if (existing_key_vals.find(key) != existing_key_vals.end()) {
        if (print) printf("Key found: %s\n", key.c_str());
            if (std::find_if(existing_key_vals[key].begin(), existing_key_vals[key].end(),
                             [value](const auto& pair) { return pair.first == value; }) != existing_key_vals[key].end()) {
                // Key: Val pair is found, no action necessary
                return {key, value};
        }
            if (std::find_if(existing_key_vals[key].begin(), existing_key_vals[key].end(),
                             [](const auto& pair) { return pair.first == "*"; }) != existing_key_vals[key].end()) {
            // Key exists, but not matching value. Empty value does exist
            return {key, ""};
        }
            if (std::find_if(existing_key_vals[key].begin(), existing_key_vals[key].end(),
                             [](const auto& pair) { return pair.first == "*"; }) != existing_key_vals[key].end()) {
            // Key exists, but not a direct match. Wildcard value does exist
            return {key, "*"};
        }
        if (print) printf("Key found: %s, but no value: %s\n", key.c_str(), value.c_str());
    } else {
        if (print) printf("Key not found: %s\n", key.c_str());
        // print = true;
    }
    if (print) printf("Key: %s\n", key.c_str());
    const auto key_tokens = split(key, ':');
    for (size_t i{key_tokens.size() - 1}; i > 0; --i ) {
        auto sub_key = join(key_tokens, 0, i, ':');
        if (print) printf("Subkey: %s\n", sub_key.c_str());
        if (existing_key_vals.find(sub_key + ":*") != existing_key_vals.end()) {
            if (print) printf("Subkey: %s:*\n", sub_key.c_str());
            return {sub_key + ":*", ""};
        }
        if (existing_key_vals.find(sub_key) != existing_key_vals.end()) {
            if (std::find_if(existing_key_vals[key].begin(), existing_key_vals[key].end(),
                             [](const auto& pair) { return pair.first == "*"; }) != existing_key_vals[key].end()) {
                // Key exists, but not a direct match. Wildcard value does exist
                if (print) printf("Subkey: %s, val: *\n", sub_key.c_str());
                return {sub_key, "*"};
            }
            if (std::find_if(existing_key_vals[key].begin(), existing_key_vals[key].end(),
                             [](const auto& pair) { return pair.first == ""; }) != existing_key_vals[key].end()) {
                // Key exists, but not a direct match. Wildcard value does exist
                if (print) printf("Subkey: %s, val: empty\n", sub_key.c_str());
                return {sub_key, ""};
            }
            // printf("Key %s found, value: %s, previous: %s\n", key_string.c_str(), value.c_str(), key.c_str());
            if (print) printf("Subkey: %s, nonexistant subkey: \n", sub_key.c_str());
        }
    }
    if (print) printf("Key %s not found, value: %s\n", key.c_str(), value.c_str());
    return {"UNK",""};
}

buffers count_occurrences(const processed::TileGroup &group) {

    // Maps to store co-occurrence counts
    std::map<std::string, std::vector<int>> tile_vector;
    for (const auto &tile: group.tiles()) {
        std::vector<int> subtile_vec;
        subtile_vec.resize(existing_key_vals_size); // Needs to be changed if indata is changed, but can't seem to make it work right now
        auto subtile = std::to_string(tile.zoom()) + "_" + std::to_string(tile.x()) + "_" + std::to_string(tile.y());

        for (const auto &feat: tile.features()) {
            // Form explicit key-value tag pairs
            const auto &tags = feat.tags();
            for (size_t i = 0; i + 1 < tags.size(); i += 2) {
                auto key_in = tags[static_cast<int>(i)];
                auto val_in = tags[static_cast<int>(i) + 1];
                const auto &[key, val] = handle_tags(key_in, val_in);
                if (key.rfind("UNK", 0) != 0) {
                    ++subtile_vec.at(existing_key_vals[key][val]);
                    // printf("kv: %s, %s: %d\n",key.c_str(), val.c_str(), subtile_vec.at(existing_key_vals[key][val]));
                } else {
                    // printf("KV miss: %s, %s, %s\n", key_in.c_str(), val_in.c_str(), key.c_str());
                }
            }
        }
        // Add feature tags to the tile's tag set
        tile_vector[subtile] = subtile_vec;
    }

    std::stringstream local_tags_buffer;
    std::stringstream local_images_buffer;
    std::stringstream local_tags_images_buffer;
            for (auto &[subtile, tag_vector]: tile_vector) {
        local_tags_buffer << subtile << "," << labels[subtile] << ",";
        local_images_buffer << subtile << "," << labels[subtile] << ",";
        local_tags_images_buffer << subtile << "," << labels[subtile] << ",";

        local_images_buffer << "data/autogluon/images/" << subtile << ".png";
        local_tags_images_buffer << "data/autogluon/images/" << subtile << ".png,";

        // for (const auto &val: tag_vector) {
        //     local_tags_buffer << val << ",";
        //     local_tags_images_buffer << val << ",";
        // }
        for (int i{}; i < tag_vector.size()-1; ++i) {
            local_tags_buffer << tag_vector.at(i) << ",";
            local_tags_images_buffer << tag_vector.at(i) << ",";
        }
        // Last entries should not be followed by comma
        local_tags_buffer << tag_vector.at(tag_vector.size()-1);
        local_tags_images_buffer << tag_vector.at(tag_vector.size()-1);

        local_tags_buffer << std::endl;
        local_images_buffer << std::endl;
        local_tags_images_buffer << std::endl;
    }
    return {local_tags_buffer.str(), local_images_buffer.str(), local_tags_images_buffer.str()};
}

void add_to_buffer(const buffers &subtile_vector) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    tags_buffer << subtile_vector.tags;
    images_buffer << subtile_vector.images;
    tags_images_buffer << subtile_vector.tags_images;
}

occurrenceMap combine_maps(const std::vector<occurrenceMap> &vec_of_maps) {
    occurrenceMap result_map{};
    // Iterate over each map in the vector
    for (const auto &single_map: vec_of_maps) {
        // Iterate over each key-value pair in the current map
        for (const auto &[key, value]: single_map) {
            // Add the value to the result map, creating the key if it doesn't exist
            result_map[key] += value;
        }
    }
    return result_map;
}


