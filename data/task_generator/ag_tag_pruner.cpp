//
// Created by Ludvig Delvret on 2024-12-05.
//

#include "ag_tag_pruner.h"

#include <BS_thread_pool.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <cassert>

std::mutex buffer_mutex;
std::mutex buffer_map_mutex;

constexpr int NUM_TAGS = 12572;
keyval_map existing_key_vals;
std::map<std::string, std::vector<std::string>> key_vals;
std::vector<int> total_counts{};
std::map<std::string, int> total_counts_map;
std::stringstream tags_buffer;
std::stringstream images_buffer;
std::stringstream tags_images_buffer;

void add_to_sum(const std::vector<int> &tag_counts) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    for (size_t i = 0; i < total_counts.size(); i++) {
        total_counts[i] += tag_counts[i];
    }
}

void add_to_sum(const std::map<std::string, int> &tag_counts) {
    std::lock_guard<std::mutex> lock(buffer_map_mutex);
    for ( auto &[tag, value] : tag_counts) {
        total_counts_map[tag] += value;
    }
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    std::string input_directory;
    std::string output_directory;
    std::string csv_file;
    float p;
    int status = parse_args(input_directory, output_directory, csv_file, p, argc, argv);
    if (status != -1) return status;

    total_counts.resize(NUM_TAGS);


    std::vector<std::string> task_division{"train", "test", "val"};
    auto [fst, snd] = readCSV(csv_file);
    existing_key_vals = fst;
    key_vals = snd;
    //Print all key and val
    // for (auto &[key, vals] : key_vals) {
    //     if (key.rfind("railway", 0) == 0) {
    //         printf("KEY: %s\n", key.c_str());
    //         for (auto & val: vals) {
    //             printf("    VAL: %s\n", val.c_str());
    //         }
    //         printf("\n");
    //     }
    // }

    // Create the main output directory
    try {
        if (!fs::exists(output_directory) || !fs::is_directory(output_directory)) {
            fs::create_directories(output_directory);
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error handling directory: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    for (const auto &div: task_division) {
        auto div_input_directory = input_directory + "/" + div;

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
                option::MaxProgress{tile_groups.size()/100}
        };

        BS::thread_pool pool;
        std::atomic<size_t> num_processed{0};
        std::mutex progress_mutex;

        // Function to update progress bar
        auto update_progress = [&]() {
            size_t processed = num_processed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (processed % 100 == 0) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                progress_bar.set_option(option::PostfixText{
                        std::to_string(processed) + "/" + std::to_string(tile_groups.size())
                });
                progress_bar.tick();
            }
        };


        int progress = 0;

        bool parallel = true;
        for (const auto &group: tile_groups) {
            auto task = [&]() {
                try {
                    auto tag_counts_pair = count_occurrences(group);
                    add_to_sum(tag_counts_pair.first);
                    add_to_sum(tag_counts_pair.second);
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
    }

    gen_new_ag(output_directory, csv_file);

    print_results(output_directory);
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
int parse_args(std::string &input_directory, std::string &output_directory, std::string &csv_file, float &p, int argc, char *argv[]) {
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

std::pair<keyval_map,std::map<std::string, std::vector<std::string>>> readCSV(const std::string &filename) {
    keyval_map data;
    std::map<std::string, std::vector<std::string>> sparse_data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {data, sparse_data};
    }

    std::string key, value;
    char ch;
    bool parsingKey = true;
    int tag_id{1};

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
                const auto &[key_copy, value_copy] = setup_tags(key, value);
                data[key_copy][value_copy] = tag_id; // Add the value to the vector for this key
                sparse_data[key_copy].push_back(value_copy); // Add the value to the vector for this key
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
        data[key][value] = tag_id; // Add the value to the vector for this key
        sparse_data[key].push_back(value);
    }

    file.close();
    return {data, sparse_data};
}

std::pair<std::string, std::string> setup_tags(const std::string &key, const std::string &value) {
    auto key_copy = key;
    auto value_copy = value;
    std::replace(key_copy.begin(), key_copy.end(), '\n', ' ');
    std::replace(value_copy.begin(), value_copy.end(), '\n', ' ');
    return {key_copy, value_copy};
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
    auto print = key.find("address") != std::string::npos;
    print = false;
    if (print) printf("IN: %s = %s\n", key.c_str(), value.c_str());
    if (print) printf("key_vals size: %lu\n", existing_key_vals.size());

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

std::pair<std::vector<int>, std::map<std::string, int>> count_occurrences(const processed::TileGroup &group) {
    // Maps to store co-occurrence counts
    std::vector<int> tag_vector;
    std::map<std::string, int> tag_vector_map;
    tag_vector.resize(NUM_TAGS); // Needs to be changed if indata is changed, but can't seem to make it work right now

    for (const auto &tile: group.tiles()) {

        for (const auto &feat: tile.features()) {

            // Form explicit key-value tag pairs
            const auto &tags = feat.tags();
            for (size_t i = 0; i + 1 < tags.size(); i += 2) {
                const auto& key_in = tags[static_cast<int>(i)];
                const auto& val_in = tags[static_cast<int>(i) + 1];
                auto print = (key_in.rfind("railway:ptc", 0) == 0);
                print = false;
                const auto &[key, val] = handle_tags(key_in, val_in);
                ++tag_vector.at(existing_key_vals[key][val]);
                // if (key.rfind("UNK", 0) == 0) {
                //     printf("existing_key_vals[%s][%s]: %d\n", key.c_str(),val.c_str(),existing_key_vals[key][val]);
                // }
                if (print) printf("OUT: %s; %s\n", key.c_str(), val.c_str());
                // if (existing_key_vals[key][val] != 0) {
                    if (print) printf("nonexistant Key: %s, val: %s\n", key.c_str(), val.c_str());
                    tag_vector_map[key + " = " + val] += 1;
                // }
            }
        }
    }
    return {tag_vector, tag_vector_map};
}

void print_results(const std::string& output_directory) {
    std::stringstream out_buffer;
    out_buffer << "tag_id,count,tag" << std::endl;
    int total_sum = 0;
    int total_zeros = 0;
    int total_sub_twenties= 0;
    int total_sub_forties= 0;
    for (size_t i = 0; i < total_counts.size(); i++) {
        out_buffer << i << "," << total_counts[i] << std::endl;
        total_sum += total_counts[i];
        if (total_counts[i] <= 0) {
            total_zeros++;
            continue;;
        }
        if (total_counts[i] <= 20) {
            total_sub_twenties++;
            continue;
        }
        if (total_counts[i] <= 40) {
            total_sub_forties++;
        }
    }
    std::cout << "sorting map by occurrences" << std::endl;
    // Copy map to a vector of pairs
    std::vector<std::pair<std::string, int>> vec(total_counts_map.begin(), total_counts_map.end());

    // Sort vector by value
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    std::stringstream out_map_buffer;
    out_map_buffer << "tag_combo,count" << std::endl;
    for (auto &[key, value]: vec) {
        out_map_buffer << key << ": " << value << std::endl;
    }

    std::cout << "Total distinct unk tags: " << total_counts_map.size()/1000<<"k" << std::endl;
    std::cout << "Total unk tags: " << total_counts[0]/1000<<"k" << std::endl;
    std::cout << "Total sum: " << total_sum/1000<<"k" << std::endl;
    std::cout << "Total known tags: " << (total_sum-total_counts[0])/1000 << "k" << std::endl;
    std::cout << "Total size: " << total_counts.size()<< std::endl;
    std::cout << "Total zero counts: " << total_zeros << std::endl;
    std::cout << "Total sub_twenties counts: " << total_sub_twenties << std::endl;
    std::cout << "Total sub_forties counts: " << total_sub_forties << std::endl;

    std::ofstream os(output_directory + "/tag_counts.csv");
    if (os.is_open()) {
        os << out_buffer.str();
    } else {
        std::cerr << "Error opening file for writing.\n";
    }
    os.close();
    std::ofstream oms(output_directory + "/tag_counts_map.csv");
    if (oms.is_open()) {
        oms << out_map_buffer.str();
    } else {
        std::cerr << "Error opening file for writing.\n";
    }
    oms.close();
}

void gen_new_ag(const std::string& output_directory, const std::string &input_file) {
    for (int limit{}; limit < 100; limit += 10) {
        std::ifstream file(input_file); // Replace with your file path
        if (!file.is_open()) {
            std::cerr << "Failed to open input file " << input_file << " in gen_new_ag!" << std::endl;
            return;
        }

        std::stringstream pruned_buffer;
        pruned_buffer << "key;value;" << std::endl;
        std::string line;
        int index = 1;
        while (std::getline(file, line) && index < total_counts.size() -1) {
            // Perform actions based on the line's index
            if (total_counts.at(index) >= limit) {
                pruned_buffer << line << std::endl;
                // pruned_buffer << line << ":" << index << ":" << total_counts.at(index) << std::endl;
            }
            ++index;
        }
        file.close();
        std::ofstream ostream(output_directory + "/pruned_ag" + std::to_string(limit) + ".csv");
        if (ostream.is_open()) {
            ostream << pruned_buffer.str();
        } else {
            std::cerr << "Error opening file for writing.\n";
        }
        ostream.close();
    }

}
