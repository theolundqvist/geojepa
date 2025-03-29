// main.cpp
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <regex>
#include <algorithm>
#include <mutex>
#include <atomic>

#include "indicators.hpp"
#include "BS_thread_pool.hpp"

// Include image processing libraries
#include <webp/encode.h>
#include <webp/decode.h>
#include <png.h>

namespace fs = std::filesystem;

uintmax_t get_directory_size(const fs::path& dir_path) {
    uintmax_t total_size = 0;

    if (!fs::exists(dir_path)) {
        std::cerr << "Directory does not exist: " << dir_path << std::endl;
        return total_size;
    }

    if (!fs::is_directory(dir_path)) {
        std::cerr << "Path is not a directory: " << dir_path << std::endl;
        return total_size;
    }

    try {
        for (const auto& entry : fs::recursive_directory_iterator(dir_path, fs::directory_options::skip_permission_denied)) {
            if (fs::is_regular_file(entry.status())) {
                total_size += entry.file_size();
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing " << dir_path << ": " << e.what() << std::endl;
    }

    return total_size;
}

std::string format_size(uintmax_t size_in_bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_index = 0;
    double size = static_cast<double>(size_in_bytes);

    while (size >= 1024 && suffix_index < 4) {
        size /= 1024;
        ++suffix_index;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << suffixes[suffix_index];
    return oss.str();
}

bool LoadWebP(const std::string &filename, std::vector<uint8_t> &image, int &width, int &height) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open WebP file: " << filename << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading WebP file: " << filename << std::endl;
        return false;
    }

    uint8_t *decoded = WebPDecodeRGBA(reinterpret_cast<const uint8_t *>(buffer.data()), size, &width, &height);
    if (decoded == nullptr) {
        std::cerr << "WebPDecodeRGBA failed for file: " << filename << std::endl;
        return false;
    }

    image.assign(decoded, decoded + (width * height * 4));

    WebPFree(decoded);

    return true;
}

bool SavePNG(const std::string &filename, const std::vector<uint8_t> &image, int width, int height) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "Cannot open file for writing PNG: " << filename << std::endl;
        return false;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        std::cerr << "Failed to create PNG write struct." << std::endl;
        fclose(fp);
        return false;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Failed to create PNG info struct." << std::endl;
        png_destroy_write_struct(&png_ptr, nullptr);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "PNG error during writing." << std::endl;
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(
            png_ptr,
            info_ptr,
            width, height,
            8, // bit depth
            PNG_COLOR_TYPE_RGBA,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png_ptr, info_ptr);

    // Write image data
    png_bytep row = new png_byte[4 * width];
    for(int y = 0; y < height; y++) {
        memcpy(row, &image[y * width * 4], 4 * width);
        png_write_row(png_ptr, row);
    }
    delete[] row;

    // End write
    png_write_end(png_ptr, nullptr);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return true;
}

// Function to parse tile coordinates from filename
bool parse_filename(const std::string& filename, int& z, int& x, int& y) {
    std::regex pattern(R"((\d+)_(\d+)_(\d+)\.webp)");
    std::smatch matches;
    if (std::regex_search(filename, matches, pattern)) {
        if (matches.size() == 4) {
            z = std::stoi(matches[1].str());
            x = std::stoi(matches[2].str());
            y = std::stoi(matches[3].str());
            return true;
        }
    }
    return false;
}

// Function to compute sub-tile coordinates based on zoom levels
std::vector<std::pair<int, int>> get_sub_tiles(int x, int y, int current_z = 14, int sub_z = 16) {
    std::vector<std::pair<int, int>> new_coords;
    int zoom_diff = sub_z - current_z;
    int factor = std::pow(2, zoom_diff); // 2^2 = 4 for zoom_diff=2
    int sub_x = x * factor;
    int sub_y = y * factor;

    for (int i = 0; i < factor; ++i) {
        for (int j = 0; j < factor; ++j) {
            new_coords.emplace_back(sub_x + i, sub_y + j);
        }
    }
    return new_coords;
}

// Function to split the image into 16 tiles
std::vector<std::vector<uint8_t>> split_image(const std::vector<uint8_t> &image, int width, int height, int sub_z = 16) {
    // Assuming current_z = 14, so factor = 4 (since 2^(16-14) = 4)
    int factor = std::pow(2, sub_z - 14); // 4 for z=16 and current_z=14
    int tile_width = width / factor;
    int tile_height = height / factor;

    std::vector<std::vector<uint8_t>> tiles;

    for(int row = 0; row < factor; ++row) {
        for(int col = 0; col < factor; ++col) {
            std::vector<uint8_t> tile(tile_width * tile_height * 4, 0);
            for(int y = 0; y < tile_height; ++y) {
                for(int x = 0; x < tile_width; ++x) {
                    int src_idx = ((row * tile_height + y) * width + (col * tile_width + x)) * 4;
                    int dest_idx = (y * tile_width + x) * 4;
                    tile[dest_idx + 0] = image[src_idx + 0];
                    tile[dest_idx + 1] = image[src_idx + 1];
                    tile[dest_idx + 2] = image[src_idx + 2];
                    tile[dest_idx + 3] = image[src_idx + 3];
                }
            }
            tiles.emplace_back(std::move(tile));
        }
    }

    return tiles;
}

class ProgressBarWrapper {
public:
    ProgressBarWrapper(size_t total) : total_(total), current_(0) {
        bar.set_option(indicators::option::BarWidth{50});
        bar.set_option(indicators::option::MaxProgress{static_cast<int>(total_)});
        bar.set_option(indicators::option::ForegroundColor{indicators::Color::green});
        bar.set_option(indicators::option::ShowElapsedTime{true});
        bar.set_option(indicators::option::ShowRemainingTime{true});
    }

    void update() {
        std::lock_guard<std::mutex> lock(mutex_);
        bar.tick();
    }

    void finish() {
        bar.mark_as_completed();
    }

private:
    size_t total_;
    std::atomic<size_t> current_;
    std::mutex mutex_;
    indicators::ProgressBar bar{indicators::option::BarWidth{50}};
};

int main(int argc, char *argv[]) {
    // Check if input and output directories are provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_folder> <output_folder>" << std::endl;
        return 1;
    }

    std::string inputDir = argv[1];
    std::string outputDir = argv[2];

    fs::path input_path(inputDir);
    fs::path output_path(outputDir);

    // Check if input directory exists
    if (!fs::exists(input_path) || !fs::is_directory(input_path)) {
        std::cerr << "Input directory does not exist or is not a directory." << std::endl;
        return 1;
    }

    // Create output directory if it doesn't exist
    if (!fs::exists(output_path)) {
        if (!fs::create_directories(output_path)) {
            std::cerr << "Failed to create output directory." << std::endl;
            return 1;
        }
    }

    // Enumerate WebP images in the input directory matching the pattern 14_x_y.webp
    std::vector<fs::path> image_files;
    std::regex pattern(R"(^14_\d+_\d+\.webp$)");
    for (const auto& entry : fs::directory_iterator(input_path)) {
        if (entry.is_regular_file()) {
            std::string fname = entry.path().filename().string();
            if (std::regex_match(fname, pattern)) {
                image_files.emplace_back(entry.path());
            }
        }
    }

    if (image_files.empty()) {
        std::cerr << "No matching WebP images found in the input directory." << std::endl;
        return 1;
    }

    // Initialize progress bar
    ProgressBarWrapper progress_bar(image_files.size());

    // Initialize thread pool (assuming BS_thread_pool.hpp provides BS::thread_pool)
    BS::thread_pool pool(std::thread::hardware_concurrency());

    // Mutex for thread-safe operations (if needed)
    std::mutex mutex;

    // Lambda function to process each image
    auto process_image = [&](const fs::path& image_path) {
        std::string filename = image_path.filename().string();
        int z, x, y;
        if (!parse_filename(filename, z, x, y)) {
            std::cerr << "Filename does not match pattern: " << filename << std::endl;
            progress_bar.update();
            return;
        }

        if (z != 14) {
            std::cerr << "Skipping non-zoom-14 tile: " << filename << std::endl;
            progress_bar.update();
            return;
        }

        // Load WebP image
        std::vector<uint8_t> image;
        int width, height;
        if (!LoadWebP(image_path.string(), image, width, height)) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            progress_bar.update();
            return;
        }

        // Split image into 16 tiles
        std::vector<std::vector<uint8_t>> tiles = split_image(image, width, height, 16);

        // Compute sub-tile coordinates
        std::vector<std::pair<int, int>> sub_tiles = get_sub_tiles(x, y, z, 16);

        // Ensure that we have 16 tiles
        if (tiles.size() != 16 || sub_tiles.size() != 16) {
            std::cerr << "Error splitting image into 16 tiles: " << filename << std::endl;
            progress_bar.update();
            return;
        }

        // Iterate over each tile and save as PNG
        for(size_t i = 0; i < tiles.size(); ++i) {
            int sub_x = sub_tiles[i].first;
            int sub_y = sub_tiles[i].second;

            // Generate output filename: 16_x2_y2.png
            std::ostringstream oss;
            oss << "16_" << sub_x << "_" << sub_y << ".png";
            fs::path output_file = output_path / oss.str();

            // Ensure the output directory exists
            fs::create_directories(output_file.parent_path());

            // Save PNG
            if (!SavePNG(output_file.string(), tiles[i], width / 4, height / 4)) {
                std::cerr << "Failed to save PNG: " << output_file << std::endl;
            }
        }

        // Update progress bar
        progress_bar.update();
    };

    // Submit tasks to the thread pool
    for (const auto& image_path : image_files) {
        pool.submit_task([&, image_path]() {
            process_image(image_path);
        });
    }

    // Wait for all tasks to complete
    pool.wait();

    // Finish progress bar
    progress_bar.finish();

    std::cout << "Image splitting completed. Split images are saved in: " << outputDir << std::endl;
    uintmax_t input_size = get_directory_size(input_path);
    uintmax_t output_size = get_directory_size(output_path);

    std::cout << "Input Directory Size: " << format_size(input_size) << std::endl;
    std::cout << "Output Directory Size: " << format_size(output_size) << std::endl;

    return 0;
}