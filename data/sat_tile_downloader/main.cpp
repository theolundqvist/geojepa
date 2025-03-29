#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include "indicators.hpp"
#include "BS_thread_pool.hpp"

// Include image processing libraries
#include <webp/encode.h>
#include <webp/decode.h>

namespace fs = std::filesystem;

// Maximum number of images to merge per relative path
constexpr size_t MAX_IMAGES_PER_PATH = 4;

// Function to load a WebP image into memory
bool LoadWebP(const std::string &filename, std::vector<uint8_t> &image, int &width, int &height) {
    // Open the file
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open WebP file: " << filename << std::endl;
        return false;
    }

    // Get the file size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the file into a buffer
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading WebP file: " << filename << std::endl;
        return false;
    }

    // Decode the WebP image
    uint8_t *decoded = WebPDecodeRGBA(reinterpret_cast<const uint8_t *>(buffer.data()), size, &width, &height);
    if (decoded == nullptr) {
        std::cerr << "WebPDecodeRGBA failed for file: " << filename << std::endl;
        return false;
    }

    // Copy the decoded data into the image vector
    image.assign(decoded, decoded + (width * height * 4));

    // Free the decoded image
    WebPFree(decoded);

    return true;
}

// Function to save an image as WebP
bool SaveWebP(const std::string &filename, const std::vector<uint8_t> &image, int width, int height) {
    uint8_t *output = nullptr;
    size_t output_size = WebPEncodeLosslessRGBA(image.data(), width, height, width * 4, &output);
    if (output_size == 0) {
        std::cerr << "WebPEncodeRGBA failed for file: " << filename << std::endl;
        return false;
    }

    // Write the encoded WebP data to a file
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Cannot open output WebP file: " << filename << std::endl;
        WebPFree(output);
        return false;
    }

    ofs.write(reinterpret_cast<char *>(output), output_size);
    WebPFree(output);

    return true;
}

bool FilterImage(const std::vector<uint8_t> &image, int width, int height) {
    auto pixels = width * height;

    int count_black = 0;
    int count_transparent = 0;
    for (size_t i = 0; i < pixels * 4; i += 4) {
        // Extract RGBA components
        const auto R = image[i];
        const auto G = image[i + 1];
        const auto B = image[i + 2];
        const auto A = image[i + 3];

        // Accumulate counts using implicit boolean to integer conversion
        count_transparent += static_cast<int>(A < 250);
        count_black += static_cast<int>((A >= 250) && (R < 5) && (G < 5) && (B < 5));
    }
    if (count_black == 0 && count_transparent == 0) {
        return false;
    }
    const float black_ratio = static_cast<float>(count_black) / static_cast<float>(pixels);
    const float transparent_ratio = static_cast<float>(count_transparent) / static_cast<float>(pixels);
    return black_ratio > 0.25 || transparent_ratio > 0.15;
}

// Function to merge multiple images
void MergeImages(const std::vector<std::vector<uint8_t> > &images, std::vector<uint8_t> &result, int width,
                 int height) {
    // Initialize result with transparency
    auto pixels = width * height;
    result.assign(pixels * 4, 0);

    for (const auto &image: images) {
        for (size_t i = 0; i < pixels * 4; i += 4) {
            uint8_t alpha = image[i + 3];
            double intensity = (image[i] + image[i + 1] + image[i + 2]) / 3.0 * alpha / 255.0;
            uint8_t alpha_res = result[i + 3];
            double intensity_res = (result[i] + result[i + 1] + result[i + 2]) / 3.0 * alpha_res / 255.0;

            if (intensity > intensity_res) {
                result[i + 0] = image[i + 0];
                result[i + 1] = image[i + 1];
                result[i + 2] = image[i + 2];
                result[i + 3] = image[i + 3];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Check if a directory is provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir> (--filter-bad)" << std::endl;
        return 1;
    }
    std::string inputDir = argv[1];
    std::string mergedDir = argv[2];
    bool filterBad = argc > 3 && std::string(argv[3]) == "--filter-bad";

    std::vector<std::string> inputDirs;
    const auto start = std::chrono::high_resolution_clock::now();
    // Assuming the directory structure is flat with subdirectories
    for (const auto &entry: std::filesystem::directory_iterator(inputDir)) {
        if (entry.is_directory()) {
            inputDirs.emplace_back(entry.path().string());
        }
    }

    if (inputDirs.empty()) {
        std::cerr << "No input directories found in " << inputDir << std::endl;
        return 1;
    }


    // Map to store relative paths and their corresponding image paths from different directories
    std::unordered_map<std::string, std::vector<fs::path> > pathMap;

    // Traverse all input directories and populate the map
    for (const auto &dir: inputDirs) {
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            std::cerr << "Invalid directory: " << dir << ". Skipping." << std::endl;
            continue;
        }

        for (const auto &entry: fs::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".webp") {
                std::string relativePath = fs::relative(entry.path(), dir).string();

                // Normalize path separators for consistency
                std::replace(relativePath.begin(), relativePath.end(), '\\', '/');

                // Insert the path into the map if under the limit
                auto &vec = pathMap[relativePath];
                if (vec.size() < MAX_IMAGES_PER_PATH) {
                    vec.emplace_back(entry.path());
                }
                // Optionally, warn if more than MAX_IMAGES_PER_PATH images exist
                else {
                    std::cerr << "Maximum images per path reached for: " << relativePath <<
                            ". Additional images are ignored." << std::endl;
                }
            }
        }
    }
    indicators::show_console_cursor(false);
    using namespace indicators;
    size_t totalToMerge = pathMap.size();
    BlockProgressBar bar{
        option::BarWidth{80},
        option::ForegroundColor{Color::grey},
        option::FontStyles{
            std::vector<FontStyle>{FontStyle::bold}
        },
        option::MaxProgress{totalToMerge},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
    };

    // Iterate over the map and process files with at least two images
    std::atomic total = 0;
    std::atomic n_merged = 0;
    std::atomic n_input_merged = 0;
    std::atomic n_copied = 0;
    std::atomic n_failed = 0;
    std::atomic n_filtered = 0;
    std::atomic n_input_images = 0;
    std::mutex mutex{};
    const auto tick = [&]() {
        ++total;
        // Show iteration as postfix text
        mutex.lock();
        bar.set_option(option::PostfixText{
            std::to_string(total) + "/" + std::to_string(totalToMerge)
        });
        bar.tick();
        mutex.unlock();
    };
    BS::thread_pool pool;
    for (const auto &[relativePath, imagePaths]: pathMap) {
        std::future<void> future = pool.submit_task([&, imagePaths, relativePath]() {
            n_input_images += static_cast<int>(imagePaths.size());
            if (imagePaths.size() < 2) {
                // Not enough images to merge
                if (filterBad) {
                    std::vector<uint8_t> image;
                    int width, height;
                    if (!LoadWebP(imagePaths[0].string(), image, width, height)) {
                        std::cerr << "Failed to load image: " << imagePaths[0].string() << ". Skipping this set." <<
                                std::endl;
                        ++n_failed;
                        return;
                    }
                    if (FilterImage(image, width, height)) {
                        tick();
                        ++n_filtered;
                        return;
                    }
                }
                //copy img to new dir
                // Prepare output path
                fs::path outputPath = fs::path(mergedDir) / relativePath;
                outputPath.replace_extension(".webp");

                // Create directories if they do not exist
                fs::create_directories(outputPath.parent_path());
                fs::copy_file(imagePaths[0], outputPath, fs::copy_options::skip_existing);

                // Save merged image
                tick();
                ++n_copied;
                return;
            }

            // Load all images
            std::vector<std::vector<uint8_t> > loadedImages;
            std::vector<int> widths, heights;
            bool loadSuccess = true;

            for (const auto &path: imagePaths) {
                std::vector<uint8_t> image;
                int width, height;
                if (!LoadWebP(path.string(), image, width, height)) {
                    std::cerr << "Failed to load image: " << path << ". Skipping this set." << std::endl;
                    loadSuccess = false;
                    break;
                }
                loadedImages.emplace_back(std::move(image));
                widths.emplace_back(width);
                heights.emplace_back(height);
            }

            if (!loadSuccess) {
                tick();
                ++n_failed;
                return;
            }

            // Check if all dimensions match
            bool dimensionsMatch = true;
            int firstWidth = widths[0];
            int firstHeight = heights[0];
            for (size_t i = 1; i < widths.size(); ++i) {
                if (widths[i] != firstWidth || heights[i] != firstHeight) {
                    dimensionsMatch = false;
                    break;
                }
            }

            if (!dimensionsMatch) {
                std::cerr << "Image dimensions do not match for: " << relativePath << ". Skipping." << std::endl;
                tick();
                ++n_failed;
                return;
            }

            // Merge images
            std::vector<uint8_t> mergedImage;
            MergeImages(loadedImages, mergedImage, firstWidth, firstHeight);
            if (filterBad && FilterImage(mergedImage, firstWidth, firstHeight)) {
                tick();
                ++n_filtered;
                return;
            }

            // Prepare output path
            fs::path outputPath = fs::path(mergedDir) / relativePath;
            outputPath.replace_extension(".webp");

            // Create directories if they do not exist
            create_directories(outputPath.parent_path());

            // Save merged image
            if (!SaveWebP(outputPath.string(), mergedImage, firstWidth, firstHeight)) {
                std::cerr << "Failed to save merged image: " << outputPath << std::endl;
                tick();
                ++n_failed;
                return;
            }
            tick();
            ++n_merged;
            n_input_merged += static_cast<int>(imagePaths.size());
        });
    }
    pool.wait();
    bar.mark_as_completed();
    indicators::show_console_cursor(true);

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cout << "Merge completed. " << std::endl;
    std::cout << "# input images: " << n_input_images << std::endl;
    std::cout << "# merged images: " << n_input_merged << " -> " << n_merged << std::endl;
    std::cout << "# copied: " << n_copied << std::endl;
    std::cout << "# failed: " << n_failed << std::endl;
    std::cout << "# filtered out: " << n_filtered << std::endl;
    std::cout << "# total images: " << n_input_images << " -> " << n_copied + n_merged - n_failed - n_filtered <<
            std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
