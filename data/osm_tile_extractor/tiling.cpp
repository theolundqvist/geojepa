#include <osmium/io/reader.hpp>
#include <osmium/io/pbf_input.hpp> //REQUIRED
#include <osmium/io/pbf_output.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
// #include <osmium/handler.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/relations/relations_manager.hpp>
// #include <osmium/geom/mercator_projection.hpp>
#include <osmium/visitor.hpp>
#include <osmium/geom/wkt.hpp>  // For geometry in WKT format
#include <fstream>
#include <iostream>
#include <filesystem>
#include <set>

#include "schema.pb.h"
#include "utils.h"
#include "indicators.hpp"
#include "cxxopts.h"

namespace fs = std::filesystem;

#define MEASURE 1
#if MEASURE

#include <chrono>  // For timing

size_t node_count = 0;
size_t way_count = 0;
size_t relation_count = 0;
size_t way_span_multiple = 0;
size_t relation_span_multiple = 0;
size_t skipped_nodes = 0;
size_t first_pass_member_count = 0;
size_t first_pass_relation_count = 0;
size_t total_num_tiles = 0;
size_t total_num_tile_groups = 0;
size_t num_tiles = 0;
size_t num_tile_groups = 0;
size_t num_high_content_removed = 0;
size_t num_low_content_removed = 0;
double lat_min = 90;
double lat_max = -90;
double lon_min = 180;
double lon_max = -180;
#endif

class MyManager : public osmium::relations::RelationsManager<MyManager, true, true, true> {
    struct TileData {
        std::unordered_map<osmium::unsigned_object_id_type, int> osm_id_to_index;
        std::vector<Feature> features;
        std::vector<Group> groups;
    };

    std::map<utils::tileZXY, TileData> tile_map;
    std::map<osmium::unsigned_object_id_type, std::vector<utils::tileZXY> > osm_id_to_tile_ids;

    // std::vector<Group> groups;
    // std::vector<std::pair<size_t, size_t> > max_member_nodes{std::make_pair(0, 0)};
#if MEASURE
    int max_tile_index_x = 0;
    int max_tile_index_y = 0;
    int min_tile_index_x = std::numeric_limits<int>::max();
    int min_tile_index_y = std::numeric_limits<int>::max();
    std::function<void()> init_bars_callback;
    std::function<void()> tick_second_pass_callback;
    std::function<void()> tick_group_tile_callback;
    std::function<void()> tick_write_pbf_callback;
#endif

public:
    explicit MyManager() = default;

    MyManager(const std::function<void()> &init_bars, const std::function<void()> &tick_second_pass,
              const std::function<void()> &tick_group_pbf,
              const std::function<void()> &tick_write_pbf) : init_bars_callback(init_bars),
                                                             tick_second_pass_callback(tick_second_pass),
                                                             tick_group_tile_callback(tick_write_pbf),
                                                             tick_write_pbf_callback(tick_group_pbf) {
        std::cout << std::setprecision(10);
        //
        // const auto tile_zxy = utils::latLonToTile(59.38278182637225, 18.01063294952013);
        // const auto tile_bbox = utils::tileZXYToLatLonBBox(tile_zxy);
        // std::cout << tile_zxy.x << ", " << tile_zxy.y << std::endl;
        // std::cout << tile_bbox.lat1 << ", " << tile_bbox.lon1 << std::endl;
        // std::cout << tile_bbox.lat2 << ", " << tile_bbox.lon2 << std::endl;
        // exit(1);
    }

    static bool new_relation(const osmium::Relation &relation) noexcept {
#if MEASURE
        ++first_pass_relation_count;
        // if (first_pass_relation_count % 1000 == 0) {
        //     std::cout << "Processed " << first_pass_relation_count / 1000 << " k relations on first pass\n";
        // }
#endif
        return true;
    }

    static bool new_member(const osmium::Relation &relation, const osmium::RelationMember &member,
                           std::size_t) noexcept {
#if MEASURE
        ++first_pass_member_count;
        // if (first_pass_member_count % 10000 == 0) {
        //     std::cout << "Processed " << first_pass_member_count / 1000 << " k members on first pass\n";
        // }
#endif
        return true;
    }

    void check_tile_index(const utils::tileZXY &t) noexcept {
        const auto x = t.x;
        const auto y = t.y;

        if (y > max_tile_index_y) max_tile_index_y = y;
        if (x > max_tile_index_x) max_tile_index_x = x;
        if (y < min_tile_index_y) min_tile_index_y = y;
        if (x < min_tile_index_x) min_tile_index_x = x;
    }

    void after_node(const osmium::Node &node) {
#if MEASURE

        if (node_count == 0) {
            init_bars_callback();
        }

        if (node.location().lat() > lat_max) {
            lat_max = node.location().lat();
        }
        if (node.location().lat() < lat_min) {
            lat_min = node.location().lat();
        }
        if (node.location().lon() > lon_max) {
            lon_max = node.location().lon();
        }
        if (node.location().lon() < lon_min) {
            lon_min = node.location().lon();
        }

        node_count++;
        // if (node_count % 1000000 == 0) {
        //     std::cout << "Processed " << node_count / 1000000 << " m nodes\n";
        // }
#endif


        Feature poi{};
        for (const auto &tag: node.tags()) {
            if (utils::should_keep_tag(tag.key(), tag.value())) {
                poi.mutable_tags()->insert({tag.key(), tag.value()});
            }
        }
//        if (poi.tags().empty()) {
//#if MEASURE
//            skipped_nodes++;
//#endif
//            return;
//        }

        const auto lat = node.location().lat();
        const auto lon = node.location().lon();

        const auto geometry = poi.mutable_geometry();
        auto *p = geometry->add_points();
        p->set_lat(static_cast<float>(lat));
        p->set_lon(static_cast<float>(lon));
        geometry->set_inner(false);
        geometry->set_is_closed(false);


        try {
            const auto tile = utils::latLonToTile(lat, lon);
            check_tile_index(tile);
            auto &tile_data = tile_map[tile];
            tile_data.osm_id_to_index[node.id()] = static_cast<int>(tile_data.features.size());
            tile_data.features.push_back(poi);
            osm_id_to_tile_ids[node.id()].push_back(tile);
        } catch (const std::invalid_argument &e) {
            std::cout << "Lat: " << lat << " Lon: " << lon << "\n";
            std::cout << e.what() << "\n";
        }
    }

    void after_way(const osmium::Way &way) {
        if (!osm_id_to_tile_ids[way.id()].empty()) {
//            printf("Skipping After Way, already been called\n");
            return;
        }
#if MEASURE
        way_count++;
        // if (way_count % 10000 == 0) {
        //     std::cout << "Processed " << way_count / 1000 << " k ways\n";
        // }
#endif

        std::map<utils::tileZXY, std::vector<utils::point> > tiles_nodes;
        const bool is_polygon = way.is_closed();
        std::set<utils::tileZXY> tiles;
        std::vector<utils::point> points;
        auto print{way.id() == 1091376273};
        // if (!print) return;
        // if (print) printf("After way on id %lld\n", way.id());

        for (auto &node: way.nodes()) {
            try {
                const auto tile = utils::latLonToTile(node.lat(), node.lon());
                tiles.emplace(tile);
                points.emplace_back(node.lat(), node.lon());
                check_tile_index(tile);
            } catch (const std::invalid_argument &e) {
                std::cout << "Lat: " << node.lat() << " Lon: " << node.lon() << "\n";
                std::cout << e.what() << std::endl;
            }
        }
        // if (print) printf("After way on id %lld, size %ld\n", way.id(), tiles.size());
        osm_id_to_tile_ids[way.id()].reserve(tiles.size());
        for (auto &tile: tiles) {
            if (tiles.size() > 1) {
                auto point_vec = utils::intersection_area(points, tile, is_polygon);
                tiles_nodes.emplace(tile, point_vec);
            } else {
                tiles_nodes.emplace(tile, points);
            }
            osm_id_to_tile_ids[way.id()].push_back(tile);
        }
        // if (print) {
        //     for (const auto &tile: osm_id_to_tile_ids[way.id()]) {
        //         printf("way_id: %lld\n", way.id());
        //         printf("Tile %lu_%d_%d\n", tile.z, tile.x, tile.y);
        //     }
        // }

        //end intersection


        for (auto &[tile, nodes]: tiles_nodes) {
            Feature feature;
            for (const auto &tag: way.tags()) {
                if (utils::should_keep_tag(tag.key(), tag.value())) {
                    (*feature.mutable_tags())[tag.key()] = tag.value();
                }
            }
//            if (feature.tags().empty()) {
//                continue;
//            }
            auto geometry = feature.mutable_geometry();
            geometry->set_is_closed(is_polygon);

            for (const auto &node: nodes) {
                const auto p = geometry->add_points();
                p->set_lat(static_cast<float>(node.lat));
                p->set_lon(static_cast<float>(node.lon));
            }

            auto &data = tile_map[tile];
            data.osm_id_to_index[way.id()] = static_cast<int>(data.features.size());
            data.features.push_back(feature);
        }
#if MEASURE
        if (tiles_nodes.size() > 1) {
            way_span_multiple++;
        }
#endif
    }

    void complete_relation(const osmium::Relation &relation) {
        // for
        auto member_nodes = 0;
#if MEASURE
        relation_count++;
        if (relation_count % 100 == 0) tick_second_pass_callback();
        // if (relation_count % 1000 == 0) {
        //     std::cout << "Processed " << relation_count / 1000 << " k relations\n";
        // }
#endif

        std::map<utils::tileZXY, std::vector<osmium::unsigned_object_id_type>> tiles;

        // auto print{relation.id() == 14534282};
        // if (print) printf("Num members: %lu\n", relation.members().size());
        // if (print) {
        //     printf("way_id in relation: %d\n", 1091376273);
        //     for (const auto &tile: osm_id_to_tile_ids[1091376273]) {
        //         printf("Tile %lu_%d_%d\n", tile.z, tile.x, tile.y);
        //     }
        // }

        for (const auto &member: relation.members()) {
            // if (print) printf("Member: ref: %lld, type: %hd\n", member.ref(), member.type());


            if (member.ref() != 0) {
                if (const auto type = member.type(); type == osmium::item_type::node) {
#if MEASURE
                    member_nodes++;
#endif
                    const osmium::Node *node = this->get_member_node(member.ref());
                    bool pr = false;
                    if (node->id() == 7008004883) {
                        pr = true;
                        printf("node id %lld\n", node->id());
                        printf("empty: %b", osm_id_to_tile_ids[node->id()].empty());
                    }
                    if (osm_id_to_tile_ids[node->id()].empty()) {
                        if (pr) {
                            printf("After node for id %lld\n", node->id());
                        }
                        after_node(*(node));
                        if (osm_id_to_tile_ids[node->id()].empty()) {
                            printf("Could not find tile for node %lld\n", node->id());
                            continue;
                        }
                        auto t = osm_id_to_tile_ids[node->id()][0];
                        if (pr) printf("tile_id0: %ld_%d_%d", t.z, t.x, t.y);
                    }

                    try {
                        auto tile = utils::latLonToTile(node->location().lat(), node->location().lon());
                        // // print = true;
                        // if (print) printf("\nStarting node\n");
                        check_tile_index(tile);
                        tiles[tile].push_back(node->id());
                    } catch (std::invalid_argument &e) {
                        std::cout << "Lat: " << node->location().lat() << " Lon: " << node->location().lon() << "\n";
                        std::cout << e.what() << "\n";
                    }

//                    if (node->tags().empty()) {
//                        continue;
//                    }
                } else if (type == osmium::item_type::way) {
                    const osmium::Way *way = this->get_member_way(member.ref());
                    if (osm_id_to_tile_ids[way->id()].empty()) {
                        after_way(*(way));
                    }
                    // if (print) {
                    //     auto node = way->nodes().back();
                    //     auto tile = utils::latLonToTile((node.lat()), (node.lon()));
                    //     printf("\nStarting way %lld, tile %d_%d\n", member.ref() ,tile.x, tile.y);
                    // }
                    for (const auto &tile: osm_id_to_tile_ids[way->id()]) {
                        // if (tile.x == 18063 && tile.y == 25891) {
                        //     print = true;
                        //     if (print) printf("\nStarting way, tile %d_%d\n", tile.x, tile.y);
                        // }
                        // if (print) printf("\nStarting way %lld, tile %d_%d\n", member.ref() ,tile.x, tile.y);
                        // if (print) {
                        //     for (auto &tag: way->tags()) {
                        //         printf("tag: %s, value: %s\n", tag.key(), tag.value());
                        //     }
                        //     printf("_________\n");
                        // }
                        tiles[tile].push_back(way->id());
                    }
                } else if (type == osmium::item_type::relation) {
                    /*
                    * const osmium::Relation *rel = this->get_member_relation(member.ref());
                    * DO SOMETHING
                    */
                }
            } else {
                const osmium::Node *node = this->get_member_node(member.ref());
                std::cout << "Lat: " << node->location().lat() << " Lon: " << node->location().lon() << "\n";
            }
        }

        for (const auto &[tile, members]: tiles) {
            // if (tile.x == 18063 && tile.y == 25891) {
            //     // print = true;
            //     if (print) printf("\nStarting tile\n");
            // }
            auto tile_data = tile_map[tile];
            Group group{};
            for (const auto &tag: relation.tags()) {
                if (utils::should_keep_tag(tag.key(), tag.value())) {
                    group.mutable_tags()->insert({tag.key(), tag.value()});
                }
                // if (print) printf("tag: %s, val: %s\n", tag.key(), tag.value());
            }
            if (group.tags().empty()) {
                continue;
            }
//            if (tile.x == 18060 && tile.y == 25890) {
//                printf("Tile: %d_%d\n", tile.x, tile.y);
//                for (const auto a: tile_data.osm_id_to_index) {
//                    if(a.second == 0 || a.first == 7008004883 || a.second == 108){
//                        printf("\nosm: %llu -> %d", a.first, a.second);
//                    }
//                }
//                printf("\n");
//            }
            for (const auto id: members) {
                auto i = tile_data.osm_id_to_index[id];
//                if (tile.x == 18060 && tile.y == 25890) {
//                    printf("index: %lu -> %d\n", id, i);
//                }
                group.add_feature_indices(i);
                // if (print) printf("index: %d\n", static_cast<int>(id));
            }
            if (group.feature_indices().empty()) {
                continue;
            }
            // if (print) printf("\n______\n");
            tile_map[tile].groups.push_back(group);
        }


#if MEASURE
        if (tiles.size() > 1) {
            relation_span_multiple++;
        }
#endif
    }

    void remove_detached_features() {
        for (auto &[tile, tile_data]: tile_map) {
            auto to_remove = std::vector<int>();
            int id = 0;

            // find features without tags and in no group
            for (const auto &f: tile_data.features) {
                if (f.tags_size() == 0) {
                    bool in_any_group = false;
                    for (const auto &g: tile_data.groups) {
                        for (const auto &i: g.feature_indices()) {
                            if (i == id) {
                                in_any_group = true;
                                break;
                            }
                        }
                        if (in_any_group) break;
                    }
                    if (!in_any_group) {
                        to_remove.push_back(id);
                    }
                }
                ++id;
            }

            // remove features
            for (auto i = to_remove.rbegin(); i != to_remove.rend(); ++i) {
                // Adjust group indices
                for (auto &group: tile_data.groups) {
                    for (int fi = 0; fi < group.feature_indices_size(); ++fi) {
                        auto old_index = group.feature_indices(fi);
                        if (old_index > *i) {
                            group.set_feature_indices(fi, old_index - 1);
                        }
                    }
                }
                tile_map[tile].features.erase(tile_map[tile].features.begin() + *i);
            }
        }
    }

    void write_to_pbf(const std::string &directory) {
        const auto max_tile = utils::tileZXY(16, max_tile_index_x, max_tile_index_y);
        const auto min_tile = utils::tileZXY(16, min_tile_index_x, min_tile_index_y);
        auto max_group = utils::get_tile_group_coordinates(max_tile);
        auto min_group = utils::get_tile_group_coordinates(min_tile);

        auto max_x = max_group.x;
        auto min_x = min_group.x;
        auto max_y = max_group.y;
        auto min_y = min_group.y;
        auto width = max_x - min_x;
        auto height = max_y - min_y;

        std::vector<TileGroup> tile_groups;
        tile_groups.resize((abs(width) + 1) * (abs(height) + 1));
        // std::cout << max_tile.x << " " << max_tile.y << "\n";
        // std::cout << min_tile.x << " " << min_tile.y << "\n";
        // std::cout << max_x << " - " << min_x << "\n";
        // std::cout << max_y << " - " << min_y << "\n";
        // std::cout << width << " * " << height << "\n";

#if MEASURE
        tick_second_pass_callback();
        total_num_tiles = tile_map.size();
        init_bars_callback();
        num_tiles = 0;
        size_t num_features = 0;
        size_t num_groups = 0;
#endif
        for (const auto &[tile, tile_data]: tile_map) {
#if MEASURE
            ++num_tiles;
            if (num_tiles % 100 == 0) tick_group_tile_callback();
#endif
            auto tile_group_id = utils::get_tile_group_coordinates(tile);
            auto vec_id = abs((tile_group_id.x - min_x) * (height)) + abs(tile_group_id.y - min_y);
            if (vec_id >= tile_groups.size()) {
                printf("\n"
                       "vec_id out of bounds: %d "
                       "vec size: %lu "
                       "tile_group_id.x: %d, "
                       "min_x: %d, "
                       "height: %d, "
                       "tile_group_id.y: %d\n",
                       vec_id,
                       tile_groups.size(),
                       tile_group_id.x,
                       min_x,
                       height,
                       tile_group_id.y);
                continue;
            }
            auto &tg = tile_groups.at(vec_id);
            tg.set_x(tile_group_id.x);
            tg.set_y(tile_group_id.y);
            tg.set_zoom(static_cast<int>(tile_group_id.z));

            const auto feat_length = tile_data.features.size() + tile_data.groups.size();
            if (feat_length > 1250) {
#if MEASURE
                ++num_high_content_removed;
#endif
                continue;
            }
            if (feat_length < 5) {
#if MEASURE
                ++num_low_content_removed;
#endif
                continue;
            }
            auto t = Tile();
            t.set_x(tile.x);
            t.set_y(tile.y);
            t.set_zoom(static_cast<int>(tile.z));

            for (const auto &f: tile_data.features) {
#if MEASURE
                ++num_features;
#endif
                t.add_features()->CopyFrom(f);
            }
            for (const auto &g: tile_data.groups) {
#if MEASURE
                ++num_groups;
#endif
                t.add_groups()->CopyFrom(g);
            }
            tg.add_tiles()->CopyFrom(t);
        }
        // std::cout << "Starting to write" << std::endl;
#if MEASURE
        tick_group_tile_callback();
        total_num_tile_groups = tile_groups.size();
        init_bars_callback();
        auto empty_tiles{0};
#endif

        for (const auto &tile_group: tile_groups) {
#if MEASURE
            ++num_tile_groups;
            if (num_tile_groups % 100 == 0) tick_write_pbf_callback();
#endif
            if (tile_group.zoom() == 0) {
                ++empty_tiles;
                continue;
            }
            std::string file_name = directory + std::to_string(tile_group.zoom()) + "_" + std::to_string(tile_group.x())
                                    + "_" + std::to_string(tile_group.y()) + ".pbf";
            // std::cout << "file_name: " << file_name << std::endl;
            std::ofstream stream(file_name, std::ios::binary);
            // printf("\n");
            // printf("Writing %d tiles\n", tile_group.tiles().size());
            // if (num_tile_groups == 20 ) {
            //     exit(1);
            // }
            if (!tile_group.SerializeToOstream(&stream)) {
                std::cerr << "Failed to write tile." << std::endl;
                return;
            }
            stream.close();
        }
#if MEASURE
        tick_write_pbf_callback();
        std::cout << "num_tiles: " << num_tiles << "\n";
        std::cout << "num_features: " << num_features << "\n";
        std::cout << "num_groups: " << num_groups << "\n";
        std::cout << "empty tiles: " << empty_tiles << "\n";
#endif
    }

#if MEASURE

    void print_summary() const {
        // std::sort(max_member_nodes.begin(), max_member_nodes.end(), std::greater<>());

        // for (int i = 0; i < 10 && i < max_member_nodes.size(); ++i) {
        //     std::cout << "(" << max_member_nodes[i].first << ", " << max_member_nodes[i].second << ")\n";
        // }
        std::cout << "Summary:\n"
                  << "Nodes processed: " << node_count << "\n"
                  << "Nodes skipped: " << skipped_nodes << "\n"
                  << "Ways processed: " << way_count << "\n"
                  << "Ways with multiple spans: " << way_span_multiple << "\n"
                  << "Relations processed: " << relation_count << "\n"
                  << "Relations with multiple spans: " << relation_span_multiple << "\n"
                  << "Max lat: " << lat_max << ", min lat: " << lat_min << "\n"
                  << "max lon: " << lon_max << ", min lon; " << lon_min << "\n"
                  << "max tile index: " << max_tile_index_x << ", " << max_tile_index_y << "\n"
                  << "min tile index: " << min_tile_index_x << ", " << min_tile_index_y << "\n"
                  << "high content tiles removed: " << num_high_content_removed << "\n"
                  << "low content tiles removed: " << num_low_content_removed << "\n"
                  << std::endl;
        // exit(1);
    }

#endif
};


int main(int argc, char *argv[]) {
#if MEASURE
    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    using namespace indicators;


    show_console_cursor(false);
    BlockProgressBar second_pass_bar{
            option::BarWidth{20},
            option::Start{"["},
            option::End{"]"},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::ForegroundColor{Color::cyan},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    BlockProgressBar group_tiles_bar{
            option::BarWidth{20},
            option::Start{"["},
            option::End{"]"},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::ForegroundColor{Color::magenta},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    BlockProgressBar write_pbf_bar{
            option::BarWidth{20},
            option::Start{"["},
            option::End{"]"},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::ForegroundColor{Color::cyan},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    auto init_bars = [&]() {
        second_pass_bar.set_option(option::MaxProgress{first_pass_relation_count / 100});
        group_tiles_bar.set_option(option::MaxProgress{total_num_tiles / 100});
        write_pbf_bar.set_option(option::MaxProgress{total_num_tile_groups / 100});
    };
    auto tick_second_bar = [&]() {
        second_pass_bar.tick();
        second_pass_bar.set_option(option::PostfixText{
                std::to_string(relation_count) + "/" + std::to_string(first_pass_relation_count) + " second pass"
        });
        if (relation_count == first_pass_relation_count) {
            second_pass_bar.mark_as_completed();
        }
    };
    auto tick_group_bar = [&]() {
        group_tiles_bar.tick();
        group_tiles_bar.set_option(option::PostfixText{
                std::to_string(num_tiles) + "/" + std::to_string(total_num_tiles) + " tile groups"
        });
        if (num_tiles == total_num_tiles) {
            group_tiles_bar.mark_as_completed();
        }
    };
    auto tick_pbf_bar = [&]() {
        write_pbf_bar.tick();
        write_pbf_bar.set_option(option::PostfixText{
                std::to_string(num_tile_groups) + "/" + std::to_string(total_num_tile_groups)
        });
        if (num_tile_groups == total_num_tile_groups) {
            write_pbf_bar.mark_as_completed();
        }
    };


#endif

    cxxopts::Options options("osm_tile_extractor", "A program to extract tiles from osm.pbf files.");

    options.add_options()
            ("o,output-dir", "output directory for tiles", cxxopts::value<std::string>())
            ("i,input-file", "osm.pbf file", cxxopts::value<std::string>());

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (cxxopts::exceptions::exception &e) {
        std::cout << e.what() << std::endl;
        exit(1);
    }

    if (!result.count("o") || !result.count("i")) {
        std::cout << options.help() << std::endl;
        exit(1);
    }
    osmium::io::File input_file{result["i"].as<std::string>()};
    std::string output_directory = result["o"].as<std::string>() + "/";
    fs::create_directories(output_directory);

    osmium::io::Reader reader(input_file);

    namespace map = osmium::index::map;
    using index_type = map::SparseMemArray<osmium::unsigned_object_id_type, osmium::Location>;
    using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;


    index_type index;
    location_handler_type location_handler{index};

#if MEASURE
    MyManager relations_manager(init_bars, tick_second_bar, tick_pbf_bar, tick_group_bar); //Pass callbacks
#else
    MyManager relations_manager;
#endif

    osmium::relations::read_relations(input_file, relations_manager);
    osmium::apply(reader, location_handler, relations_manager.handler());
    // osmium::apply(reader, location_handler, relations_manager.handler(), handler);


#if MEASURE
    auto mid_time = std::chrono::high_resolution_clock::now();
    auto mid_duration = std::chrono::duration_cast<std::chrono::seconds>(mid_time - start_time).count();
    std::cout << "Execution time: " << mid_duration << " seconds\n";
#endif

    // relations_manager.print_to_stream(output_directory);
    relations_manager.remove_detached_features();
    relations_manager.write_to_pbf(output_directory);

    relations_manager.print_summary();
    // Close reader and CSV file
    reader.close();
    // output_file.close();


#if MEASURE
    // Stop time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    auto write_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - mid_time).count();

    // Print the summary and the execution time
    std::cout << "Write time: " << write_duration << " seconds\n";
    std::cout << "Total time: " << total_duration << " seconds\n";
    show_console_cursor(true);
#endif

    return 0;
}
