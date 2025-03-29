#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>      // std::setprecision
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <cmath>
#include <filesystem>
#include "BS_thread_pool.hpp"

#include "cxxopts.h"
#include "indicators.hpp"
#include "utils.h"
#include "proto/processed_tile_group.pb.h"
#include "proto/unprocessed_tile_group.pb.h"

typedef boost::geometry::model::d2::point_xy<double> xy_t;
typedef boost::geometry::model::linestring<xy_t> linestring_t;
typedef boost::geometry::model::segment<xy_t> segment_t;

using std::min, std::max;
namespace fs = std::filesystem;
using namespace indicators;

int empty_feats{0};
int non_empty_feats{0};

struct args_t {
    int max_connections;
    std::string output_dir;
    std::string input_dir;
};

struct node_t {
    int feature_id;
    int gid;
    double lat;
    double lon;
};

xy_t xy(const node_t &p) {
    return xy_t(p.lon, p.lat);
}

segment_t segment(const node_t &p1, const node_t &p2) {
    return boost::geometry::model::segment(xy(p1), xy(p2));
}

struct edge_t {
    int gid1;
    int gid2;
};

struct polygon_t {
    int polygon_id;
    std::vector<int> node_ids; // global_node_ids
};


struct ecef_t {
    double x;
    double y;
    double z;
};

template<typename T>
void check_nan(const T &value) {
    if (std::isnan(value)) {
        throw std::runtime_error("Error: Value is NaN.");
    }
}

template<typename Func, typename... Args>
auto time(Func &&func, std::atomic<long long> &counter, Args &&... args)
-> std::invoke_result_t<Func, Args...> {
    const auto start = std::chrono::high_resolution_clock::now();

    if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...> >) {
        std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        counter += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return;
    } else {
        auto result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        const auto end = std::chrono::high_resolution_clock::now();
        counter += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return result;
    }
}

template<typename Func, typename... Args>
auto time_and_print(Func &&func, std::string text, Args &&... args)
-> std::invoke_result_t<Func, Args...> {
    const auto start = std::chrono::high_resolution_clock::now();

    if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...> >) {
        std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        printf("time [%s]: %lld ms\n", text.c_str(),
               std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        return;
    } else {
        auto result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        const auto end = std::chrono::high_resolution_clock::now();
        printf("time [%s]: %lld ms\n", text.c_str(),
               std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        return result;
    }
}

ecef_t geodeticToECEF(const xy_t &geo) {
    double latRad = geo.y() * M_PI / 180.0;
    double lonRad = geo.x() * M_PI / 180.0;
    double a = 6378137.0; // WGS84 semi-major axis
    double f = 1.0 / 298.257223563; // WGS84 flattening
    double e_sq = f * (2 - f);

    double N = a / sqrt(1 - e_sq * sin(latRad) * sin(latRad));

    ecef_t ecef;
    ecef.x = (N) * cos(latRad) * cos(lonRad);
    ecef.y = (N) * cos(latRad) * sin(lonRad);
    ecef.z = ((1 - e_sq) * N) * sin(latRad);

    return ecef;
}

struct ENUCoord {
    double east;
    double north;
    double up;
};

ENUCoord ecefToENU(const ecef_t &target, const ecef_t &origin, const xy_t origin_latlon) {
    double dx = target.x - origin.x;
    double dy = target.y - origin.y;
    double dz = target.z - origin.z;
    double lon0 = origin_latlon.x() * M_PI / 180.0;
    double lat0 = origin_latlon.y() * M_PI / 180.0;

    ENUCoord enu{};
    enu.east = -sin(lon0) * dx + cos(lon0) * dy;
    enu.north = -sin(lat0) * cos(lon0) * dx - sin(lat0) * sin(lon0) * dy + cos(lat0) * dz;
    enu.up = cos(lat0) * cos(lon0) * dx + cos(lat0) * sin(lon0) * dy + sin(lat0) * dz;

    return enu;
}

ENUCoord pointToENU(const xy_t &target, const xy_t &origin) {
    ecef_t target_ecef = geodeticToECEF(target);
    ecef_t origin_ecef = geodeticToECEF(origin);
    return ecefToENU(target_ecef, origin_ecef, origin);
}


bool is_visible(const node_t &p1, const node_t &p2, const std::vector<std::pair<node_t, node_t> > &all_edges) {
    // Check if line segment between p1 and p2 intersects any existing edge
    for (const auto &[fst, snd]: all_edges) {
        const node_t &e1 = fst;
        const node_t &e2 = snd;

        if (p1.gid == e1.gid || p1.gid == e2.gid || p2.gid == e1.gid || p2.gid == e2.gid) continue;
        linestring_t intersection_points{};
        auto intersects = boost::geometry::intersection(segment(p1, p2), segment(e1, e2), intersection_points);
        if (intersects) {
            // ignore the intersection if it is 10cm from the endpoints
            for (const auto p: intersection_points) {
                auto dist = pow(p1.lat - p.y(), 2) + pow(p1.lon - p.x(), 2);
                auto dist2 = pow(p2.lat - p.y(), 2) + pow(p2.lon - p.x(), 2);
                if (std::min(dist, dist2) > 1e-12) {
                    return false;
                }
            }
        }
    }
    return true;
}


void calculate_visibility_edges(const std::vector<polygon_t> &polygons, const std::vector<node_t> &points,
                                std::vector<edge_t> &visibility_edges,
                                const args_t &args) {
    // Build list of all edges as point pairs
    std::vector<std::pair<node_t, node_t> > all_edges;
    // time_and_print([&] {
    for (const auto &polygon: polygons) {
        const auto &node_ids = polygon.node_ids;
        for (size_t i = 0; i < node_ids.size(); ++i) {
            int gid1 = node_ids[i];
            int gid2 = node_ids[(i + 1) % node_ids.size()]; // Wrap around to form a closed loop
            const node_t &p1 = points[gid1];
            const node_t &p2 = points[gid2];
            all_edges.emplace_back(p1, p2);
        }
    }
    // }, "all_edges");

    // compute distance between all nodes
    std::vector distance_mat(points.size(), std::vector(points.size(), 1e10));
    std::vector visibility_mat(points.size(), std::vector(points.size(), false));


    // time_and_print([&] {
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            const auto p1 = points[i];
            const auto p2 = points[j];
            if (p1.feature_id == p2.feature_id) continue;
            auto dist = pow(p1.lat - p2.lat, 2) + pow(p1.lon - p2.lon, 2);
            distance_mat[p1.gid][p2.gid] = dist;
            distance_mat[p2.gid][p1.gid] = dist;
            auto visible = is_visible(p1, p2, all_edges);
            visibility_mat[p1.gid][p2.gid] = visible;
            visibility_mat[p2.gid][p1.gid] = visible;
        }
    }
    // }, "distance + visibility");

    std::vector shortest_dist_mat(points.size(), std::vector(points.size(), 0));
    // time_and_print([&] {
    for (size_t i = 0; i < points.size(); ++i) {
        std::iota(shortest_dist_mat[i].begin(), shortest_dist_mat[i].end(), 0);
        std::sort(shortest_dist_mat[i].begin(), shortest_dist_mat[i].end(),
                  [&](const size_t a, const size_t b) {
                      return distance_mat[i][a] < distance_mat[i][b];
                  });
    }
    // }, "sort by dist");

    std::vector connection_mat(points.size(), std::vector(points.size(), false));
    std::vector connection_count(points.size(), 0);
    std::vector<int> rand_index(points.size());
    std::iota(rand_index.begin(), rand_index.end(), 0);
    // time_and_print([&] {
    for (int k = 0; k < args.max_connections; ++k) {
        std::shuffle(rand_index.begin(), rand_index.end(), std::mt19937(std::random_device()()));
        for (int i = 0; i < points.size(); ++i) {
            auto p1 = points[rand_index[i]];
            auto gid1 = p1.gid;
            if (connection_count[gid1] >= args.max_connections) continue;
            for (int j = 0; j < shortest_dist_mat[i].size(); ++j) {
                auto p2 = points[shortest_dist_mat[gid1][j]];
                auto gid2 = p2.gid;
                if (p1.feature_id == p2.feature_id || connection_mat[gid1][gid2] || !visibility_mat[gid1][gid2])
                    continue;
                if (connection_count[gid2] >= args.max_connections + 1) continue;
                connection_mat[gid1][gid2] = true;
                connection_mat[gid2][gid1] = true;
                connection_count[gid1]++;
                connection_count[gid2]++;
                visibility_edges.push_back({gid1, gid2});
                break; // max one new connection per node
            }
        }
    }
    // }, "calc");
}

void normalize(unprocessed::Tile *tile) {
    const auto tile_zxy = utils::tileZXY(tile->zoom(), tile->x(), tile->y());
    const auto bbox = utils::tileZXYToLatLonBBox(tile_zxy);
    const auto [min_lat, max_lat] = std::minmax(bbox.lat1, bbox.lat2);
    const auto [min_lon, max_lon] = std::minmax(bbox.lon1, bbox.lon2);
    const auto max_xy = xy_t{max_lon, max_lat};
    const auto min_xy = xy_t{min_lon, min_lat};

    const auto tproj = pointToENU(max_xy, min_xy);
    const auto tile_size = (std::abs(tproj.east) + std::abs(tproj.north)) / 2.0;
    for (int fid = 0; fid < tile->features_size(); ++fid) {
        auto feature = tile->mutable_features(fid);
        auto geo = feature->mutable_geometry();
        for (int i = 0; i < geo->points_size(); ++i) {
            auto point = geo->mutable_points(i);
            auto [east, north, up] = pointToENU({point->lon(), point->lat()}, min_xy);
            point->set_lat(static_cast<float>(north / tile_size));
            point->set_lon(static_cast<float>(east / tile_size));
        }
    }
}

int parse_args(args_t &args, const int argc, char *argv[]) {
    cxxopts::Options options("preprocess", "Processes polygons to simplify them and generate visibility edges");
    options.add_options()
            ("k,max-connections", "Maximum number of connections per node", cxxopts::value<int>()->default_value("1"))
            ("o,output-dir", "Output dir", cxxopts::value<std::string>()->default_value("preprocessed"))
            ("h,help", "Print help", cxxopts::value<bool>())
            ("i,input-dir", "The dir to process pbf files in", cxxopts::value<std::vector<std::string> >());
    options.parse_positional({"input-dir"});
    try {
        cxxopts::ParseResult result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return EXIT_SUCCESS;
        }
        args.max_connections = result["max-connections"].as<int>();
        args.input_dir = result["input-dir"].as<std::vector<std::string> >()[0];
        args.output_dir = result["output-dir"].as<std::string>();
    } catch (const cxxopts::exceptions::exception &x) {
        std::cerr << "preprocess: " << x.what() << '\n';
        std::cerr << "usage: preprocess [options] input_dir\n";
        return EXIT_FAILURE;
    }
    return -1;
}

/**
 * Add visibility edges to the processed tile,
 * Hell breaks lose here due to indexing correctness etc...
 */
int add_visibility_edges(const args_t &args, const unprocessed::Tile &tile, Tile *processed_tile) {
    if (tile.zoom() != processed_tile->zoom() || tile.x() != processed_tile->x() || tile.y() != processed_tile->y()) {
        printf("\nWarning: tile mismatch\n");
        return 0;
    }
    // processed tiles have [points, polylines, polygons, relations], i.e. relations are duplicated and we only want to connect to those duplicated features
    int relation_start_offset = -1;
    for (int i = 0; i < processed_tile->node_to_feature_size(); ++i) {
        if (processed_tile->features(processed_tile->node_to_feature(i)).is_relation()) {
            relation_start_offset = i;
            break;
        }
    }
    if (relation_start_offset == -1) {
        //printf("\nWarning: no relations in processed tile\n");
        return 0;
    }
    std::vector<std::vector<edge_t>> visibility_edges{};
    std::vector<int> group_offsets{};
    group_offsets.push_back(0);
    if (tile.features_size() == 0) {
        return 0;
    }
//    printf("\n\n\nTILE: %d, %d, %d\n", tile.zoom(), tile.x(), tile.y());
//    printf("tile_groups: %d\n", tile.groups_size());
//    printf("tile_features: %d, old: %d\n", processed_tile->features_size(), tile.features_size());
//    printf("tile_nodes: %d\n", processed_tile->nodes_size()/2);
    for (const auto &g: tile.groups()) {
//        if (group_offsets.size() == 3) {
//            break;
//        }
        std::vector<node_t> points;
        std::vector<polygon_t> polygons;
        int gid = 0;
        int polygon_id{};
        auto added = std::set<int>{};
//        printf("\nGroup\n");
        for (const auto fid: g.feature_indices()) {
//            printf("%d,", fid);
            if(added.find(fid) != added.end()) continue;
            if (tile.features_size() <= fid) {
                throw std::runtime_error("Error: feature index out of bounds " + std::to_string(fid) + " >= " +
                                         std::to_string(tile.features_size()));
            }
            auto geo = tile.features(fid).geometry();
            polygon_t pgon{polygon_id};
            for (const auto &p: geo.points()) {
                pgon.node_ids.push_back(gid);
                points.push_back({polygon_id, gid, p.lat(), p.lon()});
                ++gid;
            }
            ++polygon_id;
            polygons.push_back(pgon);
            added.insert(fid);
        }
        group_offsets.push_back(((int) points.size()) + group_offsets[group_offsets.size() - 1]);
//        for (const auto &t: g.tags()) {
//            printf("\nTag: %s -> %s", t.first.c_str(), t.second.c_str());
//        }
//        printf("\n");
//        printf("\n    members: %d\n", g.feature_indices_size());
//        printf("    polygons: %d\n", (int) polygons.size());
//        printf("    points: %d\n", (int) points.size());
        std::vector<edge_t> new_edges{};
        if (polygons.size() > 1) {
            calculate_visibility_edges(polygons, points, new_edges, args);
        }
        visibility_edges.push_back(new_edges);
    }
    for (int i = 0; i < visibility_edges.size(); ++i) {
        if (i >= group_offsets.size()) {
            printf("\nError: i: %d, extra_offset.size: %ld\n", i, group_offsets.size());
        }
        const auto group_offset = group_offsets[i];
        int lowest = INT_MAX;
        int highest = INT_MIN;
        for (auto const &[gid1, gid2]: visibility_edges[i]) {
            int id1 = gid1 + relation_start_offset + group_offset;
            int id2 = gid2 + relation_start_offset + group_offset;
            if (id1 < lowest) lowest = id1;
            if (id2 < lowest) lowest = id2;
            if (id1 > highest) highest = id1;
            if (id2 > highest) highest = id2;
            int fid1 = processed_tile->node_to_feature(id1);
            int fid2 = processed_tile->node_to_feature(id2);
            if(i == 3 && (fid1 == 282 || fid2 == 282)){
                printf("[%d], %d->%d, %d->%d, %d->%d\n", i, id1, id2, fid1, fid2, gid1, gid2);
            }
            processed_tile->add_intra_edges(id1);
            processed_tile->add_intra_edges(id2);
        }
        int fid1 = processed_tile->node_to_feature(lowest);
        int fid2 = processed_tile->node_to_feature(highest);
        if(fid1 != fid2 && lowest != INT_MAX) {
            printf("\nError in relation (%d): group_offset: (%d)\nlowest: %d -> (feat: %d), highest: %d -> (feat: %d)\n", i, group_offset, lowest, fid1, highest, fid2);
        }
    }
    int size = 0;
    for (auto &e: visibility_edges) {
        size += (int) e.size();
    }
    return size;
}

void read_file(const std::string &input_file, unprocessed::TileGroup &tile_group) {
    std::ifstream file{};
    file.open(input_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + input_file);
    }
    if (!tile_group.ParsePartialFromIstream(&file)) {
        throw std::runtime_error("Failed to parse tile group");
    }
    file.close();
}

void write_file(const std::string &output_file, TileGroup &tile_group) {
    std::ofstream file{};
    file.open(output_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + output_file);
    }
    if (!tile_group.SerializeToOstream(&file)) {
        throw std::runtime_error("Failed to write tile group pbf");
    }
    file.close();
}

void simplify_geometries(const args_t &args, unprocessed::Tile *tile) {
    for (int i = 0; i < tile->features_size(); ++i) {
        auto f = tile->mutable_features(i);
        if (!f->has_geometry()) continue;
        const auto geo = f->mutable_geometry();
        if (geo->points_size() <= 2) continue;
        boost::geometry::model::linestring<xy_t> points;
        for (const auto &p: geo->points()) {
            points.emplace_back(p.lon(), p.lat());
        }
        boost::geometry::model::linestring<xy_t> simplified;
        boost::geometry::simplify(points, simplified, 0.000003);
        geo->clear_points();
        for (const auto s: simplified) {
            const auto p = geo->add_points();
            p->set_lat(static_cast<float>(s.y()));
            p->set_lon(static_cast<float>(s.x()));
        }
    }
};

void add_geometry(const unprocessed::Geometry &old, Tile *tile) {
    int n_points = old.points_size();
    // GEOMETRY
    int node_index = tile->nodes_size() / 2; // nbr of added nodes in this tile
    int feature_index = tile->features_size() - 1;
    for (int pi = 0; pi < n_points; ++pi) {
        const auto &p = old.points(pi);
        // NODES
        tile->add_nodes(p.lat());
        tile->add_nodes(p.lon());
        tile->add_node_to_feature(feature_index);

        // EDGES
        if (pi < n_points - 1) {
            tile->add_inter_edges(node_index + pi);
            tile->add_inter_edges(node_index + pi + 1);
        }
    }
    // CLOSE POLYGON
    if (old.is_closed()) {
        tile->add_inter_edges(node_index + n_points - 1);
        tile->add_inter_edges(node_index);
    }
}

Feature *add_feature(const google::protobuf::Map<std::string, std::string> &tags, Tile *tile, bool is_group = false) {
    // TAGS
    std::vector<std::string> wanted_tags{};
    wanted_tags.reserve(tags.size() * 2);
    std::vector<std::string> unwanted_prefix{"tiger:", "Tiger:", "source", "import", "yh:", "created_by"};
    bool has_tags = !tags.empty();
    for (const auto &[k, v]: tags) {
        // Add checks to remove unwanted tags like source, wiki, "tiger:", time created
        bool unwanted = false;
        for (const auto &prefix: unwanted_prefix) {
            if (k.rfind(prefix, 0) == 0) {
                unwanted = true;
                // printf("Unwanted tag found: %s: %s: %s\n", prefix.c_str(), k.c_str(), v.c_str());
                break;
            }
        }
        if (!unwanted) {
            wanted_tags.emplace_back(k);
            wanted_tags.emplace_back(v);
        }
    }
    if (tags.empty() && !is_group && false) {
        ++empty_feats;
        return nullptr;
    } else {
        auto feat = tile->add_features();
        feat->mutable_tags()->Reserve(static_cast<int>(wanted_tags.size()));
        for (auto &tag: wanted_tags) {
            std::replace(tag.begin(), tag.end(), ' ', '_');
            feat->add_tags(tag);
        }
        ++non_empty_feats;
        return feat;
    }
}

void transform_geometries_to_graph(const unprocessed::TileGroup &unprocessed_tile_group, TileGroup &tile_group) {
    tile_group.set_zoom(unprocessed_tile_group.zoom());
    tile_group.set_x(unprocessed_tile_group.x());
    tile_group.set_y(unprocessed_tile_group.y());
    for (int i = 0; i < unprocessed_tile_group.tiles_size(); ++i) {
        const auto tile = tile_group.add_tiles();
        const auto &old = unprocessed_tile_group.tiles(i);
        tile->set_zoom(old.zoom());
        tile->set_x(old.x());
        tile->set_y(old.y());

        for (const auto &f: old.features()) {
            if (f.geometry().points_size() == 1) {
                if (const auto feat = add_feature(f.tags(), tile)) {
//                    auto x = 18060;
//                    auto y = 25890;
//                    if (tile->x() == x && tile->y() == y) {
//                        printf("Found one point at %f, %f\n", f.geometry().points(0).lat(),
//                               f.geometry().points(0).lon());
//                    }
                    add_geometry(f.geometry(), tile);
                    feat->set_is_point(true);
                }
            }
        }
        for (const auto &f: old.features()) {
            if (f.geometry().points_size() > 1 && !f.geometry().is_closed()) {
                if (const auto feat = add_feature(f.tags(), tile)) {
                    add_geometry(f.geometry(), tile);
                    feat->set_is_polyline(true);
                }
            }
        }
        for (const auto &f: old.features()) {
            if (f.geometry().points_size() > 1 && f.geometry().is_closed()) {
                if (const auto feat = add_feature(f.tags(), tile)) {
                    add_geometry(f.geometry(), tile);
                    feat->set_is_polygon(true);
                }
            }
        }
//        if (tile->features_size() != old.features_size()) {
//            printf("\nWarning: feature size mismatch: %d != %d\n", tile->features_size(), old.features_size());
//        }
        //printf("\n\nold.groups_size(): %d\n", old.groups_size());

        for (const auto &g: old.groups()) {
            if (const auto feat = add_feature(g.tags(), tile, true)) {
                auto added = std::set<int>{};
                feat->set_is_relation(true);
                //printf("features: %d\n", g.feature_indices_size());
                for (auto fid: g.feature_indices()) {
                    if(added.find(fid) != added.end()) continue;
                    added.insert(fid);
                    //printf("    Adding relation feature: %d\n", fid);
                    if (fid >= old.features_size()) {
                        printf("\nWarning: feature index out of bounds: %d\n", fid);
                        return;
                    }
                    const auto &old_feat = old.features(fid);
                    add_geometry(old_feat.geometry(), tile);
                }
                //printf("total_features: %d\n", tile->features_size());
            }
        }

        for (int j = 0; j < tile->inter_edges_size(); j+=2) {
            auto id1 = tile->inter_edges(j);
            auto id2 = tile->inter_edges(j+1);
            auto fid1 = tile->node_to_feature(id1);
            auto fid2 = tile->node_to_feature(id2);
            if (fid1 != fid2) {
                throw std::runtime_error("Error: inter edge between different features");
            }
        }
//        if (tile->features_size() != old.features_size() + old.groups_size()) {
//            printf("\nWarning: feature size mismatch: %d != %d + %d\n", tile->features_size(), old.features_size(),
//                   old.groups_size());
//        }
    }
}

void add_min_box(const std::vector<utils::point> &points, Feature *feat) {
    const auto res = utils::min_box(points, 18, 1.5 / 300.0); // 10 deg, min_side_length = 1.5m
    for (const auto &p: res.points) {
        check_nan(p.lat);
        check_nan(p.lon);
        feat->add_min_box(static_cast<float>(p.lat));
        feat->add_min_box(static_cast<float>(p.lon));
    }
    check_nan(res.area);
    check_nan(res.width);
    check_nan(res.height);
    check_nan(res.rotation);
    feat->set_area(res.area);
    feat->set_width(res.width);
    feat->set_height(res.height);
    feat->set_rotation(res.rotation);
}


// translate, rotate and scale points to normalize in min_box local coordinate system. Uniform scaling
void create_min_box_normalized_coordinates(std::vector<utils::point> &points, Tile *tile, Feature *feat) {
    for (auto p: points) {
        p.lat -= feat->min_box(0); // min lat box coordinate
        p.lon -= feat->min_box(1);
        check_nan(p.lat);
        check_nan(p.lon);
        auto rot = utils::rotated_point(p, -feat->rotation());
        check_nan(rot.lat);
        check_nan(rot.lon);
        auto scale_factor = 1.0 / std::max(feat->height(), feat->width());
        check_nan(scale_factor);
        rot.lat *= scale_factor;
        rot.lon *= scale_factor;
        rot.lat = std::clamp(rot.lat, 0.0, 1.0);
        rot.lon = std::clamp(rot.lon, 0.0, 1.0);
        tile->add_local_coords(static_cast<float>(rot.lat));
        tile->add_local_coords(static_cast<float>(rot.lon));
    }
//    for (int i = 0; i < std::min(tile->local_coords_size(), 12); i+=2) {
//        auto lat = tile->local_coords(i);
//        auto lon = tile->local_coords(i+1);
//        printf("lat: %f, lon: %f\n", lat, lon);
//    }
//    printf("\n");

}

void calculate_min_boxes(TileGroup &new_tiles) {
    for (int tid = 0; tid < new_tiles.tiles_size(); ++tid) {
        auto tile = new_tiles.mutable_tiles(tid);
        std::vector<utils::point> point_list;
        //point_list.reserve(100);
        auto current_fid = 0;
        for (int nid = 0; nid < tile->node_to_feature_size(); ++nid) {
            const auto fid = tile->node_to_feature(nid);
            if (fid != current_fid || nid == tile->node_to_feature_size() - 1) {
                if (nid == tile->node_to_feature_size() - 1) {
                    auto lat = tile->nodes(nid * 2);
                    auto lon = tile->nodes(nid * 2 + 1);
                    check_nan(lat);
                    check_nan(lon);
                    point_list.emplace_back(lat, lon);
                }
                auto feat = tile->mutable_features(current_fid);
                add_min_box(point_list, feat);
                // translate, rotate and scale points to normalize in min_box local coordinate system. Uniform scaling
                create_min_box_normalized_coordinates(point_list, tile, feat);
                // cleanup
                current_fid = fid;
                point_list.clear();
            }
            auto lat = tile->nodes(nid * 2);
            auto lon = tile->nodes(nid * 2 + 1);
            check_nan(lat);
            check_nan(lon);
            point_list.emplace_back(lat, lon);
        }
    }
}

int main(int argc, char *argv[]) {
    args_t args{};
    int status = parse_args(args, argc, argv);
    if (status != -1) return status;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << std::setprecision(10);
    fs::create_directories(args.output_dir);
    fs::recursive_directory_iterator it(args.input_dir);
    std::vector<fs::directory_entry> files;

    for (const auto &file: it) {
        auto name = file.path().filename().string();
        //if (files.size() > 10 && name.find("8990_4818") == std::string::npos) continue;
        files.push_back(file);
    }

    BlockProgressBar bar{
            option::BarWidth{80},
            option::FontStyles{
                    std::vector{FontStyle::bold}
            },
            option::MaxProgress{files.size()},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
    };

    // Iterate over the map and process files with at least two images
    std::atomic<long> total{0};
    std::atomic<long> total_tiles{0};
    std::atomic<long long> total_read_time_ns{0};
    std::atomic<long long> total_simplify_time_ns{0};
    std::atomic<long long> total_add_visibility_time_ns{0};
    std::atomic<long long> total_normalize_time_ns{0};
    std::atomic<long long> total_write_time_ns{0};
    std::atomic<long long> total_tick_time_ns{0};
    std::atomic<int> total_visibility_edges{0};
    std::atomic<long long> total_convert_pbf_time{0};
    std::atomic<long long> total_min_boxes_time{0};
    std::atomic<long long> total_local_coords_time{0};
    std::atomic<long long> total_too_large_pruned{0};
    std::atomic<long long> total_too_small_pruned{0};
    std::mutex mutex{};
    const auto tick = [&]() {
        ++total;
        // Show iteration as postfix text
        mutex.lock();
        bar.set_option(option::PostfixText{
                std::to_string(total) + "/" + std::to_string(files.size())
        });
        bar.tick();
        mutex.unlock();
    };
    BS::thread_pool pool;
    bool parallel = true;
    for (const auto &file: files) {
        auto a = [&, file, args] {
            if (file.is_directory()) return tick();
            unprocessed::TileGroup unprocessed_tile_group;
            TileGroup tile_group;

            // Time the read_file operation
            time([&] {
                read_file(file.path(), unprocessed_tile_group);
            }, total_read_time_ns);

            // REMOVE ALL TILES WITH MORE THAN 1500 FEATURES FOR GEOJEPA MEMORY LIMITS, (its only 15 out of 290k)
	    total_tiles += unprocessed_tile_group.tiles_size();
            auto tiles = unprocessed_tile_group.mutable_tiles();
            for (auto it = tiles->begin(); it != tiles->end();) {
                const auto &tile = *it;
//                if (tile.features_size() > 1250) { # cant do this here, removed from pretraining but not masked datasets, result in crash
//                    ++total_too_large_pruned;
//                    it = tiles->erase(it);
//                } else if (tile.features_size() < 5) {
                if (tile.features_size() < 5) {
                    ++total_too_small_pruned;
                    it = tiles->erase(it);
                } else {
                    ++it;
                }
            }

            // Time the simplify_geometries operation
            time([&] {
                for (int i = 0; i < unprocessed_tile_group.tiles_size(); ++i) {
                    simplify_geometries(args, unprocessed_tile_group.mutable_tiles(i));
                }
            }, total_simplify_time_ns);

            // Time the normalize operation
            time([&] {
                for (int i = 0; i < unprocessed_tile_group.tiles_size(); ++i) {
                    normalize(unprocessed_tile_group.mutable_tiles(i));
                }
            }, total_normalize_time_ns);

            time([&] {
                transform_geometries_to_graph(unprocessed_tile_group, tile_group);
            }, total_convert_pbf_time);

            time([&] {
                calculate_min_boxes(tile_group);
            }, total_min_boxes_time);


            // Time the add_visibility_edges operation and accumulate edges
            total_visibility_edges += time([&]() -> int {
                int local_edges = 0;
                for (int i = 0; i < unprocessed_tile_group.tiles_size(); ++i) {
//                    auto x = 18063;
//                    auto y = 25889;
//                    auto tile = unprocessed_tile_group.tiles(i);
//                    if (tile.x() == x && tile.y() == y) {
//printf("found tile %d %d\n", x, y);
                        local_edges += add_visibility_edges(args, unprocessed_tile_group.tiles(i),
                                                            tile_group.mutable_tiles(i));
//                    }
                }
                return local_edges;
            }, total_add_visibility_time_ns);


            // Time the write_file operation
            time([&] {
                fs::path output_path{args.output_dir};
                write_file(output_path / file.path().filename(), tile_group);
            }, total_write_time_ns);

            // Update progress
            time([&] {
                tick();
            }, total_tick_time_ns);
        };
        if (parallel) {
            auto _future = pool.submit_task(a);
        } else {
            a();
        }
    }
    pool.wait();

    bar.mark_as_completed();

// After processing, display the collected statistics
    auto ns_to_ms = [](long long ns) -> double {
        return static_cast<double>(ns) / 1'000'000.0;
    };


    printf("\nProcessing Complete!\n");
    printf("Total files processed: %d\n", total.load());
    printf("Total tiles processed: %d\n", total_tiles.load());
    printf("Total visibility edges constructed: %d\n", total_visibility_edges.load());
    printf("Total read time: %.2f ms\n", ns_to_ms(total_read_time_ns.load()));
    printf("Total simplify geometries time: %.2f ms\n", ns_to_ms(total_simplify_time_ns.load()));
    printf("Total normalize time: %.2f ms\n", ns_to_ms(total_normalize_time_ns.load()));
    printf("Total convert pbf format time: %.2f ms\n", ns_to_ms(total_convert_pbf_time.load()));
    printf("Total min boxes time: %.2f ms\n", ns_to_ms(total_min_boxes_time.load()));
    printf("Total add visibility edges time: %.2f ms\n", ns_to_ms(total_add_visibility_time_ns.load()));
    printf("Total write time: %.2f ms\n", ns_to_ms(total_write_time_ns.load()));
    printf("Total too many feature tiles pruned: %lld\n", total_too_large_pruned.load());
    printf("Total too few feature tiles pruned: %lld\n", total_too_small_pruned.load());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\nWall clock time: " << duration.count() << " ms" << std::endl;
    return 0;
}

