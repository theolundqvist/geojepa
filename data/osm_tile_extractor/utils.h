//
// Created by Ludvig Delvret on 2024-09-26.
//

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <boost/geometry.hpp>

namespace bg = boost::geometry;

struct utils {
    typedef bg::model::d2::point_xy<double> xy_t;
    typedef bg::model::linestring<xy_t> linestring_t;
    typedef bg::model::segment<xy_t> segment_t;
    typedef bg::model::polygon<xy_t> polygon_t;
    static constexpr float EPSILON = 1e-6;

    struct point {
        double lat;
        double lon;

        point() : lat(0), lon(0) {}

        point(const double _lat, const double _lon) : lat(_lat), lon(_lon) {}

        explicit point(const xy_t point) : lat(point.y()), lon(point.x()) {}

        explicit operator bool() const {
            constexpr double eps = std::numeric_limits<double>::epsilon();
            return !(std::abs(lat) < 0 + eps && std::abs(lon) < 0 + eps);
        }

        point operator+(const point &other) const {
            return {lat + other.lat, lon + other.lon};
        }

        point operator-(const point &other) const {
            return {lat - other.lat, lon - other.lon};
        }

        point operator/(const double val) const {
            return {lat / val, lon / val};
        }
    };

    static xy_t xy(const point &p) {
        return xy_t{p.lon, p.lat};
    }

    static segment_t segment(const point &p1, const point &p2) {
        return bg::model::segment{xy(p1), xy(p2)};
    }

    static linestring_t linestring(const std::vector<point> &points) {
        linestring_t linestring;
        for (const auto &p: points) {
            bg::append(linestring, xy(p));
        }
        return linestring;
    }

    static polygon_t polygon(const std::vector<point> &points) {
        polygon_t polygon;
        for (const auto &p: points) {
            bg::append(polygon.outer(), xy(p));
        }
        return polygon;
    }

    struct tileBBox {
        double lat1;
        double lon1;
        double lat2;
        double lon2;
    };

    struct tileZXY {

        tileZXY(const size_t z, const int x, const int y) : z(z), x(x), y(y) {}

        size_t z;
        int x;
        int y;

        bool operator==(const tileZXY &other) const {
            return (x == other.x && y == other.y && z == other.z);
        }

        bool operator!=(const tileZXY &other) const {
            return !(*this == other);
        }

        bool operator<(const tileZXY &other) const {
            if (z < other.z) {
                throw std::out_of_range("ERROR: different zoom levels");
            }
            if (x != other.x) {
                return x < other.x;
            }
            return y < other.y;
        }
    };

    struct bbox_edges {
        std::pair<point, point> north;
        std::pair<point, point> west;
        std::pair<point, point> south;
        std::pair<point, point> east;
    };


    static tileZXY latLonToTile(double lat, const double lon, const unsigned int z = 16) {
        constexpr unsigned int MAX_ZOOM_LEVEL = 22;
        constexpr double MIN_LAT = -85.051128779807;
        constexpr double MAX_LAT = 85.051128779806;
        constexpr double MIN_LON = -180.0;
        constexpr double MAX_LON = 180.0;

        if (z > MAX_ZOOM_LEVEL) {
            throw std::invalid_argument("Zoom level value is out of range [0, " +
                                        std::to_string(MAX_ZOOM_LEVEL) + "]");
        }

        if (!std::isfinite(lat) || (lat < MIN_LAT) || (lat > MAX_LAT)) {
            throw std::invalid_argument("Latitude value is out of range [" +
                                        std::to_string(MIN_LAT) + ", " + std::to_string(MAX_LAT) + "]");
        }

        if (!std::isfinite(lon) || (lon < MIN_LON) || (lon > MAX_LON)) {
            throw std::invalid_argument("Longitude value is out of range [" +
                                        std::to_string(MIN_LON) + ", " + std::to_string(MAX_LON) + "]");
        }

        const int xyTilesCount = 1 << z;
        int x = floor((lon + 180.0) / 360.0 * xyTilesCount);
        int y = floor((1.0 - log(tan(lat * M_PI / 180.0) + 1.0 / cos(lat * M_PI / 180.0)) / M_PI) /
                      2.0 * xyTilesCount);

        // return std::to_string(zoom) + "/" + std::to_string(x) + "/" + std::to_string(y);
        return {z, x, y};
    }

    static point tileZXYToLatLon(const tileZXY &tile) {
        const auto z = tile.z;
        const auto x = tile.x;
        const auto y = tile.y;

        constexpr unsigned int MAX_ZOOM_LEVEL = 22;
        if (z > MAX_ZOOM_LEVEL) {
            throw std::invalid_argument("Zoom level value is out of range [0, " +
                                        std::to_string(MAX_ZOOM_LEVEL) + "]");
        }

        const long maxXY = (1 << z) - 1;
        if (x > maxXY) {
            throw std::invalid_argument("Tile x value " + std::to_string(x) + " is out of range [0, " +
                                        std::to_string(maxXY) + "]");
        }

        if (y > maxXY) {
            throw std::invalid_argument("Tile y value is out of range [0, " +
                                        std::to_string(maxXY) + "]");
        }

        const int xyTilesCount = 1 << z;
        const double lon = static_cast<double>(x) / xyTilesCount * 360.0 - 180.0;

        const double n = M_PI - 2.0 * M_PI * static_cast<double>(y) / xyTilesCount;
        const double lat = 180.0 / M_PI * atan(0.5 * (exp(n) - exp(-n)));

        return {lat, lon};
    }

    static tileBBox tileZXYToLatLonBBox(const tileZXY &tile) {
        const auto z = tile.z;
        const auto x = tile.x;
        const auto y = tile.y;

        constexpr unsigned int MAX_ZOOM_LEVEL = 22;
        if (z > MAX_ZOOM_LEVEL) {
            throw std::invalid_argument("Zoom level value is out of range [0, " +
                                        std::to_string(MAX_ZOOM_LEVEL) + "]");
        }

        const long maxXY = (1 << z) - 1;
        if (x > maxXY) {
            throw std::invalid_argument("Tile x value is out of range [0, " +
                                        std::to_string(maxXY) + "]");
        }

        if (y > maxXY) {
            throw std::invalid_argument("Tile y value is out of range [0, " +
                                        std::to_string(maxXY) + "]");
        }

        const int xyTilesCount = 1 << z;
        const double lon1 = static_cast<double>(x) / xyTilesCount * 360.0 - 180.0;

        const double n1 = M_PI - 2.0 * M_PI * static_cast<double>(y) / xyTilesCount;
        const double lat1 = 180.0 / M_PI * atan(0.5 * (exp(n1) - exp(-n1)));

        const double lon2 = static_cast<double>(x + 1) / xyTilesCount * 360.0 - 180.0;

        const double n2 = M_PI - 2.0 * M_PI * static_cast<double>(y + 1) / xyTilesCount;
        const double lat2 = 180.0 / M_PI * atan(0.5 * (exp(n2) - exp(-n2)));

        return {lat1, lon1, lat2, lon2};
    }

    static tileZXY get_tile_group_coordinates(const tileZXY &tile) {
        return {14, tile.x / 4, tile.y / 4};
    }

    static bbox_edges bbox_edgesFromBBox(const tileBBox &bbox) {
        const auto lat1 = bbox.lat1;
        const auto lat2 = bbox.lat2;
        const auto lon1 = bbox.lon1;
        const auto lon2 = bbox.lon2;
        const point sw{lat1, lon1};
        const point se{lat1, lon2};
        const point nw{lat2, lon1};
        const point ne{lat2, lon2};
        const auto south = std::make_pair(sw, se);
        const auto west = std::make_pair(sw, nw);
        const auto east = std::make_pair(se, ne);
        const auto north = std::make_pair(nw, ne);
        return {north, west, south, east};
    }

    static point intersection(const tileBBox &bbox, const point p1, const point p2) {
        const auto way = segment(p1, p2);
        auto [north, west, south, east] = bbox_edgesFromBBox(bbox);
        std::vector<std::pair<point, point>> edges = {north, west, south, east};
        for (auto &[x, y]: edges) {
            linestring_t intersection_points;
            auto geom = segment(x, y);
            bg::intersection(geom, way, intersection_points);
            if (!intersection_points.empty()) {
                if (intersection_points.size() != 1) {
                    throw std::invalid_argument("Intersection points must contain only one");
                }
                const auto lat = intersection_points.front().y();
                const auto lon = intersection_points.front().x();
                return point{lat, lon};
            }
        }
        return point{0, 0};
    }

    static point cornerpoint(const tileBBox &bbox, const point entry, const point exit) {
        point corner = {0, 0};

        if (std::abs(entry.lat - exit.lat) < EPSILON || std::abs(entry.lon - exit.lon) < EPSILON) {
            return corner;
        }
        if (std::abs(entry.lat - bbox.lat1) < EPSILON || std::abs(entry.lat - bbox.lat2) < EPSILON) {
            corner.lat = entry.lat;
        } else {
            if (std::abs(exit.lat - bbox.lat1) < EPSILON) {
                corner.lat = bbox.lat2;
            } else {
                corner.lat = bbox.lat1;
            }
        }
        if (std::abs(entry.lon - bbox.lon1) < EPSILON || std::abs(entry.lon - bbox.lon2) < EPSILON) {
            corner.lon = entry.lon;
        } else {
            if (std::abs(exit.lon - bbox.lon1) < EPSILON) {
                corner.lon = bbox.lon2;
            } else {
                corner.lon = bbox.lon1;
            }
        }
        return corner;
    }

    static std::vector<point>
    intersection_area(const std::vector<point> &nodes, const tileZXY &tile, const bool is_polygon) {
        const auto [lat1, lon1, lat2, lon2] = tileZXYToLatLonBBox(tile);
        const std::vector<point> tile_bbox{{lat1, lon1},
                                           {lat1, lon2},
                                           {lat2, lon2},
                                           {lat2, lon1},
                                           {lat1, lon1}};

        std::vector<point> result;
        const polygon_t bbox_polygon = polygon(tile_bbox);
        if (is_polygon) {
            polygon_t way_polygon = polygon(nodes);
            bg::correct(way_polygon);
            std::deque<polygon_t> new_geom;
            bg::intersection(way_polygon, bbox_polygon, new_geom);
            for (auto &poly: new_geom) {
                std::vector<point> point_vector;
                for (auto &pt: poly.outer()) {
                    point res{pt.y(), pt.x()};
                    result.push_back(res);
                }
            }

        } else {
            std::deque<linestring_t> new_geom;
            auto way_linestring = linestring(nodes);
            bg::correct(way_linestring);
            bg::intersection(bbox_polygon, way_linestring, new_geom);
            for (auto &g: new_geom) {
                for (auto &p: g) {
                    result.emplace_back(p.y(), p.x());
                }
            }
        }
        return result;
    }

    static double bbox_area(const bg::model::box<xy_t> &bbox) {
        const auto max = point(bbox.max_corner());
        const auto min = point(bbox.min_corner());
        return (max.lat - min.lat) * (max.lon - min.lon);
    }

    static point rotated_point(const point &p, const double radians) {
        double new_x = p.lat * std::cos(radians) - p.lon * std::sin(radians);
        double new_y = p.lat * std::sin(radians) + p.lon * std::cos(radians);
        return {new_x, new_y};
    }

    static std::vector<point> min_box(const std::vector<point> &nodes, const int segments) {
        const auto node_polygon = polygon(nodes);
        // std::cout << std::setprecision(9) << std::endl;

        // Remove internal points
        polygon_t hull{};
        bg::convex_hull(node_polygon, hull);

        // get angle-aligned bounding box
        bg::model::box<xy_t> envelope_box{};
        bg::envelope(hull, envelope_box);
        // Calculate center-point
        const point center_point = (point(envelope_box.min_corner()) + point(envelope_box.max_corner())) / 2;

        // normalize convex_hull between -1 & 1
        std::vector<point> normalized_hull;
        for (const auto &pt: hull.outer()) {
            normalized_hull.push_back(point(pt) - center_point);
        }

        // Create candidates rotated by pi/segments radians
        std::vector<std::pair<std::vector<point>, double>> candidates;
        for (int n = 0; n < segments; n++) {
            std::vector<point> candidate;
            const auto angle = boost::geometry::math::pi<double>() / n;
            candidate.reserve(normalized_hull.size());
            for (auto &pt: normalized_hull) {
                candidate.push_back(rotated_point(pt, angle));
            }
            candidates.emplace_back(candidate, -angle);
        }

        boost::geometry::model::box<xy_t> smallest{};
        double smallest_area = std::numeric_limits<int>::max();
        double min_angle = 0;

        for (const auto &[points, angle]: candidates) {
            auto poly = polygon(points);
            boost::geometry::model::box<xy_t> bbox{};
            bg::envelope(poly, bbox);
            if (const auto area = bg::area(bbox); area < smallest_area) {
                smallest = bbox;
                smallest_area = area;
                min_angle = angle;
            }
            // const auto min_corner = rotated_point(point(bbox.min_corner()), angle) + center_point;
            // const auto max_corner = rotated_point(point(bbox.max_corner()), angle) + center_point;
            // std::cout << min_corner.lat << " " << min_corner.lon << std::endl;
            // std::cout << min_corner.lat << " " <<  max_corner.lon << std::endl;
            // std::cout << max_corner.lat << " " << max_corner.lon << std::endl;
            // std::cout << max_corner.lat << " " <<  min_corner.lon << std::endl;
            // std::cout << min_corner.lat << " " << min_corner.lon << std::endl;
            // std::cout << "END" << std::endl;
        }

        const auto min_corner = point(smallest.min_corner());
        const auto max_corner = point(smallest.max_corner());

        return {
                rotated_point(min_corner, min_angle) + center_point,
                rotated_point({min_corner.lat, max_corner.lon}, min_angle) + center_point,
                rotated_point(max_corner, min_angle) + center_point,
                rotated_point({max_corner.lat, min_corner.lon}, min_angle) + center_point,
                // rotated_point(min_corner, min_angle) + center_point
        };
    }

    static bool should_keep_tag(std::string key, std::string val) {
        // TAGS
        std::vector<std::string> unwanted_prefix{"tiger:", "Tiger:", "source", "import", "yh:", "created_by"};
        for (const auto &prefix: unwanted_prefix) {
            if (key.rfind(prefix, 0) == 0)
                return false;
        }
        return true;
//        for (auto it = tags->begin(); it != tags->end();) {
//            const auto &tag = *it;
//            auto unwanted = false;
//            for (const auto &prefix: unwanted_prefix) {
//                if (tag.first.rfind(prefix, 0) == 0) {
//                    it = tags->erase(it);
//                    unwanted = true;
//                    break;
//                }
//            }
//            if (!unwanted) {
//                ++it;
//            }
//        }
    }
};

#endif //UTILS_H
