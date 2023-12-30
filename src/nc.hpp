#ifndef NC_HPP
#define NC_HPP
#define JSON_HAS_RANGES 0

#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"

namespace NC
{
    using json = nlohmann::json;
    struct Pose
    {
        double x, y, z, r;
        double alpha, beta, phi, theta;
        double alphaVelocity, betaVelocity;
        double phiVelocity, thetaVelocity;
    };
    void to_json(json &j, const Pose &p);
    void from_json(const json &j, Pose &p);
} // namespace NC

#endif // NC_HPP
