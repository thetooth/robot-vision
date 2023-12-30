#include "nc.hpp"

void NC::to_json(json &j, const Pose &p)
{
    j = json{
        {"x", p.x},
        {"y", p.y},
        {"z", p.z},
        {"r", p.r},
        {"alpha", p.alpha},
        {"beta", p.beta},
        {"phi", p.phi},
        {"theta", p.theta},
        {"alphaVelocity", p.alphaVelocity},
        {"betaVelocity", p.betaVelocity},
        {"phiVelocity", p.phiVelocity},
        {"thetaVelocity", p.thetaVelocity},
    };
}
void NC::from_json(const json &j, Pose &p)
{
    p.x = j.value("x", 0.0);
    p.y = j.value("y", 0.0);
    p.z = j.value("z", 0.0);
    p.r = j.value("r", 0.0);

    p.alpha = j.value("alpha", 0.0);
    p.beta = j.value("beta", 0.0);
    p.phi = j.value("phi", 0.0);
    p.theta = j.value("theta", 0.0);
}
