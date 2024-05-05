#include "Falcor.h"
using namespace Falcor;

inline float sqr(float value)
{
    return value * value;
}

float3 pow(float3 value, float power)
{
    return float3(pow(value.x, power), pow(value.y, power), pow(value.z, power));

}
