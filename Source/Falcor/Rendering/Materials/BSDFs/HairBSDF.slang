/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;
__exported import Rendering.Materials.IBSDF;

/**
 * Lambertian diffuse transmission.
 */

#define INTEGRATOR_NUM_SAMPLE 140
#define AZIMUTHAL_PRECOMPUTE_RESOLUTION 64
#define GAUSSIAN_DETECTOR_SAMPLES 2048

//// Convert radians to degrees
inline float radToDeg(float value) { return value * (180.0f / M_PI); }

/// Convert degrees to radians
inline float degToRad(float value) { return value * (M_PI / 180.0f); }

inline float gaussian_function(float b, float t)
{
    return exp(-t * t / (2 * b * b)) / sqrt(2.0f * M_PI) / b;
}

#define INV_FOURPI 0.07957747154594766788f

inline float squareToUniformSpherePdf() { return INV_FOURPI; }

float3 squareToUniformSphere<S : ISampleGenerator>(inout S sg) {
    float2 sample = sampleNext2D(sg);
    float z = 1.0f - 2.0f * sample.y;
    float r = sqrt(1.0f - z * z); // safe sqrt todo
    float sinPhi, cosPhi;
    sincos(2.0f * M_PI * sample.x, sinPhi, cosPhi);
    return float3(r * cosPhi, r * sinPhi, z);
}

struct GaussLegendre
{
    const int N = INTEGRATOR_NUM_SAMPLE;
    float _points[INTEGRATOR_NUM_SAMPLE];
    float _weights[INTEGRATOR_NUM_SAMPLE];

    // Note: Actual integration is performed in floating point precision,
    // but the roots and weights are determined in double precision
    // before rounding. This fixes cancellation badness for higher-degree
    // polynomials. This is only done once during precomputation and
    // should not be much of an issue (hence why this code is also
    // not very optimized).
    double legendre(double x, int n)
    {
        if (n == 0)
            return 1.0;
        if (n == 1)
            return x;

        double P0 = 1.0;
        double P1 = x;
        for (int i = 2; i <= n; ++i)
        {
            double Pi = double(((double)2.0 * i - (double)1.0) * x * P1 - (i - (double)1.0) * P0) / i;
            P0 = P1;
            P1 = Pi;
        }
        return P1;
    }

    double legendreDeriv(double x, int n)
    {
        return n / (x * x - 1.0) * (x * legendre(x, n) - legendre(x, n - 1));
    }

    double kthRoot(int k)
    {
        // Initial guess due to Francesco Tricomi
        // See http://math.stackexchange.com/questions/12160/roots-of-legendre-polynomial
        double x = cos(M_PI * (4.0 * k - 1.0) / (4.0 * N + 2.0)) *
                   (1.0 - 1.0 / (8.0 * N * N) + 1.0 / (8.0 * N * N * N));

        // Newton-Raphson
        for (int i = 0; i < 100; ++i)
        {
            double f = legendre(x, N);
            x -= f / legendreDeriv(x, N);
            if (abs(f) < 1e-6)
                break;
        }

        return x;
    }
    inline double sqr(double p)
    {
        return p * p;
    }

    __init()
    {
        for (int i = 0; i < N; ++i)
        {
            _points[i] = float(kthRoot(i + 1));
            _weights[i] = float((double)2.0 / (((double)1.0 - (double)sqr(_points[i])) * (double)sqr(legendreDeriv(double(_points[i]), N))));
        }
    }

    inline float integrate(float f)
    {
        float result = _points[0] * _weights[0];
        for (int i = 1; i < N; ++i)
            result += _points[i] * _weights[i];
        return result;
    }

    int numSamples()
    {
        return N;
    }
};

float fresnelDielectricExt(float cosThetaI_, inout float cosThetaT_, float eta) {
    if (eta == 1) {
        cosThetaT_ = -cosThetaI_;
        return 0.0f;
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float scale = (cosThetaI_ > 0) ? 1 / eta : eta,
          cosThetaTSqr = 1 - (1 - cosThetaI_ * cosThetaI_) * (scale * scale);

    /* Check for total internal reflection */
    if (cosThetaTSqr <= 0.0f) {
        cosThetaT_ = 0.0f;
        return 1.0f;
    }

    /* Find the absolute cosines of the incident/transmitted rays */
    float cosThetaI = abs(cosThetaI_);
    float cosThetaT = sqrt(cosThetaTSqr);

    float Rs = (cosThetaI - eta * cosThetaT)
             / (cosThetaI + eta * cosThetaT);
    float Rp = (eta * cosThetaI - cosThetaT)
             / (eta * cosThetaI + cosThetaT);

    cosThetaT_ = (cosThetaI_ > 0) ? -cosThetaT : cosThetaT;

    /* No polarization -- return the unpolarized reflectance */
    return 0.5f * (Rs * Rs + Rp * Rp);
}

inline float fresnelDielectricExt(float cosThetaI, float eta) { float cosThetaT;
    return fresnelDielectricExt(cosThetaI, cosThetaT, eta); }

struct HairBSDF : IBSDF, IDifferentiable
{
    float3 albedo; ///< Diffuse albedo.
    float m_eta;
    // longtitudinal shift
    float m_alpha;
    float alpha[3];
    // longtitudinal width
    float m_beta;
    float beta[3];
    float3 m_absorption;
    GaussLegendre integrator;

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, -wo.z) < kMinCosTheta)
            return float3(0.f);

        return M_1_PI * albedo * -wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        wo.z = -wo.z;
        lobeType = (uint)LobeType::DiffuseTransmission;

        if (min(wi.z, -wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = albedo;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, -wo.z) < kMinCosTheta)
            return 0.f;

        return M_1_PI * -wo.z;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        return AlbedoContributions(0.0f, 0.0f, albedo, 1.0f - albedo);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(0.5f, 0.5f);
        return r;
    }
}
