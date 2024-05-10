#pragma once
#include <cmath>
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "MathUtils.h"
#include "algorithm"

constexpr int INTEGRATOR_NUM_SAMPLE = 140;
constexpr int AZIMUTHAL_PRECOMPUTE_RESOLUTION = 64;
constexpr int GAUSSIAN_DETECTOR_SAMPLES = 2048;

using namespace Falcor;

#define INTEGRATOR_NUM_SAMPLE 140
#define AZIMUTHAL_PRECOMPUTE_RESOLUTION 64
#define GAUSSIAN_DETECTOR_SAMPLES 2048
#define M_PI       3.14159265358979323846   // pi
static const float kMinCosTheta = 1e-6f;
static const float M_PIx2 = (float)M_PI * 2;
//// Convert radians to degrees
inline float radToDeg(float value) { return value * (180.0f / M_PI); }

/// Convert degrees to radians
inline float degToRad(float value) { return value * (M_PI / 180.0f); }

inline float gaussian_function(float b, float t)
{
    return exp(-t * t / (2 * b * b)) / sqrt(2.0f * M_PI) / b;
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

    GaussLegendre::GaussLegendre()
    {
        for (int i = 0; i < N; ++i)
        {
            _points[i] = float(kthRoot(i + 1));
            _weights[i] = float((double)2.0 / (((double)1.0 - (double)sqr(_points[i])) * (double)sqr(legendreDeriv(double(_points[i]), N))));
        }
    }


    inline float integrate()
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

float fresnelDielectricExt(float cosThetaI_, float &cosThetaT_, float eta) {
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

class MarschnerHair
{
public:

    float3 albedo;
    float m_eta;
    // longtitudinal shift
    float m_alpha;
    float alpha[3];
    // longtitudinal width
    float m_beta;
    float beta[3];
    float GD_table[GAUSSIAN_DETECTOR_SAMPLES][3];
    float3 m_absorption;
    float3 Np_table[AZIMUTHAL_PRECOMPUTE_RESOLUTION][AZIMUTHAL_PRECOMPUTE_RESOLUTION * 3];

    MarschnerHair()
    {
        m_eta = 1.55f;
        m_beta = 2.2f;
        m_absorption =  float3(0.3, 0.6, 0.9);
        m_beta = degToRad(m_beta);
        beta[0] = m_beta;
        beta[1] = m_beta / 2;
        beta[2] = m_beta * 2;
        m_alpha = 3.0f;
        m_alpha = degToRad(m_alpha);
        alpha[0] = m_alpha;
        alpha[1] = -m_alpha / 2;
        alpha[2] = -3 * m_alpha / 2;

        precompute();
    }

    inline float gaussian_function(float b, float t) const
    {
        return std::exp(-t * t / (2 * b * b)) / std::sqrt(2.0f * M_PI) / b;
    }

    float gaussian_detector(float b, float phi) const
    {
        float result = 0;
        float x, y, delta;
        int k = 0;

        do
        {
            x = gaussian_function(b, phi - M_PIx2 * float(k));
            y = gaussian_function(b, phi + M_PIx2 * float(k + 1));
            delta = x + y;
            k++;
            result += x + y;
        } while (delta > 1e-4f);
        return result;
    }

    inline float Phi(int p, float gamma_i, float gamma_t) const
    {
        return 2 * p * gamma_t - 2 * gamma_i + p * M_PI;
    }


    void precompute()
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < GAUSSIAN_DETECTOR_SAMPLES; j++)
            {
                GD_table[i][j] = gaussian_detector(beta[i], j / (GAUSSIAN_DETECTOR_SAMPLES - 1.0f) * M_PI * 2.0f);
            }
        }
        // precompute gamma_i table
        float gamma_i_table[INTEGRATOR_NUM_SAMPLE];
        GaussLegendre integrator;
        for (int i = 0; i < INTEGRATOR_NUM_SAMPLE; i++)
        {
            gamma_i_table[i] = std::asin(integrator._points[i]);
        }
        // precompute azimuthal scattering function from 2 dimension
        // cos(theta_d): [0, 1]
        // phi: [0, 2*PI]
        for (int y = 0; y < AZIMUTHAL_PRECOMPUTE_RESOLUTION; y++)
        {
            float cos_theta_d = float(y) / float(AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1);
            float sin_td = std::sqrt(1.f - cos_theta_d * cos_theta_d);
            float eta_prime = std::sqrt(m_eta * m_eta - sin_td * sin_td) / cos_theta_d;
            float cos_theta_t = std::sqrt(1.f - sin_td * sin_td * (1.f / m_eta) * (1.f / m_eta));
            float3 absorption_prime = m_absorption / cos_theta_t;

            float precom_fresnel[INTEGRATOR_NUM_SAMPLE], gamma_t_table[INTEGRATOR_NUM_SAMPLE];
            float3 absorption_table[INTEGRATOR_NUM_SAMPLE];
            for (int k = 0; k < INTEGRATOR_NUM_SAMPLE; k++)
            {
                precom_fresnel[k] = fresnelDielectricExt(cos_theta_d * std::cos(gamma_i_table[k]), m_eta);
                gamma_t_table[k] = std::asin(std::clamp(integrator._points[k] / eta_prime, -1.f, 1.f));
                absorption_table[k] = exp((-2.f * absorption_prime * (1.f + std::cos(2.f * gamma_t_table[k]))));
            }

            for (int x = 0; x < AZIMUTHAL_PRECOMPUTE_RESOLUTION; x++)
            {
                float phi = M_PI * 2.f * x / (AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1.f);
                for (int i = 0; i < 3; i++)
                {
                    float3 Np(0.f);
                    for (int j = 0; j < INTEGRATOR_NUM_SAMPLE; j++)
                    {
                        float Dp = approxGD(i, phi - Phi(i, gamma_i_table[j], gamma_t_table[j]));
                        float f = precom_fresnel[j];
                        if (i == 0)
                        {
                            Np += float3(0.5 * f * Dp * integrator._weights[j]);
                        }
                        else
                        {
                            float3 T = absorption_table[j];
                            float3 A = pow(T, i) * ((float)1.f - f) * ((float)1.f - f) * (float)std::pow(f, i - 1);
                            Np += (float)0.5 * A * Dp * integrator._weights[j];
                        }
                    }
                    Np_table[y][x * 3 + i] = Np;
                }
            }
        }
    }

    float approxGD(int p, float phi) const
    {
        float u = std::abs(phi * (0.5f * M_1_PI * (GAUSSIAN_DETECTOR_SAMPLES - 1)));
        int x0 = int(u);
        int x1 = x0 + 1;
        u -= x0;
        return GD_table[p][x0 % GAUSSIAN_DETECTOR_SAMPLES] * (1.0f - u) + GD_table[p][x1 % GAUSSIAN_DETECTOR_SAMPLES] * u;
    }

};
