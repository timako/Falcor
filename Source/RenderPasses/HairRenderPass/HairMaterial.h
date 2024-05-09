#pragma once
#include <cmath>
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "MathUtils.h"
#include "algorithm"

using namespace Falcor;

#define INTEGRATOR_NUM_SAMPLE 140
#define AZIMUTHAL_PRECOMPUTE_RESOLUTION 64
#define GAUSSIAN_DETECTOR_SAMPLES 2048
#define M_PI       3.14159265358979323846   // pi
static const float kMinCosTheta = 1e-6f;

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

// float3 squareToUniformSphere<S : ISampleGenerator>(inout S sg) {
//     float2 sample = sampleNext2D(sg);
//     float z = 1.0f - 2.0f * sample.y;
//     float r = sqrt(1.0f - z * z); //safe sqrt todo
//     float sinPhi, cosPhi;
//     sincos(2.0f * M_PI * sample.x, sinPhi, cosPhi);
//     return float3(r * cosPhi, r * sinPhi, z);
// }


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

struct HairBSDF
{
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

    /** Evaluates the BSDF.
    \param[in] wi Incident direction.
    \param[in] wo Outgoing direction.
    \param[in,out] sg Sample generator.
    \return Returns f(wi, wo) * dot(wo, n).
*/

    float approxGD(int p, float phi)
    {
        float u = abs(phi * (0.5f * M_1_PI * (GAUSSIAN_DETECTOR_SAMPLES - 1)));
        int x0 = int(u);
        int x1 = x0 + 1;
        u -= x0;
        return GD_table[x0 % GAUSSIAN_DETECTOR_SAMPLES][p] * (1.0f - u) + GD_table[x1 % GAUSSIAN_DETECTOR_SAMPLES][p] * u;
    };

    float3 approxNp(float cos_theta_d, float phi, int i)
    {
        // costhetad -> phi
        float u = (AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1) * phi * M_1_PI * 0.5f;
        float v = (AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1) * cos_theta_d;
        int x0 = std::clamp(int(u), 0, AZIMUTHAL_PRECOMPUTE_RESOLUTION - 2);
        int y0 = std::clamp(int(v), 0, AZIMUTHAL_PRECOMPUTE_RESOLUTION - 2);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        u = std::clamp(u - x0, 0.0f, 1.0f);
        v = std::clamp(v - y0, 0.0f, 1.0f);

        return (Np_table[y0][x0 * 3 + i] * (1.0f - u) + Np_table[y0][x1 * 3 + i] * u) * (1.0f - v) +
               (Np_table[y1][x0 * 3 + i] * (1.0f - u) + Np_table[y1][x1 * 3 + i] * u) * v;
    }

    HairBSDF::HairBSDF()
    {
        m_eta = 1.55f;
        m_beta = 7.5f;
        m_absorption =  float3(0.3, 0.6, 0.9);
        m_beta = degToRad(m_beta);
        beta[0] = m_beta;
        beta[1] = m_beta / 2;
        beta[2] = m_beta * 2;
        m_alpha = -5.0;
        m_alpha = degToRad(m_alpha);
        alpha[0] = m_alpha;
        alpha[1] = -m_alpha / 2;
        alpha[2] = -3 * m_alpha / 2;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < GAUSSIAN_DETECTOR_SAMPLES; j++)
            {
                GD_table[j][i] = gaussian_detector(beta[i], j / (GAUSSIAN_DETECTOR_SAMPLES - 1.0f) * M_PI * 2.0f);
            }
        }
        // precompute gamma_i table
        float gamma_i_table[INTEGRATOR_NUM_SAMPLE];
        GaussLegendre integrator;
        for (int i = 0; i < INTEGRATOR_NUM_SAMPLE; i++)
        {
            gamma_i_table[i] = asin(integrator._points[i]);
        }


        // precompute azimuthal scattering function from 2 dimension
        // cos(theta_d): [0, 1]
        // phi: [0, 2*PI]
        //预计算 cos_theta_d: longitudinal方向
        for (int y = 0; y < AZIMUTHAL_PRECOMPUTE_RESOLUTION; y++)
        {
            // cos theta_d, sin theta_d
            float cos_theta_d = std::clamp(float(y) / float(AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1) + float(0.00001), 0.0f, 1.0f);
            float sin_td = sqrt(1.f - cos_theta_d * cos_theta_d);
            // eta_prime 是在azimuthal scattering方向的折射率分量
            float eta_prime = sqrt(m_eta * m_eta - sin_td * sin_td) / cos_theta_d;
            float cos_theta_t = sqrt(1.f - sin_td * sin_td * sqr(1.f / m_eta));
            float3 absorption_prime = m_absorption / cos_theta_t;

            float precom_fresnel[INTEGRATOR_NUM_SAMPLE], gamma_t_table[INTEGRATOR_NUM_SAMPLE];
            float3 absorption_table[INTEGRATOR_NUM_SAMPLE];

            // FILE *out;
            // out = fopen( "D:/Debug/debug.txt", "w" );
            // if( out == NULL )
            //     exit(1);

            for (int k = 0; k < INTEGRATOR_NUM_SAMPLE; k++)
            {
                precom_fresnel[k] = fresnelDielectricExt(cos_theta_d * cos(gamma_i_table[k]), m_eta);
                gamma_t_table[k] = asin(std::clamp(integrator._points[k] / eta_prime, -1.f, 1.f));
                absorption_table[k] = exp((-2.f * absorption_prime * (1.f + cos(2.f * gamma_t_table[k]))));
                // fprintf(out, "precom_fresnel: %f when k = %d, cos_theta_d = %f, cos(gamma_i_table[k]) = %f\n", precom_fresnel[k], k, cos_theta_d, cos(gamma_i_table[k]));
            }




            // 预计算Phi: azimuthal scattering方向

            for (int x = 0; x < AZIMUTHAL_PRECOMPUTE_RESOLUTION; x++)
            {
                float phi = M_PI * 2.f * x / (AZIMUTHAL_PRECOMPUTE_RESOLUTION - 1.f);

                // fprintf(out, "When phi = %f\n", phi);

                for (int i = 0; i < 3; i++)
                {
                    // fprintf(out, "When i = %d\n", i);
                    float3 Np = float3(0.f);
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
                            // printf("T = %f %f %f\n", T.x, T.y, T.z);
                            float3 A = pow(T, i) * sqr(1.f - f) * float(pow(f, i - 1));

                            // fprintf(out, "pow(T, i)= %f %f %f when i = %d\n", pow(T, i).x, pow(T, i).y, pow(T, i).z, i);
                            // fprintf(out, "sqr(1.f - f) = %f when i = %d\n", sqr(1.f - f), i);
                            // fprintf(out, "A = %f %f %f when i = %d\n", A.x, A.y, A.z, i);
                            Np += A * float(0.5) * Dp * integrator._weights[j];
                            // fprintf(out, "Dp: %f when i = %d\n", Dp, i);
                            // fprintf(out, "Np: %f %f %f when i = %d\n", Np.x, Np.y, Np.z, i);
                        }

                    }
                    // fprintf(out, "Np: %f %f %f when i = %d, phi = %f \n", Np.x, Np.y, Np.z, i, phi);
                    Np_table[y][x * 3 + i] = Np;
                }
            }

        }
    }

    float gaussian_detector(float b, float phi)
    {
        float result = 0;
        float x, y, delta;
        int k = 0;
        float M_PIx2 = float(M_PI) * 2;
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

    inline float Phi(int p, float gamma_i, float gamma_t)
    {
        return 2 * p * gamma_t - 2 * gamma_i + p * M_PI;
    }

    /** Evaluates the BSDF.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \param[in,out] sg Sample generator.
        \return Returns f(wi, wo) * dot(wo, n).
    */

    float3 eval(const float3 wi, const float3 wo) {
        if (std::min(wi.z, -wo.z) < kMinCosTheta)
            return float3(0.f);

        float theta_i = asin(std::clamp(wi.x, -1.0f, 1.0f));
        float theta_r = asin(std::clamp(wo.x, -1.0f, 1.0f));
        float phi_i = atan2(wi.y, wi.z);
        float phi_r = atan2(wo.y, wo.z);
        float phi = phi_r - phi_i, theta_d = abs((theta_r - theta_i)) / 2, theta_h = (theta_i + theta_r) / 2;
        if (phi < 0.f)
            phi += float(M_PI) * 2;
        if (phi > M_PI * 2)
            phi -= float(M_PI) * 2;

        float3 result = float3(0.f);

        // M term, we'll simply use gaussian distribution here
        float Mp[3];
        for (int i = 0; i < 3; i++)
        {
            Mp[i] = gaussian_function(beta[i], theta_h - alpha[i]);
        }
        float cos_theta_d = cos(theta_d);
        Mp[0] = gaussian_function(beta[0], theta_h * 2);
        result += Mp[0] * approxNp(cos_theta_d, phi, 0);
        result += Mp[1] * approxNp(cos_theta_d, phi, 1);
        result += Mp[2] * approxNp(cos_theta_d, phi, 2);

        return result;
    }


    /** Samples the BSDF.
        \param[in] wi Incident direction.
        \param[out] wo Outgoing direction.
        \param[out] pdf pdf with respect to solid angle for sampling outgoing direction wo (0 if a delta event is sampled).
        \param[out] weight Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
        \param[out] lobeType Sampled lobeType (see LobeType).
        \param[in,out] sg Sample generator.
        \return Returns true if successful.
    */
    // bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float _pdf, out float3 weight, out uint lobeType, inout S sg) {
    //     wo = squareToUniformSphere(sg);
    //     _pdf = evalPdf(wi, wo);
    //     lobeType = (uint)LobeType::DiffuseReflection;
    //     if (min(wi.z, -wo.z) < kMinCosTheta)
    //     {
    //         weight = {};
    //         return false;
    //     }
    //     weight = eval(wi, wo, sg) / _pdf;
    //     return true;
    // }

    /** Evaluates the directional pdf for sampling outgoing direction wo.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \return Returns the pdf with respect to solid angle for sampling outgoing direction wo (0 for delta events).
    */
    float evalPdf(const float3 wi, const float3 wo) {
        return INV_FOURPI;
    }

    /** Albedo (hemispherical reflectance) of the BSDF. Relfection+transmission hemisphere should be <= 1.0.
        \param[in] wi Incident direction.
        \param[in] lobetype lobe types to be evaluated
        \return Returns the albedo.
    */
    // AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype) {
    //     return AlbedoContributions(albedo, 1.0f - albedo, 0.0f, 0.0f);
    // }

    /** Information about roughness of the BSDF in (various) forms.
        \param[in] wi Incident direction.
        \return Returns the roughness.
    */
    // RoughnessInformation getRoughnessInformation(const float3 wi) {
    //     RoughnessInformation r;
    //     r.roughnessBSDFNotation = float2(0.5f, 0.5f);
    //     return r;
    // }

};
