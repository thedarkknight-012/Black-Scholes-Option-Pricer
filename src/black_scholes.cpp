// black_scholes_pybind.cpp
// Single-file Black-Scholes pricer + pybind11 bindings
// Build (example):
//   pip install pybind11
//   c++ -O3 -Wall -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
//       black_scholes_pybind.cpp -o black_scholes$(python3-config --extension-suffix)
// Then in Python: import black_scholes

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// ---------- Types ----------
using Real = double;

// ---------- Normal PDF / CDF ----------
inline Real norm_pdf(Real x) {
    static const Real INV_SQRT_2PI = 0.3989422804014327; // 1/sqrt(2π)
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

// high-accuracy using erf
inline Real norm_cdf_erf(Real x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Abramowitz-Stegun fast approximation
inline Real norm_cdf_as(Real x) {
    // Abramowitz & Stegun 7.1.26 approximation
    const Real a1 = 0.254829592;
    const Real a2 = -0.284496736;
    const Real a3 = 1.421413741;
    const Real a4 = -1.453152027;
    const Real a5 = 1.061405429;
    const Real p  = 0.3275911;
    Real sign = (x < 0.0) ? -1.0 : 1.0;
    Real ax = std::fabs(x) / std::sqrt(2.0);
    Real t = 1.0 / (1.0 + p * ax);
    Real y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * std::exp(-ax * ax);
    return 0.5 * (1.0 + sign * y);
}

// wrapper to pick method
inline Real norm_cdf(Real x, int mode = 0) {
    // mode: 0 -> erf (accurate); 1 -> Abramowitz-Stegun (faster)
    return (mode == 0) ? norm_cdf_erf(x) : norm_cdf_as(x);
}

// ---------- BlackScholes class ----------
struct BlackScholes {
    // Core parameters
    Real S;      // spot
    Real K;      // strike
    Real r;      // risk-free rate (annual, continuous)
    Real q;      // continuous dividend yield (annual)
    Real sigma;  // volatility (annual)
    Real T;      // time to maturity (years)
    // cdf_mode: 0 = erf (accurate), 1 = as (fast)
    int cdf_mode;

    BlackScholes(Real S_=100.0, Real K_=100.0, Real r_=0.0, Real q_=0.0,
                 Real sigma_=0.2, Real T_=1.0, int cdf_mode_ = 0)
        : S(S_), K(K_), r(r_), q(q_), sigma(sigma_), T(T_), cdf_mode(cdf_mode_) {}

    // d1 and d2 with guards
    inline void d1_d2(Real &d1, Real &d2) const {
        if (T <= 0.0 || sigma <= 0.0) {
            // degenerate => push to +/- large values
            Real sign = (S > K) ? 1.0 : -1.0;
            d1 = d2 = sign * 1e9;
            return;
        }
        Real sqrtT = std::sqrt(T);
        Real denom = sigma * sqrtT;
        d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / denom;
        d2 = d1 - denom;
    }

    // Prices
    inline Real call_price() const {
        if (T <= 0.0) return std::max((Real)0.0, S - K);
        if (sigma <= 0.0) {
            Real discS = S * std::exp(-q * T);
            Real discK = K * std::exp(-r * T);
            return std::max((Real)0.0, discS - discK);
        }
        Real d1, d2; d1_d2(d1, d2);
        Real discS = S * std::exp(-q * T);
        Real discK = K * std::exp(-r * T);
        return discS * norm_cdf(d1, cdf_mode) - discK * norm_cdf(d2, cdf_mode);
    }

    inline Real put_price() const {
        if (T <= 0.0) return std::max((Real)0.0, K - S);
        if (sigma <= 0.0) {
            Real discS = S * std::exp(-q * T);
            Real discK = K * std::exp(-r * T);
            return std::max((Real)0.0, discK - discS);
        }
        Real d1, d2; d1_d2(d1, d2);
        Real discS = S * std::exp(-q * T);
        Real discK = K * std::exp(-r * T);
        return discK * norm_cdf(-d2, cdf_mode) - discS * norm_cdf(-d1, cdf_mode);
    }

    // Greeks
    inline Real delta(bool call=true) const {
        if (T <= 0.0) {
            if (call) return (S > K) ? 1.0 : 0.0;
            else return (S < K) ? -1.0 : 0.0;
        }
        if (sigma <= 0.0) {
            // degenerate heuristics: use discounted step
            return call ? std::exp(-q * T) * ((S > K) ? 1.0 : 0.0) :
                          std::exp(-q * T) * ((S < K) ? -1.0 : 0.0);
        }
        Real d1, d2; d1_d2(d1, d2);
        if (call) return std::exp(-q * T) * norm_cdf(d1, cdf_mode);
        else return std::exp(-q * T) * (norm_cdf(d1, cdf_mode) - 1.0);
    }

    inline Real gamma() const {
        if (T <= 0.0 || sigma <= 0.0) return 0.0;
        Real d1, d2; d1_d2(d1, d2);
        return std::exp(-q * T) * norm_pdf(d1) / (S * sigma * std::sqrt(T));
    }

    inline Real vega() const {
        if (T <= 0.0) return 0.0;
        Real d1, d2; d1_d2(d1, d2);
        return S * std::exp(-q * T) * norm_pdf(d1) * std::sqrt(T); // per 1 vol unit
    }

    inline Real theta(bool call=true) const {
        if (T <= 0.0) return 0.0;
        if (sigma <= 0.0) return 0.0;
        Real d1, d2; d1_d2(d1, d2);
        Real term1 = - (S * norm_pdf(d1) * sigma * std::exp(-q * T)) / (2.0 * std::sqrt(T));
        if (call) {
            Real term2 = q * S * std::exp(-q * T) * norm_cdf(d1, cdf_mode);
            Real term3 = -r * K * std::exp(-r * T) * norm_cdf(d2, cdf_mode);
            return term1 + term2 + term3;
        } else {
            Real term2 = q * S * std::exp(-q * T) * norm_cdf(-d1, cdf_mode);
            Real term3 = -r * K * std::exp(-r * T) * norm_cdf(-d2, cdf_mode);
            return term1 - term2 + term3;
        }
    }

    inline Real rho(bool call=true) const {
        if (T <= 0.0) return 0.0;
        Real d1, d2; d1_d2(d1, d2);
        if (call) return K * T * std::exp(-r * T) * norm_cdf(d2, cdf_mode);
        else return -K * T * std::exp(-r * T) * norm_cdf(-d2, cdf_mode);
    }

    // Implied volatility: Newton-Raphson then bisection fallback.
    // market_price: market option price
    // call_flag: true = call, false = put
    inline Real implied_volatility(Real market_price, bool call_flag=true,
                                   Real initial_guess=0.2, Real tol=1e-10, int max_iters=100) const
    {
        // compute intrinsic lower bound (discounted)
        Real discS = S * std::exp(-q * T);
        Real discK = K * std::exp(-r * T);
        Real intrinsic = call_flag ? std::max((Real)0.0, discS - discK) : std::max((Real)0.0, discK - discS);
        if (market_price < intrinsic - 1e-12) return -1.0; // no solution

        if (T <= 0.0) {
            return (std::fabs(market_price - intrinsic) < tol) ? 0.0 : -1.0;
        }

        // Newton-Raphson
        Real sigma_est = std::max(1e-12, initial_guess);
        for (int i = 0; i < max_iters; ++i) {
            BlackScholes tmp = *this;
            tmp.sigma = sigma_est;
            Real price = call_flag ? tmp.call_price() : tmp.put_price();
            Real diff = price - market_price;
            if (std::fabs(diff) < tol) return sigma_est;
            Real v = tmp.vega();
            if (v < 1e-12) break;
            sigma_est -= diff / v;
            if (sigma_est <= 0.0) sigma_est = 1e-12;
            if (sigma_est > 10.0) sigma_est = 10.0;
        }

        // Bisection fallback
        Real lo = 1e-12, hi = 5.0;
        auto price_at = [&](Real vol)->Real {
            BlackScholes b = *this;
            b.sigma = vol;
            return call_flag ? b.call_price() : b.put_price();
        };
        Real plo = price_at(lo), phi = price_at(hi);
        // expand hi if needed
        for (int e = 0; e < 50 && !((plo <= market_price && market_price <= phi) || (phi <= market_price && market_price <= plo)); ++e) {
            hi *= 2.0;
            phi = price_at(hi);
        }
        if (!((plo <= market_price && market_price <= phi) || (phi <= market_price && market_price <= plo))) return -1.0;

        for (int it = 0; it < 200; ++it) {
            Real mid = 0.5 * (lo + hi);
            Real pm = price_at(mid);
            if (std::fabs(pm - market_price) < tol) return mid;
            if ((plo - market_price) * (pm - market_price) <= 0.0) {
                hi = mid; phi = pm;
            } else {
                lo = mid; plo = pm;
            }
        }
        return 0.5 * (lo + hi);
    }
};

// ---------- Batch helpers using numpy arrays ----------

// Helper to assert shapes and convert py::array to C++ vector
template<class T>
std::vector<T> pyarray_to_vector(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    std::vector<T> out;
    out.reserve((size_t)arr.size());
    auto r = arr.unchecked<1>();
    for (ssize_t i = 0; i < r.shape(0); ++i) out.push_back((T)r(i));
    return out;
}

// Batch price: accept arrays for parameters (broadcasting singletons allowed)
py::array_t<Real> batch_price(
    py::array_t<Real, py::array::c_style | py::array::forcecast> S_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> r_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> q_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> sigma_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> T_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> type_arr, // 0=call,1=put
    int cdf_mode = 0)
{
    // find output length (max of input lengths)
    auto len = [&](auto &a){ return (size_t)a.size(); };
    size_t n = 0;
    n = std::max(n, len(S_arr));
    n = std::max(n, len(K_arr));
    n = std::max(n, len(r_arr));
    n = std::max(n, len(q_arr));
    n = std::max(n, len(sigma_arr));
    n = std::max(n, len(T_arr));
    n = std::max(n, len(type_arr));
    if (n == 0) return py::array_t<Real>(); // empty

    // convenience views (fast index access)
    auto vS = S_arr.unchecked<1>();
    auto vK = K_arr.unchecked<1>();
    auto vr = r_arr.unchecked<1>();
    auto vq = q_arr.unchecked<1>();
    auto vsigma = sigma_arr.unchecked<1>();
    auto vT = T_arr.unchecked<1>();
    auto vtype = type_arr.unchecked<1>();

    py::array_t<Real> out(n);
    auto out_mut = out.mutable_unchecked<1>();

    // helper to pick element i with modulo broadcasting if len==1
    auto get = [&](auto &view, size_t i)->Real {
        size_t L = (size_t)view.shape(0);
        if (L == 0) return 0.0;
        return (Real)view[(ssize_t)(i % L)];
    };
    auto get_int = [&](auto &view, size_t i)->int {
        size_t L = (size_t)view.shape(0);
        if (L == 0) return 0;
        return (int)view[(ssize_t)(i % L)];
    };

    for (size_t i = 0; i < n; ++i) {
        BlackScholes bs(
            get(vS, i),
            get(vK, i),
            get(vr, i),
            get(vq, i),
            get(vsigma, i),
            get(vT, i),
            cdf_mode
        );
        int t = get_int(vtype, i);
        Real price = (t == 0) ? bs.call_price() : bs.put_price();
        out_mut[(ssize_t)i] = price;
    }
    return out;
}

// Batch implied vols: provide market prices and types arrays
py::array_t<Real> batch_implied_vol(
    py::array_t<Real, py::array::c_style | py::array::forcecast> S_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> K_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> r_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> q_arr,
    py::array_t<Real, py::array::c_style | py::array::forcecast> T_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> type_arr, // 0=call,1=put
    py::array_t<Real, py::array::c_style | py::array::forcecast> market_price_arr,
    int cdf_mode = 0,
    Real initial_guess = 0.2)
{
    // determine broadcast length
    auto len = [&](auto &a){ return (size_t)a.size(); };
    size_t n = 0;
    n = std::max(n, len(S_arr));
    n = std::max(n, len(K_arr));
    n = std::max(n, len(r_arr));
    n = std::max(n, len(q_arr));
    n = std::max(n, len(T_arr));
    n = std::max(n, len(type_arr));
    n = std::max(n, len(market_price_arr));
    if (n == 0) return py::array_t<Real>();

    auto vS = S_arr.unchecked<1>();
    auto vK = K_arr.unchecked<1>();
    auto vr = r_arr.unchecked<1>();
    auto vq = q_arr.unchecked<1>();
    auto vT = T_arr.unchecked<1>();
    auto vtype = type_arr.unchecked<1>();
    auto vmp = market_price_arr.unchecked<1>();

    py::array_t<Real> out(n);
    auto out_mut = out.mutable_unchecked<1>();

    auto get = [&](auto &view, size_t i)->Real {
        size_t L = (size_t)view.shape(0);
        if (L == 0) return 0.0;
        return (Real)view[(ssize_t)(i % L)];
    };
    auto get_int = [&](auto &view, size_t i)->int {
        size_t L = (size_t)view.shape(0);
        if (L == 0) return 0;
        return (int)view[(ssize_t)(i % L)];
    };

    for (size_t i = 0; i < n; ++i) {
        BlackScholes bs(
            get(vS, i),
            get(vK, i),
            get(vr, i),
            get(vq, i),
            (Real)0.2, // sigma initial (overwritten by solver)
            get(vT, i),
            cdf_mode
        );
        int t = get_int(vtype, i);
        Real mp = get(vmp, i);
        Real iv = bs.implied_volatility(mp, t == 0, initial_guess);
        out_mut[(ssize_t)i] = iv;
    }
    return out;
}

// ---------- pybind11 module ----------
PYBIND11_MODULE(black_scholes, m) {
    m.doc() = "Black-Scholes pricer with Greeks and implied vol (pybind11)";

    // BlackScholes class wrapper
    py::class_<BlackScholes>(m, "BlackScholes")
        .def(py::init<Real,Real,Real,Real,Real,Real,int>(),
             py::arg("S")=100.0, py::arg("K")=100.0, py::arg("r")=0.0, py::arg("q")=0.0,
             py::arg("sigma")=0.2, py::arg("T")=1.0, py::arg("cdf_mode")=0)
        .def_readwrite("S", &BlackScholes::S)
        .def_readwrite("K", &BlackScholes::K)
        .def_readwrite("r", &BlackScholes::r)
        .def_readwrite("q", &BlackScholes::q)
        .def_readwrite("sigma", &BlackScholes::sigma)
        .def_readwrite("T", &BlackScholes::T)
        .def_readwrite("cdf_mode", &BlackScholes::cdf_mode)
        .def("call_price", &BlackScholes::call_price)
        .def("put_price",  &BlackScholes::put_price)
        .def("delta", [](const BlackScholes &b, bool call){ return b.delta(call); }, py::arg("call")=true)
        .def("gamma", &BlackScholes::gamma)
        .def("vega", &BlackScholes::vega)
        .def("theta", [](const BlackScholes &b, bool call){ return b.theta(call); }, py::arg("call")=true)
        .def("rho", [](const BlackScholes &b, bool call){ return b.rho(call); }, py::arg("call")=true)
        .def("implied_vol", &BlackScholes::implied_volatility,
             py::arg("market_price"), py::arg("call_flag")=true, py::arg("initial_guess")=0.2,
             py::arg("tol")=1e-10, py::arg("max_iters")=100)
        .def("__repr__", [](const BlackScholes &b){
            return "<BlackScholes S=" + std::to_string(b.S) + " K=" + std::to_string(b.K) +
                   " r=" + std::to_string(b.r) + " q=" + std::to_string(b.q) +
                   " sigma=" + std::to_string(b.sigma) + " T=" + std::to_string(b.T) + ">";
        });

    // batch functions
    m.def("batch_price", &batch_price,
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"),
          py::arg("sigma"), py::arg("T"), py::arg("type"),
          py::arg("cdf_mode")=0,
          "Batch price arrays. type: 0=call,1=put. Broadcast singleton arrays.");

    m.def("batch_implied_vol", &batch_implied_vol,
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"),
          py::arg("T"), py::arg("type"), py::arg("market_price"),
          py::arg("cdf_mode")=0, py::arg("initial_guess")=0.2,
          "Batch implied vol solver. Broadcast singleton arrays.");
}
