import numpy as np
import matplotlib.pyplot as plt
import black_scholes as bs

# single option
b = bs.BlackScholes(S=100, K=100, r=0.01, q=0.0, sigma=0.2, T=0.5, cdf_mode=0)
print("Call price:", b.call_price(), "Delta:", b.delta(True))

# price across strikes for a surface plot
S = 100.0
strikes = np.linspace(60,140,201)
sig = np.full_like(strikes, 0.2)
r = np.full_like(strikes, 0.01)
q = np.zeros_like(strikes)
T = np.full_like(strikes, 0.5)
types = np.zeros_like(strikes, dtype=np.int32)  # calls

prices = bs.batch_price(
    S=np.full_like(strikes, S),
    K=strikes,
    r=r, q=q, sigma=sig, T=T, type=types, cdf_mode=0
)

plt.plot(strikes, prices)
plt.xlabel('Strike')
plt.ylabel('Call Price')
plt.title('Call price vs Strike (S=100, sigma=20%)')
plt.show()

# implied vol example: given market prices (here artificially from model)
market_prices = prices  # pretend these are market prices
ivs = bs.batch_implied_vol(
    S=np.full_like(strikes, S),
    K=strikes, r=r, q=q, T=T,
    type=types, market_price=market_prices, cdf_mode=0
)
plt.plot(strikes, ivs)
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.title('Implied vol (roundtrip)')
plt.show()
