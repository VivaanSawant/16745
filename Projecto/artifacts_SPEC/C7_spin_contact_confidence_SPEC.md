# C7 spin + contact confidence

- seeds: `[7, 13, 19, 29, 37]`
- spin relevant (99% gate): `True`
- wrist slope significant: `True`
- release-time slope significant: `True`
- phase pass: `True`

## 99% CI summary

- max shift mm CI99: `121.49` to `124.63`
- max |score delta| CI99: `4.00` to `7.81`
- max |radial delta| mm CI99: `49.05` to `61.55`
- wrist slope CI99: `0.221` to `0.221`
- release-time slope CI99: `425.011` to `425.011`
- wrist radial slope CI99: `-0.027` to `-0.027`
- release-time radial slope CI99: `3.857` to `3.857`

Interpretation:
- Wrist channel is accepted if CI99 excludes zero.
- Release-time channel can be down-weighted if CI99 crosses zero.
