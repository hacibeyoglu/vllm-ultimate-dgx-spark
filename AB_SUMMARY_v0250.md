# v0.25.0 vs maxsafe A/B (2026-07-14, GB10)

- **26B_maxsafe**: single-stream [74.8, 68.1, 71.0] tok/s · sweep {'c1': 71.7, 'c4': 164.1, 'c8': 329.6, 'c16': 558.6} · acceptance {'mean_len': 2.86, 'accept_pct': 18.6} · tools True
- **26B_v0250_clean**: single-stream [69.9, 71.3, 68.8] tok/s · sweep {'c8': 305.0, 'c16': 511.5} · acceptance {'mean_len': 2.97, 'accept_pct': 19.7} · tools True
- **26B_v0250**: single-stream [74.1, 73.0, 72.6] tok/s · sweep {'c1': 74.4, 'c4': 162.3, 'c8': 313.0, 'c16': 226.8} · acceptance {'mean_len': 2.69, 'accept_pct': 16.9} · tools True
- **27B_maxsafe**: single-stream [23.3, 23.2, 23.2] tok/s · sweep {'c1': 18.8, 'c4': 55.8, 'c8': 99.7} · acceptance {'mean_len': 3.77, 'accept_pct': 23.1} · tools True
- **27B_v0250**: single-stream [22.5, 22.1, 22.0] tok/s · sweep {'c1': 18.4, 'c4': 52.0, 'c8': 88.4} · acceptance {'mean_len': 3.31, 'accept_pct': 19.2} · tools True
- **35B_maxsafe**: single-stream [72.1, 75.4, 75.8] tok/s · sweep {'c1': 65.3, 'c4': 168.9, 'c8': 254.0, 'c12': 348.1} · acceptance {'mean_len': 2.59, 'accept_pct': 26.5} · tools True
- **35B_v0250**: single-stream [77.4, 78.7, 79.1] tok/s · sweep {'c1': 66.2, 'c4': 162.2, 'c8': 269.6, 'c12': 340.6} · acceptance {'mean_len': 2.61, 'accept_pct': 26.8} · tools True