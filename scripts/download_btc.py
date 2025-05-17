import requests
import pandas as pd
import time
import datetime

BASE = "https://min-api.cryptocompare.com/data/v2/histohour"
PAIR  = {"fsym": "BTC", "tsym": "USD", "limit": 2000}      # 2 000 = max bars per call

def fetch_all_histohour(pair=PAIR):
    frames = []
    params = pair.copy()
    to_ts  = None                          # when None we start from "now" and page back
    # Unix timestamp for January 1, 2012
    start_of_2012 = int(datetime.datetime(2012, 1, 1).timestamp())
    while True:
        if to_ts is not None:
            params["toTs"] = to_ts
        res = requests.get(BASE, params=params, timeout=15).json()
        batch = res["Data"]["Data"]
        print(batch)
        if not batch:                      # no more data → finished
            break
        if batch[0]["close"] is None:
            break
        # Check if we've reached data before 2012 or if close price is 0
        if batch[0]["time"] < start_of_2012 or batch[0]["close"] == 0:
            frames.append(pd.DataFrame(batch))  # Include the current batch
            break
        frames.append(pd.DataFrame(batch))
        to_ts = batch[0]["time"] - 1       # step back 1 s before oldest bar fetched
        time.sleep(0.25)                   # polite pause – keeps you far below free-tier limits
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)   # oldest-to-newest

df = fetch_all_histohour()
df.to_csv("btc_hourly_cryptocompare.csv", index=False)
print(df.head(), "\n", df.tail())