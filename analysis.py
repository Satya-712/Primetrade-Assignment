import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
trades = pd.read_csv("historical_data.csv")
sentiment = pd.read_csv("fear_greed_index.csv")

# Convert dates
sentiment['date'] = pd.to_datetime(sentiment['date'])

trades['Date'] = pd.to_datetime(
    trades['Timestamp IST'],
    dayfirst=True
).dt.date

trades['Date'] = pd.to_datetime(trades['Date'])

# Merge datasets
merged = pd.merge(
    trades,
    sentiment,
    left_on='Date',
    right_on='date',
    how='left'
)

print("Merged Shape:")
print(merged.shape)

print("\nSentiment Counts:")
print(merged['classification'].value_counts())

# Average PnL
print("\n===== AVERAGE PNL BY SENTIMENT =====")

avg_pnl = merged.groupby(
    'classification'
)['Closed PnL'].mean()

print(avg_pnl.sort_values(ascending=False))

# Total PnL
print("\n===== TOTAL PNL BY SENTIMENT =====")

total_pnl = merged.groupby(
    'classification'
)['Closed PnL'].sum()

print(total_pnl.sort_values(ascending=False))

# Graph
avg_pnl.sort_values().plot(kind='bar')

plt.title("Average PnL by Market Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Closed PnL")

plt.tight_layout()
plt.show()
print("\n===== BUY vs SELL ANALYSIS =====")

buy_sell = merged.groupby('Side')['Closed PnL'].mean()

print(buy_sell)
print("\n===== TOP 10 TRADERS =====")

top_traders = merged.groupby(
    'Account'
)['Closed PnL'].sum()

top_traders = top_traders.sort_values(
    ascending=False
)

print(top_traders.head(10))
print("\n===== TOP 10 COINS =====")

top_coins = merged.groupby(
    'Coin'
)['Closed PnL'].sum()

top_coins = top_coins.sort_values(
    ascending=False
)
print(top_coins.head(10))
print("\n===== TOP 10 COINS =====")

top_coins = merged.groupby('Coin')['Closed PnL'].sum()

top_coins = top_coins.sort_values(
    ascending=False
)

print(top_coins.head(10))