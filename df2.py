import os
import sys
import json
import time
import asyncio
import random
import configparser
import pandas as pd
import numpy as np
from datetime import datetime
from quotexapi.stable_api import Quotex
from quotexapi.config import email, password

# --- Default Strategy Parameters ---
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # Adjusted oversold level for CALL
RSI_OVERBOUGHT = 70  # Adjusted overbought level for PUT
KELTNER_PERIOD = 20
KELTNER_MULTIPLIER = 2
CONFIRMATION_CANDLES = 3  # Number of candles for confirmation
BBANDS_PERIOD = 20
BBANDS_STDDEV = 2
STOCHASTIC_K = 14
STOCHASTIC_D = 3
STOCHASTIC_J = 3

# --- File Paths ---
SETTINGS_FILE = "settingz.ini"

# --- Create or Load Settings ---
def initialize_settings():
    config = configparser.ConfigParser()
    if not os.path.exists(SETTINGS_FILE):
        config['ACCOUNT'] = {
            'ACCOUNT_TYPE': 'PRACTICE'
        }
        config['TRADE'] = {
            'TRADE_AMOUNT': '1',
            'TRADE_DURATION': '60'
        }
        config['ASSETS'] = {
            'ASSET_LIST': 'USDARS_otc'
        }
        with open(SETTINGS_FILE, 'w') as configfile:
            config.write(configfile)
    config.read(SETTINGS_FILE)
    return config

settings = initialize_settings()

# --- Read Settings ---
ACCOUNT_TYPE = settings['ACCOUNT']['ACCOUNT_TYPE']
TRADE_AMOUNT = float(settings['TRADE']['TRADE_AMOUNT'])
TRADE_DURATION = int(settings['TRADE']['TRADE_DURATION'])
ASSETS = settings['ASSETS']['ASSET_LIST'].split(',')
CANDLE_PERIOD = 5  # Candle period (5 seconds)

# --- Client Initialization ---
client = Quotex(email=email, password=password, lang="pt")
trade_active = False  # Tracks if a trade is ongoing

# --- Utility Functions ---
async def connect():
    """Ensure connection to Quotex API."""
    for _ in range(5):
        if not await client.check_connect():
            connected, reason = await client.connect()
            if connected:
                print("Connected successfully!")
                return True
            print(f"Connection failed: {reason}")
            await asyncio.sleep(5)
        else:
            return True
    return False

async def fetch_candles(asset, count, timeframe=CANDLE_PERIOD):
    """Fetch historical candles."""
    try:
        current_time = time.time()
        candles = await client.get_candles(asset, current_time, count, timeframe)
        if candles is None:
            raise Exception("Failed to fetch candles")
        return candles
    except Exception as e:
        print(f"Error fetching candles for {asset}: {e}")
        raise

def validate_conditions(rsi, keltner, bbands, stochastic, last_candle):
    """Validate all strategy conditions using the last candle in the analysis set."""
    keltner_upper, keltner_lower, close_price = keltner
    bbands_upper, bbands_lower, _ = bbands
    stochastic_k, stochastic_d, stochastic_j = stochastic

    # Check conditions for CALL
    if (close_price < keltner_lower and rsi < RSI_OVERSOLD and
        close_price < bbands_lower and stochastic_k < 20 and stochastic_d < 20 and stochastic_j < 20):
        return "call"

    # Check conditions for PUT
    elif (close_price > keltner_upper and rsi > RSI_OVERBOUGHT and
          close_price > bbands_upper and stochastic_k > 80 and stochastic_d > 80 and stochastic_j > 80):
        return "put"

    return None
def confirm_trade(candles, direction):
    """Confirm trade with the last 5 confirmation candles."""
    confirmation_candles = candles[-CONFIRMATION_CANDLES:]  # Last 5 candles for confirmation

    if direction == "call":
        # Confirm if the trend of the confirmation candles is upward
        for i in range(1, len(confirmation_candles)):
            if confirmation_candles[i]['close'] < confirmation_candles[i-1]['close']:
                return False
        return True
    elif direction == "put":
        # Confirm if the trend of the confirmation candles is downward
        for i in range(1, len(confirmation_candles)):
            if confirmation_candles[i]['close'] > confirmation_candles[i-1]['close']:
                return False
        return True
    return False


def calculate_rsi(candles):
    """Calculate RSI based on candle data."""
    gains, losses = [], []
    for i in range(1, len(candles)):
        change = candles[i]['close'] - candles[i-1]['close']
        gains.append(max(0, change))
        losses.append(abs(min(0, change)))
    avg_gain = sum(gains[-RSI_PERIOD:]) / RSI_PERIOD
    avg_loss = sum(losses[-RSI_PERIOD:]) / RSI_PERIOD
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calculate_keltner_channels(candles):
    """Calculate Keltner Channels."""
    closes = pd.Series([c['close'] for c in candles])
    ema = closes.rolling(KELTNER_PERIOD).mean()
    atr = pd.Series([abs(c['high'] - c['low']) for c in candles]).rolling(KELTNER_PERIOD).mean()
    upper_channel = ema + (KELTNER_MULTIPLIER * atr)
    lower_channel = ema - (KELTNER_MULTIPLIER * atr)
    return upper_channel.iloc[-1], lower_channel.iloc[-1], closes.iloc[-1]

def calculate_bbands(candles):
    """Calculate Bollinger Bands."""
    closes = pd.Series([c['close'] for c in candles])
    sma = closes.rolling(BBANDS_PERIOD).mean()
    std_dev = closes.rolling(BBANDS_PERIOD).std()
    upper_band = sma + (BBANDS_STDDEV * std_dev)
    lower_band = sma - (BBANDS_STDDEV * std_dev)
    return upper_band.iloc[-1], lower_band.iloc[-1], closes.iloc[-1]

def calculate_stochastic(candles):
    """Calculate Stochastic Oscillator."""
    closes = pd.Series([c['close'] for c in candles])
    highs = pd.Series([c['high'] for c in candles])
    lows = pd.Series([c['low'] for c in candles])
    k = ((closes - lows.rolling(STOCHASTIC_K).min()) / (highs.rolling(STOCHASTIC_K).max() - lows.rolling(STOCHASTIC_K).min())) * 100
    d = k.rolling(STOCHASTIC_D).mean()
    j = (3 * k) - (2 * d)
    return k.iloc[-1], d.iloc[-1], j.iloc[-1]

async def place_trade(asset, direction):
    """Execute a trade and wait for the result."""
    global trade_active
    trade_active = True

    print(f"Placing trade: {direction.upper()} | {asset} | Amount: {TRADE_AMOUNT}")
    status, trade_info = await client.buy(TRADE_AMOUNT, asset, direction, TRADE_DURATION)

    if status and "id" in trade_info:
        trade_id = trade_info["id"]
        print("Trade placed. Waiting for result...")
        await asyncio.sleep(TRADE_DURATION + 5)

        result = await client.check_win(trade_id)
        if result > 0:
            print(f"Trade WON: +{result}")
        else:
            print(f"Trade LOST: -{TRADE_AMOUNT}")
    else:
        print("Trade placement failed.")
    trade_active = False

async def analyze_and_trade():
    """Main function to analyze and trade."""
    global trade_active
    settings = initialize_settings()  # Re-read settings each loop
    ASSETS = settings['ASSETS']['ASSET_LIST'].split(',')
    TRADE_AMOUNT = float(settings['TRADE']['TRADE_AMOUNT'])

    for asset in ASSETS:
        if trade_active:
            print("Trade active. Waiting for it to finish...")
            await asyncio.sleep(5)
            continue

        print(f"Analyzing asset: {asset}")

        try:
            candles = await fetch_candles(asset, 30)
            analysis_candles = candles[:-CONFIRMATION_CANDLES]  # First 25 candles for analysis
            confirmation_candles = candles[-CONFIRMATION_CANDLES:]  # Last 5 candles for confirmation

            # Calculate indicators
            rsi = calculate_rsi(analysis_candles)
            keltner = calculate_keltner_channels(analysis_candles)
            bbands = calculate_bbands(analysis_candles)
            stochastic = calculate_stochastic(analysis_candles)

            # Validate conditions
            last_analysis_candle = analysis_candles[-1]
            direction = validate_conditions(rsi, keltner, bbands, stochastic, last_analysis_candle)

            if direction:
                confirmed = confirm_trade(candles, direction)
                if confirmed:
                    await place_trade(asset, direction)
                else:
                    print(f"{direction.upper()} not confirmed. Skipping trade.")
            else:
                print("No trade conditions met.")

        except Exception as e:
            print(f"Error analyzing {asset}: {e}")

async def main():
    connected = await connect()
    if not connected:
        print("Unable to connect to Quotex API.")
        return

    while True:
        await analyze_and_trade()
        await asyncio.sleep(5)  # Random delay to avoid being flagged

if __name__ == "__main__":
    asyncio.run(main())
