import ccxt
import time
import math
import requests
import pandas as pd
import numpy as np
import logging
from collections import deque
from datetime import datetime
import json
from datetime import datetime, timedelta


#slow internet fix for turkey
def get_server_time_offset():
    res = requests.get('https://api.bybit.com/v5/market/time')
    res.raise_for_status()
    data = res.json()
    print(f"Response data: {data}")

    # Use the 'time' field directly (milliseconds)
    server_time = int(data['time'])  # This is already in milliseconds
    local_time = int(time.time() * 1000)
    offset = server_time - local_time
    print(f"Server time: {server_time}, Local time: {local_time}, Offset: {offset} ms")
    return offset

offset = get_server_time_offset()

def get_corrected_timestamp():
    margin = 500
    return int(time.time() * 1000) + offset - margin

# Configure logging

log_filename = f"bot_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
        logging.StreamHandler()  # still prints to terminal
    ]
)

# Set trading pair
SYMBOL = 'ETH/USDT:USDT'

# Configure Bybit API for real trading (use with caution)
exchange = ccxt.bybit({
    'apiKey': 'Z4JbKx0Ck80GhZI08A',
    'secret': 'BJDFVGfR8WdPODBv864wXBJBSnu5liGIWqoG',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'linear',
        'recvWindow': 15000, # 10 secs
    }
})
exchange.load_markets()
market = exchange.market(SYMBOL)

# Leverage setting
LEVERAGE = 50
TRADE_USDT = 10  # Amount of USDT to use per trade
ATR_MULTIPLIER = 1.5  # ATR multiplier for take profit calculation

# Set leverage for the trading symbol
def set_leverage(symbol, leverage):
    try:
        market = exchange.market(symbol)
        if not market.get('linear', False):
            logging.warning(f"{symbol} is not a linear market. Leverage not set.")
            return

        exchange.set_leverage(leverage, symbol)
        logging.info(f"Leverage set to {leverage}x for {symbol}")
    except Exception as e:
        if '110043' in str(e):
            logging.info(f"Leverage already set to {leverage}x for {symbol}")
        else:
            logging.error(f"Failed to set leverage: {e}")

# Check API connection
def check_api_connection():
    try:
        exchange.fetch_ticker(SYMBOL)
        logging.info("API Connection successful!")
        return True
    except Exception as e:
        logging.error(f"API connection failed: {e}")
        return False
#Global Variables
in_position = False              # Tracks whether a trade is open
tp_triggered = False             # Prevents TP from triggering multiple times
is_trading = False               # Prevents duplicate trade() calls
current_position_size = 0.0      # Stores current position size locally
stop_loss_price = None            # Stores stop loss price locally
take_profit_price = None  
          # Stores take profit price locally
# ATR Settings
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2   # stop loss = entry ± ATR × multiplier
TIMEFRAME = '1m'  # Main timeframe for signals
#Trend Filters
ADX_PERIOD = 14
ADX_THRESHOLD = 25          # strength gate
USE_DI_GATE = True         # require DI direction to agree with trade
DI_BUFFER = 0               # optional, set >0 if you also require DI dominance
USE_LAST_CLOSED = True      # use last closed bar for ADX
MAX_STRETCH = 0.005         # 0.5% max distance from EMA20 to allow entry
# --- EMA Filters Settings ---
USE_EMA_DISTANCE_FILTER = True
USE_EMA_BREAKOUT_FILTER = True
USE_ATR_EMA_OVERLAP_FILTER = True
USE_EMA_SQUEEZE_FILTER = True

EMA_DISTANCE_THRESHOLD = 0.004  # 0.3% distance from EMA100/200
EMA_SQUEEZE_THRESHOLD = 0.002   # 0.2% → EMAs converging too close

# --- Regime Filter Settings ---
USE_REGIME_FILTER = True
ATR_LOOKBACK = 20
ATR_TREND_WINDOW = 5
ATR_MIN_MULTIPLIER = 1.2   # require ATR rising by at least 20% vs past
R2_THRESHOLD = 0.3         # require 1m trend R² > 0.3

# --- Adaptive Sizing Settings ---
USE_ADAPTIVE_SIZING = True
VOL_SPIKE_MULTIPLIER = 2.0  # shrink size if ATR > 2x rolling median
MIN_POSITION_SCALE = 0.25   # minimum size fraction (25%)

# --- Stop Hunt Filter Settings ---
USE_STOP_HUNT_FILTER = True
WICK_RATIO_THRESHOLD = 0.6   # wick > 60% of candle = stop hunt
VOLUME_SPIKE_MULTIPLIER = 2  # volume > 2x median = suspicious

# --- Volatility Shock Filter ---
USE_VOLATILITY_SHOCK_FILTER = True
BIG_MOVE_MULTIPLIER = 3.0      # 3x ATR move in one candle = skip
COOLDOWN_MINUTES = 30          # skip trading for 30 minutes after shock



# Helper indicator functions
def ema(prices, length):
    return prices.ewm(span=length, adjust=False).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = EMA with alpha=1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_ohlcv(symbol, timeframe, limit=100):
    since = exchange.milliseconds() - (limit * 60 * 1000)  # 1m candles
    data = exchange.fetch_ohlcv(
        symbol,
        timeframe,
        since=since,
        limit=limit,
        params={"category": "linear"}  # required for v5
    )
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def calculate_order_quantity(symbol, current_price, usdt_amount, leverage):
    market = exchange.market(symbol)
    #target coin amount
    raw_qty = (usdt_amount * leverage) / current_price
    #Snap to exchange precision
    qty = float(exchange.amount_to_precision(symbol, raw_qty))

    #enforce minimum order size
    min_qty = float(market['limits']['amount']['min'] or 0)
    if min_qty and qty < min_qty:
        qty = float(exchange.amount_to_precision(symbol, min_qty))
    
    return qty




def get_open_position(symbol=SYMBOL, retries=3, delay=2):
    """
    Retrieves the current position for a given symbol with retries.
    Returns None if no position is found.
    """
    for attempt in range(retries):
        try:
            positions = exchange.fetch_positions([symbol])
            if positions:
                for pos in positions:
                    if pos['symbol'] == symbol and float(pos.get('contracts', 0)) > 0:
                        return {
                            'side': pos['side'].lower(),  # 'long' or 'short'
                            'size': float(pos.get('contracts', 0)),
                            'entryPrice': float(pos.get('entryPrice', 0) or pos.get('avgEntryPrice', 0))
                        }
            return None
        except Exception as e:
            print(f"[get_position] Error fetching position (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None












def fetch_last_price(symbol):
    # Fetch the latest ticker price from exchange
    ticker = exchange.fetch_ticker(symbol)
    return ticker['last']




def place_limit_order(side, amount, price, post_only=True, reduce_only=False):
    try:
        params = {
            'reduceOnly': reduce_only,
            'postOnly': post_only
        }
        order = exchange.create_limit_order(
            symbol=SYMBOL,
            side=side,
            amount=amount,
            price=price,
            params=params
        )
        logging.info(f"Limit {side.upper()} order placed: Side={side}, Amount={amount}, Price={price}, ReduceOnly={reduce_only}")
        return {
            'id': order['id'],
            side: side,
            'amount': amount,
            'price': price,
            reduce_only: reduce_only,
            'raw': order
        }
        
    except Exception as e:
        print(f"Order error: {e}")
        return None
def get_order_status(order_id):
    try:
        open_orders = exchange.fetch_open_orders(SYMBOL)
        for order in open_orders:
            if order['id'] == order_id:
                return order['status']  # still open
        return 'closed'  # Not in open orders → assume filled or canceled
    except Exception as e:
        print(f"Fetch order status error: {e}")
        return None
def cancel_order(order_id):
    try:
        result = exchange.cancel_order(order_id, SYMBOL)
        print(f"Order {order_id} canceled: {result}")
        return True
    except Exception as e:
        print(f"Cancel order error: {e}")
        return False
def get_entry_price_long():
    last_price = fetch_last_price(SYMBOL)  # You'll need to implement this or reuse last_1m['close']
    return last_price * 0.999  # 0.1% below

def get_entry_price_short():
    last_price = fetch_last_price(SYMBOL)
    return last_price * 1.001  # 0.1% above
def place_stop_loss_limit(side, qty, stop_price):
    try:
        opposite = 'sell' if side == 'long' else 'buy'

        # define the actual limit price once the stop triggers
        if side == 'long':
            # stop loss triggers BELOW, so set limit slightly below stop
            limit_price = round(stop_price * 0.999, 2)
        else:
            # stop loss triggers ABOVE, so set limit slightly above stop
            limit_price = round(stop_price * 1.001, 2)

        order = exchange.create_order(
            symbol=SYMBOL,
            type='limit',      # this is the execution type (stop-limit)
            side=opposite,
            amount=qty,
            price=limit_price, # must include this!
            params={
                'stopLossPrice': stop_price,  # trigger level
                'reduceOnly': True,
                'triggerBy': 'MarkPrice'
            }
        )

        print(f"✅ Stop-limit placed: Trigger={stop_price}, Limit={limit_price}, Side={opposite}")
        return order

    except Exception as e:
        print(f"❌ Stop loss error: {e}")
        return None




def retry_limit_order(side, qty, price_func, max_retries=20, max_cycles=6, wait_sec=120, post_only=False, reduce_only=False):
    """
    Retry placing limit order up to max_retries times per cycle, max_cycles cycles.
    price_func is a function that returns current limit price for order.
    """
    for cycle in range(1, max_cycles + 1):
        print(f"Cycle {cycle} starting...")
        for attempt in range(1, max_retries + 1):
            price = price_func()
            print(f"Attempt {attempt}: Placing {side} limit order at {price}")
            order = place_limit_order(side, qty, price, post_only=post_only, reduce_only=reduce_only)
            if order is None:
                print("Order placement failed, retrying...")
                continue

            order_id = order['id']
            start_time = time.time()

            while time.time() - start_time < wait_sec:
                status = get_order_status(order_id)
                if status == 'closed':
                    print(f"Order {order_id} filled successfully.")
                    return order  # filled, exit retry
                
                
                
                elif status == 'canceled':
                    print(f"Order {order_id} was canceled externally.")
                    break
                time.sleep(1)

            # Timeout reached, cancel order and retry
            print(f"Order {order_id} not filled in {wait_sec} seconds, canceling...")
            cancel_order(order_id)
        print(f"Cycle {cycle} ended without fill, moving to next cycle.")

    print(f"All {max_cycles} cycles and {max_retries} retries exhausted without fill.")
    return None

def close_position(side, amount):
    try:
        last_price = fetch_last_price(SYMBOL)
        if side == 'long':
            # Close long by selling slightly above last price to get filled limit sell
            limit_price = last_price * 1.001  # 0.1% above last price
            order_side = 'sell'
        else:
            # Close short by buying slightly below last price to get filled limit buy
            limit_price = last_price * 0.999  # 0.1% below last price
            order_side = 'buy'

        order = exchange.create_limit_order(SYMBOL, order_side, amount, limit_price, {'postOnly': True, 'reduceOnly': True})
        print(f"{datetime.now()} - Limit {order_side} order placed to close {side} position: {order}")
        return order
    except Exception as e:
        print(f"Close position limit order error: {e}")
        return None
def close_position_market(side, size):
    try:
        opposite_side = 'sell' if side == 'long' else 'buy'
        order = exchange.create_order(
            symbol=SYMBOL,
            type='market',
            side=opposite_side,
            amount=size,
            params={"reduceOnly": True}
        )
        logging.info(f"🛑 Market {opposite_side.upper()} to close {side} position: {order}")
        print(f"🛑 Market {opposite_side.upper()} to close {side} position: {order}")
        return order
    except Exception as e:
        logging.error(f"❌ Error closing {side} with market order: {e}")
        return None

def calculate_adx(df, period=14, eps=1e-9):
    if df is None or df.empty or len(df) < period + 5:
        out = df.copy()
        out['+DI'] = np.nan; out['-DI'] = np.nan; out['ADX'] = np.nan
        return out

    d = df[['high','low','close']].astype(float).copy()



    prev_close = d['close'].shift(1)
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - prev_close).abs(),
        (d['low']  - prev_close).abs()
    ], axis=1).max(axis=1)

    upMove   = d['high'] - d['high'].shift(1)
    downMove = d['low'].shift(1) - d['low']
    plus_dm  = np.where((upMove > downMove) & (upMove > 0), upMove, 0.0)
    minus_dm = np.where((downMove > upMove) & (downMove > 0), downMove, 0.0)

    alpha = 1.0 / period
    tr_rma       = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_rma  = pd.Series(plus_dm, index=d.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_rma = pd.Series(minus_dm, index=d.index).ewm(alpha=alpha, adjust=False).mean()

    denom = (tr_rma + eps)
    plus_di  = 100.0 * (plus_dm_rma  / denom)
    minus_di = 100.0 * (minus_dm_rma / denom)

    dx = 100.0 * ((plus_di - minus_di).abs() / ((plus_di + minus_di).replace(0, np.nan) + eps))
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    out = df.copy()
    out['+DI'] = plus_di.replace([np.inf, -np.inf], np.nan)
    out['-DI'] = minus_di.replace([np.inf, -np.inf], np.nan)
    out['ADX'] = adx.replace([np.inf, -np.inf], np.nan)
    return out

def calc_r2(series, window=20):
    """Calculate R² of linear regression slope on close s."""
    if len(series) < window:
        return 0
    y = series[-window:]
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    p = np.poly1d(coeffs)
    y_hat = p(x)
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / (ss_tot + 1e-9))

def detect_stop_hunt(candle, df, wick_ratio_threshold=0.6, vol_mult=2):
    """
    Detect stop-hunt candle: long wick + closes back inside + volume spike.
    candle: last row (Series)
    df: dataframe with volume
    """
    body = abs(candle['close'] - candle['open'])
    high_wick = candle['high'] - max(candle['open'], candle['close'])
    low_wick  = min(candle['open'], candle['close']) - candle['low']
    full_range = candle['high'] - candle['low']

    # wick ratios
    upper_ratio = high_wick / (full_range + 1e-9)
    lower_ratio = low_wick / (full_range + 1e-9)

    # volume check
    vol_med = df['volume'].rolling(20).median().iloc[-2]
    vol_spike = candle['volume'] > vol_med * vol_mult

    if (upper_ratio > wick_ratio_threshold or lower_ratio > wick_ratio_threshold) and vol_spike:
        return True
    return False


# --- MAIN STRATEGY LOOP ---
def main():
    in_position = False              # Tracks whether a trade is open
    tp_triggered = False             # Prevents TP from triggering multiple times
    is_trading = False               # Prevents duplicate trade() calls
    current_position_size = 0.0      # Stores current position size locally
    stop_loss_price = None            # Stores stop loss price locally
    take_profit_price = None  
    set_leverage(SYMBOL, LEVERAGE)
    position = get_open_position()
    logging.info(f"DEBUG: position returned: {position} (type: {type(position)})")
    prev_side = position_side if 'position_side' in locals() else None
    prev_tp = take_profit_price if 'take_profit_price' in locals() else None
    if position is None:
        if prev_side:
            logging.warning("⚠️ Position temporarily missing — keeping last known side & TP.")
            position_side = prev_side
            take_profit_price = prev_tp
        else:
            position_side = None
            position_size = 0
            entry_price = None

    else:
        # safe access using get()
        position_side = position.get('side')
        position_size = abs(position.get('size', 0))
        entry_price = position.get('entryPrice')


    # You can now use position_side, position_size, and entry_price safely




    #Local Variables
    entry_order = None
    take_profit_price = None
    entry_order = None
    tp_order = None
    stop_loss_order = None
    tp_set = False
    sl_set = False
    last_skip_reason = None  # track last skip reason to avoid log spam
    skip_until = datetime.now() - timedelta(minutes=1)


    while True:
        try:
            #start variables
            rsi = None
            atr_val = None
            # Fetch data
            df_1m = fetch_ohlcv(SYMBOL, '1m')
            if df_1m.empty:
                logging.warning("1m OHLCV data is empty, skipping iteration.")
                time.sleep(60)
                continue
            df_5m = fetch_ohlcv(SYMBOL, '5m', limit=100)
            if df_5m.empty:
                logging.warning("5m OHLCV data is empty, skipping iteration.")
                time.sleep(60)
                continue
            df_1h = fetch_ohlcv(SYMBOL, '1h')
            if df_1h.empty:
                logging.warning("1h OHLCV data is empty, skipping iteration.")
                time.sleep(60)
                continue

            # --- Cooldown check (add this right here) ---
            if 'skip_until' in locals() and datetime.now() < skip_until:
                logging.info(f"Cooling down due to volatility shock until {skip_until.strftime('%H:%M:%S')}")
                time.sleep(60)
                continue
            # Calculate indicators
            for df in [df_1m, df_5m, df_1h]:
                df['ema_fast'] = ema(df['close'], 20)
                df['ema_mid'] = ema(df['close'], 50)
                df['ema_slow'] = ema(df['close'], 100)

            df_1m['rsi'] = calculate_rsi(df_1m['close'])
            df_1m['atr'] = atr(df_1m)
            last_1m = df_1m.iloc[-1]

            df_5m = calculate_adx(df_5m, period=14)
            adx_val = df_5m['ADX'].iloc[-2] if len(df_5m) >= 2 else np.nan



            atr_val = last_1m['atr']
            rsi_val = last_1m['rsi']

            if pd.isna(atr_val) or pd.isna(rsi_val):
                logging.warning(f"Indicators not ready yet (ATR={atr_val}, RSI={rsi_val}), skipping...")
                time.sleep(60)
                continue

            
            if pd.isna(adx_val):
                # One-time diagnostics to find root cause
                # (Guard with a flag if you don’t want this every minute.)
                last_rows = df_5m[['timestamp','high','low','close','+DI','-DI','ADX']].tail(5) \
                    if 'timestamp' in df_5m.columns else df_5m[['high','low','close','+DI','-DI','ADX']].tail(5)
                logging.info("ADX not ready (NaN). Diagnostics:")
                logging.info(f"df_5m len={len(df_5m)}, NaNs: "
                            f"TR? can't see here, +DI NaNs={df_5m['+DI'].isna().sum()}, "
                            f"-DI NaNs={df_5m['-DI'].isna().sum()}, ADX NaNs={df_5m['ADX'].isna().sum()}")
                logging.info(f"Tail:\n{last_rows}")
                time.sleep(60)
                continue




            # Latest bars
            last_1m = df_1m.iloc[-1]
            prev_1m = df_1m.iloc[-2]
            last_5m = df_5m.iloc[-1]
            prev_5m = df_5m.iloc[-2]

            # Conditions
            long_trend = (
                df_1m['ema_fast'].iloc[-1] > df_1m['ema_mid'].iloc[-1] > df_1m['ema_slow'].iloc[-1] and
                df_5m['ema_fast'].iloc[-1] > df_5m['ema_mid'].iloc[-1] > df_5m['ema_slow'].iloc[-1] and
                df_1h['ema_fast'].iloc[-1] > df_1h['ema_mid'].iloc[-1] > df_1h['ema_slow'].iloc[-1]
            )
            short_trend = (
                df_1m['ema_fast'].iloc[-1] < df_1m['ema_mid'].iloc[-1] < df_1m['ema_slow'].iloc[-1] and
                df_5m['ema_fast'].iloc[-1] < df_5m['ema_mid'].iloc[-1] < df_5m['ema_slow'].iloc[-1] and
                df_1h['ema_fast'].iloc[-1] < df_1h['ema_mid'].iloc[-1] < df_1h['ema_slow'].iloc[-1]
            )

            price_up_5m = last_5m['close'] > prev_5m['close']
            price_down_5m = last_5m['close'] < prev_5m['close']
            price_up_1m = last_1m['close'] > prev_1m['close']
            price_down_1m = last_1m['close'] < prev_1m['close']

            price_above_emas = (
                last_1m['close'] > last_1m['ema_fast'] and
                last_1m['close'] > last_1m['ema_mid'] and
                last_1m['close'] > last_1m['ema_slow']
            )
            price_below_emas = (
                last_1m['close'] < last_1m['ema_fast'] and
                last_1m['close'] < last_1m['ema_mid'] and
                last_1m['close'] < last_1m['ema_slow']
            )

            rsi_val = last_1m['rsi']
            atr_val = last_1m['atr']
            # Fetch more 5m bars so ADX has history
            df_5m = fetch_ohlcv(SYMBOL, '5m', limit=300)  # ~25 hours
            df_5m = calculate_adx(df_5m, period=ADX_PERIOD)

            # ADX values (use last closed bar to avoid flicker)
            adx_idx  = -2 if USE_LAST_CLOSED and len(df_5m) >= 2 else -1
            adx_prev = df_5m['ADX'].iloc[adx_idx-1] if len(df_5m) >= 3 else np.nan
            adx_val  = df_5m['ADX'].iloc[adx_idx]
            di_plus  = df_5m['+DI'].iloc[adx_idx]
            di_minus = df_5m['-DI'].iloc[adx_idx]
            adx_rising = adx_val > adx_prev

            # --- NaN guard ---
            if pd.isna(adx_val) or pd.isna(adx_prev) or pd.isna(di_plus) or pd.isna(di_minus):
                logging.info("ADX/DI not ready (insufficient history), skipping iteration.")
                time.sleep(60)
                continue

            # 1m overextension vs EMA20 (use last closed 1m bar)
            last_1m_closed = df_1m.iloc[-2]
            ema_fast_1m    = df_1m['ema_fast'].iloc[-2]
            stretch = (last_1m_closed['close'] - ema_fast_1m) / ema_fast_1m  # + = above EMA, - = below

            # =============================
            # --- NEW: EMA Filters block ---
            # =============================
            skip_trade = False
            skip_reason = None  # collect the reason instead of logging immediately

            # 1. Distance Filter
            if USE_EMA_DISTANCE_FILTER:
                dist_ema100 = abs(last_1m['close'] - last_1m['ema_mid']) / last_1m['ema_mid']
                dist_ema200 = abs(last_1m['close'] - last_1m['ema_slow']) / last_1m['ema_slow']
                if dist_ema100 < EMA_DISTANCE_THRESHOLD or dist_ema200 < EMA_DISTANCE_THRESHOLD:
                    skip_reason = f"too close to EMA100/200 (dist100={dist_ema100:.4f}, dist200={dist_ema200:.4f})"
                    skip_trade = True

            # 2. Breakout Confirmation
            if USE_EMA_BREAKOUT_FILTER and not skip_trade:
                if long_trend and last_1m['close'] <= last_1m['ema_mid']:
                    skip_reason = "LONG — not closed above EMA100."
                    skip_trade = True
                elif short_trend and last_1m['close'] >= last_1m['ema_mid']:
                    skip_reason = "SHORT — not closed below EMA100."
                    skip_trade = True

            # 3. ATR + EMA Overlap Filter
            if USE_ATR_EMA_OVERLAP_FILTER and not skip_trade and entry_price is not None and atr_val is not None:
                stop_distance = atr_val * ATR_STOP_MULTIPLIER
                atr_stop_long = entry_price - stop_distance
                atr_stop_short = entry_price + stop_distance

                if long_trend and (min(last_1m['ema_mid'], last_1m['ema_slow']) <= atr_stop_long <= max(last_1m['ema_mid'], last_1m['ema_slow'])):
                    skip_reason = "LONG — ATR stop overlaps EMA100/200 zone."
                    skip_trade = True
                elif short_trend and (min(last_1m['ema_mid'], last_1m['ema_slow']) <= atr_stop_short <= max(last_1m['ema_mid'], last_1m['ema_slow'])):
                    skip_reason = "SHORT — ATR stop overlaps EMA100/200 zone."
                    skip_trade = True

            # 4. EMA Squeeze Filter
            if USE_EMA_SQUEEZE_FILTER and not skip_trade:
                squeeze_100_200 = abs(last_1m['ema_mid'] - last_1m['ema_slow']) / last_1m['ema_mid']
                if squeeze_100_200 < EMA_SQUEEZE_THRESHOLD:
                    skip_reason = f"EMA100/200 are squeezing together (dist={squeeze_100_200:.4f})"
                    skip_trade = True

            # =============================
            # --- NEW: Regime / Stop Hunt Filters ---
            # =============================
            if USE_REGIME_FILTER and not skip_trade:
                # ATR rising check
                recent_atr = df_1m['atr'].rolling(ATR_LOOKBACK).mean().iloc[-ATR_TREND_WINDOW:]
                atr_rising = recent_atr.iloc[-1] > recent_atr.iloc[0] * ATR_MIN_MULTIPLIER

                # R² trend check
                r2_val = calc_r2(df_1m['close'], 30)

                if not (atr_rising or r2_val > R2_THRESHOLD):
                    skip_trade = True
                    logging.info(f"Skip entry — regime filter failed (ATR rising={atr_rising}, R²={r2_val:.2f})")

            if USE_STOP_HUNT_FILTER and not skip_trade:
                if detect_stop_hunt(last_1m, df_1m):  # df_1m already fetched
                    skip_trade = True
                    logging.info("Skip entry — stop-hunt candle detected.")
            

            if USE_VOLATILITY_SHOCK_FILTER and not skip_trade:
                # how much did last candle move compared to ATR?
                move = abs(last_1m['close'] - last_1m['open'])
                atr_now = df_1m['atr'].iloc[-1]
                move_ratio = move / (atr_now + 1e-9)
                direction = "UP" if last_1m['close'] > last_1m['open'] else "DOWN"

                if move_ratio > BIG_MOVE_MULTIPLIER:
                    skip_trade = True
                    skip_until = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
                    logging.info(f"🚨 Volatility shock detected! {direction} move {move_ratio:.2f}×ATR → skipping new trades for {COOLDOWN_MINUTES} min.")


            # Final check → block trade here also
            if skip_trade:
                time.sleep(60)
                continue

            # Final check → block the trade
            if skip_trade:
                logging.info(f"Skip entry — {skip_reason}")
                time.sleep(60)
                continue


            if entry_order:
                status = get_order_status(entry_order['id'])
                if status == 'closed':  # Entry filled
                    print(f"Entry order filled: {entry_order['id']}")
                    
                    
                    
                    
                    for attempt in range(10):  # Retry up to 5 times
                        position = get_open_position()
                        print(f"Attempt {attempt+1}: position = {position}")
                        if position:
                            break
                        time.sleep(3)  # Wait 1 second between retries

                    if not position:
                        print("⚠️ Entry filled but no open position detected. Possibly closed by TP. Skipping TP placement.")
                        entry_order = None  # Clear entry order so we don’t loop forever
                        continue  # Go to next loop iteration
                    # Place TP limit order
                    position = get_open_position()
                    if position:
                        position_side = position.get('side')
                        position_size = abs(position.get('size', position.get('contracts', 0)))
                        entry_price = position.get('entryPrice')
                        
                        
                        
                        if entry_price is None or atr_val is None:
                            print(f"⚠️ Missing data for TP calculation: entry_price={entry_price}, atr_val={atr_val}")
                        else:
                            if position_side == 'long':
                                take_profit_price = float(entry_price) + atr_val * ATR_MULTIPLIER
                                tp_order = place_limit_order('sell', position_size, take_profit_price, reduce_only=True)
                            elif position_side == 'short':
                                take_profit_price = float(entry_price) - atr_val * ATR_MULTIPLIER
                                tp_order = place_limit_order('buy', position_size, take_profit_price, reduce_only=True)
                            
                            # >>> ADD: set FIXED ATR stop right after fill (non-trailing)
                            atr_value_for_sl = df_1m['atr'].iloc[-1]   # already calculated
                            stop_distance = atr_value_for_sl * ATR_STOP_MULTIPLIER



                            if position_side == 'long':
                                stop_loss_price = float(entry_price) - stop_distance
                                logging.info(
                                    f"🔒 FIXED ATR SL set for LONG at {stop_loss_price:.2f} "
                                    f"(entry={entry_price:.2f}, ATR={atr_value_for_sl:.2f}, x{ATR_STOP_MULTIPLIER}, take_profit={take_profit_price:.2f})"
                                )

                            elif position_side == 'short':
                                stop_loss_price = float(entry_price) + stop_distance
                                logging.info(
                                    f"🔒 FIXED ATR SL set for SHORT at {stop_loss_price:.2f} "
                                    f"(entry={entry_price:.2f}, ATR={atr_value_for_sl:.2f}, x{ATR_STOP_MULTIPLIER}, take_profit={take_profit_price:.2f})"
                                )

                        # Clear entry order once TP is placed
                        entry_order = None
                    else:
                        print("No position found after entry order filled, cannot place TP order")
                    

                elif status == 'canceled' or status is None:
                    print("Entry order canceled or unknown status, clearing order")
                    entry_order = None

            # Prevent re-entry if entry is in progress
            if entry_order:
                logging.info("Waiting for entry order to fill...")
                time.sleep(5)
                continue

            

            position = get_open_position()
            if position is None:
                position_side = None
                position_size = 0
                entry_price = None
            else:
                position_side = position.get('side')
                position_size = abs(position.get('size', position.get('contracts', 0)))
                entry_price = position.get('entryPrice')
            
            position = get_open_position()

            if position:
                position_side = position.get('side')
                position_size = abs(position.get('size', position.get('contracts', 0)))
                entry_price = position.get('entryPrice')
                logging.info(f"✅ In a {position_side} position, size={position_size}. Checking exit logic...")
            else:
                position_side = None
                position_size = 0
                entry_price = None
            # === ATR Stop Loss ===
            if position_side and entry_price:
                atr_value = df_1m['atr'].iloc[-1]   # reuse instead of fetching
                stop_distance = atr_value * ATR_STOP_MULTIPLIER


                if position_side.lower() == "long":
                    atr_stop = entry_price - stop_distance
                elif position_side.lower() == "short":
                    atr_stop = entry_price + stop_distance
                else:
                    atr_stop = None

                if atr_stop:
                    logging.info(f"📉[INFO] ATR model level (not active SL): {atr_stop:.2f}")
                    logging.info(f"🔒 Fixed SL (protective): {stop_loss_price:.2f}")





            # 2. Check if TP order exists and if filled
            if tp_order:
                if status == 'closed':
                    print(f"Take profit order filled: {tp_order['id']}")
                    tp_order = None
                    take_profit_price = None
                    # After TP filled, ready for new entries next loop
                elif status == 'canceled' or status is None:
                    print("TP order canceled or unknown status, clearing order")
                    tp_order = None
                    take_profit_price = None
                    if stop_loss_order:
                            status = get_order_status(stop_loss_order['id'])
                            if status == 'closed':
                                print(f"Stop loss order filled: {stop_loss_order['id']}")
                            stop_loss_order = None
                            take_profit_price = None  # reset TP price when stopped out

                            # Check if the position is truly closed
                            position = get_open_position()
                            if position is None or position.get('size', 0) == 0:
                                print("Position closed after stop loss.")
                            # Optionally reset any other variables or states here
                            else:
                                print(f"Warning: Position still open after stop loss order filled! Size={position['size']}")
                                # You might want to wait, retry closing, or alert here

                    elif status == 'canceled' or status is None:
                            print("Stop loss order canceled or unknown status, clearing order")
                            stop_loss_order = None


            # 3. Entry logic (only if no open position and no pending entry order)
            if (position_side is None or position_size == 0) and entry_order is None:
                # --- ADX slope gate ---
                if not (adx_val >= 25 and adx_rising):
                    logging.info(f"Skip entry — ADX {adx_val:.2f} (prev {adx_prev:.2f}) not strong+rising.")
                    time.sleep(60)
                    continue

                # --- Overextension filter ---
                stretch = (last_1m['close'] - last_1m['ema_fast']) / last_1m['ema_fast']

                if long_trend and price_above_emas and rsi_val < 70 and price_up_5m and price_up_1m:
                    # DI direction filter (buyers must dominate)
                    di_ok = (di_plus > di_minus + DI_BUFFER)
                    logging.info(f"[DI] Long setup +DI={di_plus:.2f} -DI={di_minus:.2f} ok={di_ok}")


                    if USE_DI_GATE and not di_ok:
                        logging.info("Skip LONG — +DI not in favor.")
                    elif stretch > 0.005:
                        logging.info(f"Skip LONG — overextended above EMA20 ({stretch*100:.2f}%).")
                    else:
                        # STEP 3: Adaptive sizing
                        if USE_ADAPTIVE_SIZING:
                            atr_now = last_1m['atr']
                            atr_median = df_1m['atr'].rolling(100).median().iloc[-2]
                            scale = 1.0
                            if atr_now > atr_median * VOL_SPIKE_MULTIPLIER:
                                scale = MIN_POSITION_SCALE
                                logging.info(f"Adaptive sizing: volatility spike, scaling down to {scale*100:.0f}% size.")
                            qty = calculate_order_quantity(SYMBOL, last_1m['close'], TRADE_USDT * scale, LEVERAGE)
                        else:
                            qty = calculate_order_quantity(SYMBOL, last_1m['close'], TRADE_USDT, LEVERAGE)

                        logging.info(f"[SIZE] px={last_1m['close']:.2f} lev={LEVERAGE} -> qty={qty}")
                        if tp_order:
                            cancel_order(tp_order['id'])
                            tp_order = None
                            take_profit_price = None

                        if qty > 0:
                            # Place entry limit order slightly below current price (e.g., 0.1% below market for long)
                            entry_order = retry_limit_order('buy', qty, get_entry_price_long, post_only=True, reduce_only=False)

                elif short_trend and price_below_emas and rsi_val > 30 and price_down_5m and price_down_1m:
                        #DI direction filter (sellers must dominate)
                        di_ok = (di_minus > di_plus + DI_BUFFER)
                        logging.info(f"[DI] Short setup +DI={di_plus:.2f} -DI={di_minus:.2f} ok={di_ok}")
                        if USE_DI_GATE and not di_ok:
                            logging.info("Skip SHORT — -DI not in favor.")
                        elif -stretch > 0.005:
                            logging.info(f"Skip SHORT — overextended below EMA20 ({-stretch*100:.2f}%).")
                        else:
                            # STEP 3: Adaptive sizing
                            if USE_ADAPTIVE_SIZING:
                                atr_now = last_1m['atr']
                                atr_median = df_1m['atr'].rolling(100).median().iloc[-2]
                                scale = 1.0
                                if atr_now > atr_median * VOL_SPIKE_MULTIPLIER:
                                    scale = MIN_POSITION_SCALE
                                    logging.info(f"Adaptive sizing: volatility spike, scaling down to {scale*100:.0f}% size.")
                                qty = calculate_order_quantity(SYMBOL, last_1m['close'], TRADE_USDT * scale, LEVERAGE)
                            else:
                                qty = calculate_order_quantity(SYMBOL, last_1m['close'], TRADE_USDT, LEVERAGE)

                            logging.info(f"[SIZE] px={last_1m['close']:.2f} lev={LEVERAGE} -> qty={qty}")
                                
                            if tp_order:
                                cancel_order(tp_order['id'])
                                tp_order = None
                                take_profit_price = None

                            if qty > 0:
                                # Place entry limit order slightly above current price (e.g., 0.1% above market for short)
                                entry_order = retry_limit_order('sell', qty, get_entry_price_short, post_only=True, reduce_only=False)
            
            # Check for exit conditions
            # Take Profit Check
            current_price = fetch_last_price(SYMBOL)

            if position_side == 'long' and position_size > 0 and entry_price is not None:   

                # Take Profit Check
                if take_profit_price is not None and current_price >= take_profit_price:
                    print(f"{datetime.now()} - TP hit, closing LONG")
                    
                    if tp_order:
                        cancel_order(tp_order['id'])
                        print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        tp_order = None
                    close_position_market('long', position_size)
                    take_profit_price = None
                    tp_set = False
                    sl_set = False
                #Stop Loss Check
                elif stop_loss_price is not None and last_1m['close'] <= stop_loss_price:
                    print(f"{datetime.now()} - SL hit, closing LONG")
                    if tp_order:
                        cancel_order(tp_order['id'])
                        tp_order = None
                        print(f"{datetime.now()} - Cancelled TP order")

                    close_position_market('long', position_size)
                    print(take_profit_price)
                    take_profit_price = None
                    stop_loss_price = None
                    tp_set = False
                    sl_set = False


                # RSI Exit (overbought) or EMA Exit (trend reversal)
                #if rsi_val > 70:
                    #reason = "RSI exit"
                    #print(f"{datetime.now()} - {reason}, placing stop loss LIMIT for LONG")
                    # Cancel both TP and SL before placing fresh
                    #if tp_order:
                        #try:
                            #cancel_order(tp_order['id'])
                            #print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        #except Exception as e:
                            #print(f"Error cancelling TP: {e}")
                        #tp_order = None

                    #if stop_loss_order:
                        #try:
                            #cancel_order(stop_loss_order['id'])
                            #print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        #except Exception as e:
                            #print(f"Error cancelling SL: {e}")
                        #stop_loss_order = None

                    # Place fresh SL after RSI exit
                    #close_position('long', position_size)
                    #atr_sl_active = False  # Reset ATR SL flag after position closed

                # EMA Exit (trend reversal)
                #elif not long_trend or not price_above_emas:
                    #print(f"{datetime.now()} - EMA exit, refreshing SL")
                    #if tp_order:
                        #try:
                            #cancel_order(tp_order['id'])
                            #print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        #except Exception as e:
                            #print(f"Error cancelling TP: {e}")
                        #tp_order = None

                    #if stop_loss_order:
                        #try:
                            #cancel_order(stop_loss_order['id'])
                            #print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        #except Exception as e:
                            #print(f"Error cancelling SL: {e}")
                        #stop_loss_order = None

                    # Place fresh SL after EMA exit
                    #close_position('long', position_size)
                    #atr_sl_active = False  # Reset ATR SL flag after position closed
                #else:
                    # Neither RSI nor EMA triggered
                    #print(f"{datetime.now()} - Holding LONG (RSI={rsi_val}, Trend={long_trend}, PriceAboveEMAs={price_above_emas})")

               
            else:
                # No position at all → skip stop loss logic completely
                pass

               

                




            if position_side == 'short':
                if take_profit_price is None and position_size > 0 and entry_price is not None and atr_val is not None:
                    pass
                # Take Profit Check
                current_price = fetch_last_price(SYMBOL)

                if take_profit_price is not None and current_price <= take_profit_price:
                    print(f"{datetime.now()} - TP hit, closing SHORT at {last_1m['close']}")
                    pass
                    
                    if tp_order:
                        cancel_order(tp_order['id'])
                        tp_order = None
                    
                    
                    close_position_market('short', position_size)
                    print(take_profit_price)
                    take_profit_price = None
                    stop_loss_price = None
                    tp_set = False
                    sl_set = False


                #Stop Loss Check
                elif stop_loss_price is not None and last_1m['close'] >= stop_loss_price:
                    print(f"{datetime.now()} - SL hit, closing SHORT")
                    if tp_order:
                        cancel_order(tp_order['id'])
                        tp_order = None
                        print(f"{datetime.now()} - Cancelled TP order")

                    close_position_market('short', position_size)
                    take_profit_price = None
                    stop_loss_price = None
                    tp_set = False
                    sl_set = False

                ## RSI Exit (oversold) or EMA Exit (trend reversal)
                #if rsi_val < 30:
                    #reason = "RSI exit" if rsi_val < 30 else "EMA exit"
                    #print(f"{datetime.now()} - {reason}, placing stop loss LIMIT for SHORT")
                    ## Cancel both TP and SL before placing fresh
                    #if tp_order:
                        #try:
                            #cancel_order(tp_order['id'])
                            #print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        #except Exception as e:
                            #print(f"Error cancelling TP: {e}")
                        #tp_order = None

                    #if stop_loss_order:
                        #try:
                            #cancel_order(stop_loss_order['id'])
                            #print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        #except Exception as e:
                            #print(f"Error cancelling SL: {e}")
                        #stop_loss_order = None
                     # Place fresh SL after RSI exit
                    #close_position('short', position_size)

                ## EMA Exit (trend reversal)
                #elif not short_trend or not price_below_emas:
                    #print(f"{datetime.now()} - EMA exit, refreshing SL")
                    #if tp_order:
                        #try:
                            #cancel_order(tp_order['id'])
                            #print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        #except Exception as e:
                            #print(f"Error cancelling TP: {e}")
                        #tp_order = None

                    #if stop_loss_order:
                        #try:
                            #cancel_order(stop_loss_order['id'])
                            #print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        #except Exception as e:
                            #print(f"Error cancelling SL: {e}")
                        #stop_loss_order = None

                    # Place fresh SL after EMA exit
                    #close_position('short', position_size)
                #else:
                    ##neither RSI nor EMA triggered
                    #print(f"{datetime.now()} - Holding SHORT (RSI={rsi_val}, Trend={short_trend}, PriceBelowEMAs={price_below_emas})")
               

                else:
                    # No position at all → skip stop loss logic completely
                    pass

                if position_side and take_profit_price:
                    current_price = fetch_last_price(SYMBOL)

                    #Long TP hit
                    if position_side == 'long' and current_price >= take_profit_price:
                        print(f"⚡ Safety TP triggered — LONG closed at {current_price:.2f} ≥ {take_profit_price:.2f}")
                        
                        close_position_market('long', position_size)
                        take_profit_price = None

                    #Short TP hit
                    elif position_side == 'short' and current_price <= take_profit_price:
                        print(f"⚡ Safety TP triggered — SHORT closed at {current_price:.2f} ≤ {take_profit_price:.2f}")
                        
                        close_position_market('short', position_size)
                        take_profit_price = None        

            

            # Reduce loop delay when in trade
            if position_side:
                time.sleep(5)    # check every 5 seconds when in trade
            else:
                time.sleep(60)   # check once per minute when idle


        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()