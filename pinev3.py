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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set trading pair
SYMBOL = 'ETH/USDT:USDT'

# Configure Bybit API for real trading (use with caution)
exchange = ccxt.bybit({
    'apiKey': 'jUoXF93eP3xF91FUBZ',
    'secret': 'e248gw6I9CLmlWW2SibEL45ZuXJXp3u55Q9R',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'linear''future',
        'recvWindow': 15000, # 10 secs
    }
})
exchange.load_markets()
market = exchange.market(SYMBOL)

# Leverage setting
LEVERAGE = 50
TRADE_USDT = 20  # Amount of USDT to use per trade
ATR_MULTIPLIER = 1.5

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
take_profit_price = None                 # Stores take profit price locally
# ATR Settings
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5   # stop loss = entry ± ATR × multiplier
TIMEFRAME = '1m'  # Main timeframe for signals
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


def calculate_order_quantity(current_price, usdt_amount, leverage, precision, min_qty):
    qty = (usdt_amount * leverage) / current_price
    qty = math.floor(qty * (10 ** precision)) / (10 ** precision)
    return max(qty, min_qty)



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
            limit_price = round(stop_price * 0.9995, 2)
        else:
            # stop loss triggers ABOVE, so set limit slightly above stop
            limit_price = round(stop_price * 1.0005, 2)

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


# --- MAIN STRATEGY LOOP ---
def main():
    set_leverage(SYMBOL, LEVERAGE)
    position = get_open_position()
    logging.info(f"DEBUG: position returned: {position} (type: {type(position)})")
    if position is None:
        logging.info("No open position found.")
        position_side = None
        position_size = 0
        entry_price = None
    else:
        # safe access using get()
        position_side = position.get('side')
        position_size = abs(position.get('size', 0))
        entry_price = position.get('entryPrice')


    # You can now use position_side, position_size, and entry_price safely





    entry_order = None
    take_profit_price = None
    entry_order = None
    tp_order = None
    stop_loss_order = None

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
            df_5m = fetch_ohlcv(SYMBOL, '5m')
            if df_5m.empty:
                logging.warning("5m OHLCV data is empty, skipping iteration.")
                time.sleep(60)
                continue
            df_1h = fetch_ohlcv(SYMBOL, '1h')
            if df_1h.empty:
                logging.warning("1h OHLCV data is empty, skipping iteration.")
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

            atr_val = last_1m['atr']
            rsi_val = last_1m['rsi']

            if pd.isna(atr_val) or pd.isna(rsi_val):
                logging.warning(f"Indicators not ready yet (ATR={atr_val}, RSI={rsi_val}), skipping...")
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
                                take_profit_price = take_profit_price

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
                # Calculate ATR
                df = fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=50)
                df['ATR'] = atr(df, period=ATR_PERIOD)
                atr_value = df['ATR'].iloc[-1]

                stop_distance = atr_value * ATR_STOP_MULTIPLIER

                if position_side.lower() == "long":
                    atr_stop = entry_price - stop_distance
                elif position_side.lower() == "short":
                    atr_stop = entry_price + stop_distance
                else:
                    atr_stop = None

                if atr_stop:
                    logging.info(f"📉 ATR Stop Loss for {position_side}: {atr_stop:.2f}")
                    place_stop_loss_limit(position_side, position_size, atr_stop)





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
                if long_trend and price_above_emas and rsi_val < 70 and price_up_5m and price_up_1m:
                    market = exchange.market(SYMBOL)
                    min_qty = market['limits']['amount']['min']
                    precision = int(market['precision']['amount'])
                    qty = calculate_order_quantity(last_1m['close'], TRADE_USDT, LEVERAGE, precision, min_qty)
                    # Cancel any existing SL and TP orders before placing new entry
                    if stop_loss_order:
                        cancel_order(stop_loss_order['id'])
                        stop_loss_order = None
                    if tp_order:
                        cancel_order(tp_order['id'])
                        tp_order = None
                        take_profit_price = None

                    if qty > 0:
                        # Place entry limit order slightly below current price (e.g., 0.1% below market for long)
                        entry_price = last_1m['close'] * 1.001
                        entry_order = retry_limit_order('buy', qty, get_entry_price_long, post_only=True, reduce_only=False)

                elif short_trend and price_below_emas and rsi_val > 30 and price_down_5m and price_down_1m:
                    market = exchange.market(SYMBOL)
                    min_qty = market['limits']['amount']['min']
                    precision = market['precision']['amount']
                    qty = calculate_order_quantity(last_1m['close'], TRADE_USDT, LEVERAGE, precision, min_qty)
                    # Cancel any existing SL and TP orders before placing new entry
                    if stop_loss_order:
                        cancel_order(stop_loss_order['id'])
                        stop_loss_order = None
                    if tp_order:
                        cancel_order(tp_order['id'])
                        tp_order = None
                        take_profit_price = None

                    if qty > 0:
                        # Place entry limit order slightly above current price (e.g., 0.1% above market for short)
                        entry_price = last_1m['close'] * 1.001
                        entry_order = retry_limit_order('sell', qty, get_entry_price_short, post_only=True, reduce_only=False)

            # Check for exit conditions
            # Take Profit Check
            if position_side == 'long':
                if take_profit_price is None and position_size > 0:    
                    take_profit_price = (position_size and last_1m['close'] + atr_val * ATR_MULTIPLIER)

                # Take Profit Check
                if take_profit_price is not None and last_1m['close'] >= take_profit_price:
                    print(f"{datetime.now()} - TP hit, closing LONG")
                    if stop_loss_order:
                        try:
                            cancel_order(stop_loss_order['id'])
                            print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling SL: {e}")
                        stop_loss_order = None
                    if tp_order:
                        try:
                            cancel_order(tp_order['id'])
                            print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling TP: {e}")
                        tp_order = None
                    close_position('long', position_size)
                    take_profit_price = None

                # RSI Exit (overbought) or EMA Exit (trend reversal)
                if rsi_val > 70:
                    reason = "RSI exit" if rsi_val > 70 else "EMA exit"
                    print(f"{datetime.now()} - {reason}, placing stop loss LIMIT for LONG")
                    # Cancel both TP and SL before placing fresh
                    if tp_order:
                        try:
                            cancel_order(tp_order['id'])
                            print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling TP: {e}")
                        tp_order = None

                    if stop_loss_order:
                        try:
                            cancel_order(stop_loss_order['id'])
                            print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling SL: {e}")
                        stop_loss_order = None

                    # Place fresh SL after RSI exit
                    close_position('long', position_size)

                # EMA Exit (trend reversal)
                elif not long_trend or not price_above_emas:
                    print(f"{datetime.now()} - EMA exit, refreshing SL")
                    if tp_order:
                        try:
                            cancel_order(tp_order['id'])
                            print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling TP: {e}")
                        tp_order = None

                    if stop_loss_order:
                        try:
                            cancel_order(stop_loss_order['id'])
                            print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling SL: {e}")
                        stop_loss_order = None

                    # Place fresh SL after EMA exit
                    close_position('long', position_size)
                else:
                    # Neither RSI nor EMA triggered
                    print(f"{datetime.now()} - Holding LONG (RSI={rsi_val}, Trend={long_trend}, PriceAboveEMAs={price_above_emas})")

                # Guard for SL
                if (position is not None and position_size > 0 
                    and entry_price is not None 
                    and atr_val is not None and not pd.isna(atr_val)):
                    # Continuous ATR stop loss independent of EMA/RSI
                    atr_sl_price = round(entry_price - atr_val * ATR_STOP_MULTIPLIER, 2)

                    # If price goes below ATR SL → exit immediately
                    if last_1m['close'] <= atr_sl_price:
                        print(f"{datetime.now()} - ATR STOP LOSS hit, closing LONG at {atr_sl_price}")

                        # Cancel TP + SL before placing fresh
                        if tp_order:
                            try:
                                cancel_order(tp_order['id'])
                                print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                            except Exception as e:
                                print(f"Error cancelling TP: {e}")
                            tp_order = None

                        if stop_loss_order:
                            try:
                                cancel_order(stop_loss_order['id'])
                                print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                            except Exception as e:
                                print(f"Error cancelling SL: {e}")
                            stop_loss_order = None

                        # Use your existing close_position (LIMIT order)
                        close_position('long', position_size)
            else:
                # No position at all → skip stop loss logic completely
                pass

               

                




            if position_side == 'short':
                if take_profit_price is None and position_size > 0:    
                    take_profit_price = (position_size and last_1m['close'] - atr_val * ATR_MULTIPLIER)
    
                # Take Profit Check
                if take_profit_price is not None and last_1m['close'] <= take_profit_price:
                    print(f"{datetime.now()} - TP hit, closing SHORT")
                    if stop_loss_order:
                        cancel_order(stop_loss_order['id'])
                        stop_loss_order = None
                    if tp_order:
                        cancel_order(tp_order['id'])
                        tp_order = None
                    close_position('short', position_size)
                    take_profit_price = None

                # RSI Exit (oversold) or EMA Exit (trend reversal)
                if rsi_val < 30:
                    reason = "RSI exit" if rsi_val < 30 else "EMA exit"
                    print(f"{datetime.now()} - {reason}, placing stop loss LIMIT for SHORT")
                    # Cancel both TP and SL before placing fresh
                    if tp_order:
                        try:
                            cancel_order(tp_order['id'])
                            print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling TP: {e}")
                        tp_order = None

                    if stop_loss_order:
                        try:
                            cancel_order(stop_loss_order['id'])
                            print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling SL: {e}")
                        stop_loss_order = None
                     # Place fresh SL after RSI exit
                    close_position('short', position_size)

                # EMA Exit (trend reversal)
                elif not short_trend or not price_below_emas:
                    print(f"{datetime.now()} - EMA exit, refreshing SL")
                    if tp_order:
                        try:
                            cancel_order(tp_order['id'])
                            print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling TP: {e}")
                        tp_order = None

                    if stop_loss_order:
                        try:
                            cancel_order(stop_loss_order['id'])
                            print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                        except Exception as e:
                            print(f"Error cancelling SL: {e}")
                        stop_loss_order = None

                    # Place fresh SL after EMA exit
                    close_position('short', position_size)
                else:
                    #neither RSI nor EMA triggered
                    print(f"{datetime.now()} - Holding SHORT (RSI={rsi_val}, Trend={short_trend}, PriceBelowEMAs={price_below_emas})")
                #Guard for SL
                if (position is not None and position_size > 0 
                    and entry_price is not None 
                    and atr_val is not None and not pd.isna(atr_val)):
                # Continuous ATR stop loss independent of EMA/RSI
                    if (position is not None and position_size > 0 
                        and entry_price is not None 
                        and atr_val is not None and not pd.isna(atr_val)):

                        atr_sl_price = round(entry_price + atr_val * ATR_STOP_MULTIPLIER, 2)

                        # If price goes above ATR SL → exit immediately
                        if last_1m['close'] >= atr_sl_price:
                            print(f"{datetime.now()} - ATR STOP LOSS hit, closing SHORT at {atr_sl_price}")

                            # Cancel TP + SL before placing fresh
                            if tp_order:
                                try:
                                    cancel_order(tp_order['id'])
                                    print(f"{datetime.now()} - Cancelled TP {tp_order['id']}")
                                except Exception as e:
                                    print(f"Error cancelling TP: {e}")
                                tp_order = None

                            if stop_loss_order:
                                try:
                                    cancel_order(stop_loss_order['id'])
                                    print(f"{datetime.now()} - Cancelled SL {stop_loss_order['id']}")
                                except Exception as e:
                                    print(f"Error cancelling SL: {e}")
                                stop_loss_order = None

                            # Use your existing close_position (LIMIT order)
                            close_position('short', position_size)
                else:
                    # No position at all → skip stop loss logic completely
                    pass



            

            time.sleep(60)  # run once per minute

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()