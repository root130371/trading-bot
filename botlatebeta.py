import ccxt
import time
import math
import pandas as pd
import numpy as np
import logging
from collections import deque

#CurrentTestingModule
#Testing version botv2alpha2.1
#Trade amount locked in at 10 usd * 50x, overall balance 50 usd.
#3 main logics of trading:Min risk: x trade amount, 10x trading balance, overall time to double account is 20days,
#mid risk logic; x trade amount, 5x trading balance, (This is what we are currently testing), overall time to double  account is around 10 days.
#High Risk logic, Current testing of mid risk will give us more data about this; Use entire balance for trade and increase exponentinally assuming
#no losing trade is of a great amount.
#The potential major problem is data, 6/20/2025 12.00 pm to 6/21/2025 17.00 pm shhows good results with only one minor loss, good results came in 22nd too.
# #1 loss out of 5 trades., only danger is  this 30 hours of data was of luck not consistency, the iniatiation of the 10 usd of the 5x balance
#system will prove this, initial balance is set at 50 usd.
#22nd 2 loss of total 1.5 usd and got 2 wins worth of 5.5 usd
#Only 2 questions remain: Remove RSI take profit and use RSI for stop loss, 2., The ATR tp profit isnt set at trade executing but when
#the time comes to close a position it sets it below the current price, but if this is the max area and price jimps back and the group setting
# proceeds to group 2 price might bounce back and cause a loss showing instability and this has happened 2 times, monitor further logging actions.


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
        'recvWindow': 10000, # 10 secs
    }
})

# Leverage setting
LEVERAGE = 50
USDT_TRADE_AMOUNT = 55  # Amount of USDT to use per trade
TP_ATR_MULTIPLIER = 1.5

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

# Dynamic calculation of the price so the usdt is set on price not order
def calculate_order_quantity(current_price, leverage, usdt_amount, precision, min_qty):
    raw_qty = (usdt_amount * leverage * 0.90) / current_price


    #Convert fload to decimal
    if precision < 1:
        decimal_places = abs(int(round(math.log10(precision))))
    
    else:
        decimal_places = 0
    precision_qty = round(raw_qty, decimal_places) # round to exchange precision
    return max(precision_qty, min_qty)

#Wilder style rma smoothed atr
def calculate_atr_rma(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    #true range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    #Initialize ATR_RMA
    atr = tr.rolling(window=period, min_periods=period).mean().iloc[:period].tolist()

    #First ATR simple average than wilder smoothing
    for i in range(period, len(tr)):
        prev = atr[-1]
        atr.append((prev * (period - 1) + tr.iat[i]) / period)

    df['ATR_RMA'] = atr + [None] * (len(df) - len(atr))
    return df


# Check API connection
def check_api_connection():
    try:
        exchange.fetch_ticker(SYMBOL)
        logging.info("API Connection successful!")
        return True
    except Exception as e:
        logging.error(f"API connection failed: {e}")
        return False

# Fetch OHLCV data
def fetch_ohlcv(symbol, timeframe='1m', limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params = {'category': 'linear'}) 
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Failed to fetch OHLCV: {e}")
        return None

# Calculate EMAs
def calculate_emas(df):
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    return df



# Check trend using EMAs
def check_trend(df):
    last = df.iloc[-1]
    if last['EMA50'] > last['EMA100'] > last['EMA200']:
        return "📈 BULLISH TREND", "long"
    elif last['EMA50'] < last['EMA100'] < last['EMA200']:
        return "📉 BEARISH TREND", "short"
    else:
        return "🔍 NEUTRAL / CONSOLIDATING - Market is neutral or consolidating.", "no_trade"

# Check if price is clearly above or below all EMAs without touching
def price_cleared_emas(df, buffer = 0.0015):
    last = df.iloc[-1]
    close_price = last['close']
    ema50 = last['EMA50']
    ema100 = last['EMA100']
    ema200 = last['EMA200']

    buffer = 0.0015  # 0.1% buffer to avoid touching

    if close_price > max(ema50, ema100, ema200) * (1 + buffer):
        return "long_clear"
    elif close_price < min(ema50, ema100, ema200) * (1 - buffer):
        return "short_clear"
    else:
        return "touching"

# calculate rsi
def calculate_rsi_wilder(df, period=14):
    close = df['close']
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Initialize the first average gain/loss with a simple mean
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use Wilder’s smoothing from the second value onward
    avg_gain = avg_gain.copy()
    avg_loss = avg_loss.copy()

    for i in range(period, len(df)):
        avg_gain.iat[i] = (avg_gain.iat[i-1] * (period - 1) + gain.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i-1] * (period - 1) + loss.iat[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['rsi'] = rsi
    return df

# Open interest and volume histories
open_interest_history = {
    '5m': deque(maxlen=3),
    '15m': deque(maxlen=3)
}

#Trackers for open orders
open_order_id = None
tp_order_id = None

# Fetch open interest
def fetch_open_interest(symbol, interval='5min'):
    try:
        market_id = exchange.market(symbol)['id']
        response = exchange.public_get_v5_market_open_interest({
            'category': 'linear',
            'symbol': market_id,
            'intervalTime': interval
        })
        result_list = response['result']['list']
        if not result_list:
            logging.warning(f"No open interest data returned for {interval}.")
            return None, None

        open_interest = float(result_list[0].get('openInterest'))
        timestamp = int(result_list[0].get('timestamp'))
        return open_interest, timestamp
    except Exception as e:
        logging.error(f"Failed to fetch open interest: {e}")
        return None, None

# Place real order
#Retryorder are batches, retries are the individual attempts.
def place_limit_order_with_retry(symbol, side, quantity, current_price=None, retries=3, retryorder=10, wait_timeout=180, retrydelay=5, delay=10, reduce_only=False):
    for retry_count in range(1, retryorder + 1):  # Outer retry group
        for attempt in range(1, retries + 1):  # Inner retry attempts
            try:
                # Fetch updated price each retry
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']

                # Adaptive offset logic
                base_offset = 0.01 / 100
                adjusted_offset = max(0.01 / 100, base_offset - ((attempt - 1) * (0.005 / 100)))
                price = round(current_price * (1 - adjusted_offset), 2) if side == 'buy' else round(current_price * (1 + adjusted_offset), 2)

                logging.info(f"[Group {retry_count}, Attempt {attempt}/{retries}] LIMIT {side.upper()} order: {quantity} {symbol} @ {price}")
                response = exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=quantity,
                    price=price,
                    params={"time_in_force": "PostOnly", "reduce_only": reduce_only}
                )

                order_id = response['id']
                logging.info(f"Submitted order ID: {order_id}")
                time.sleep(5)

                # Wait for order to fill
                start_time = time.time()
                while time.time() - start_time < wait_timeout:
                    order_status = exchange.fetch_order(order_id, symbol, params={"acknowledged": True})
                    status = order_status.get('status', 'open')
                    filled = order_status.get('filled', 0)

                    logging.info(f"Order status: {status}, Filled: {filled}/{quantity}")

                    if status == 'closed' or filled >= quantity:
                        logging.info("✅ Limit order filled successfully")

                        fee = order_status.get('fee', {})
                        if fee:
                            fee_cost = fee.get('cost', 0)
                            logging.info(f"[FEE INFO] Order fee: {fee_cost} {fee.get('currency', 'USDT')}")
                        else:
                            fee_cost = current_price * quantity * 0.0006
                            logging.warning(f"⚠️ Fee not found. Estimated fee: {fee_cost:.6f} USDT")

                        return order_status, fee_cost, filled

                    time.sleep(5)

                logging.warning("⏰ Order not filled in time. Cancelling...")
                exchange.cancel_order(order_id, symbol)

            except Exception as e:
                logging.error(f"[Group {retry_count}, Attempt {attempt}/{retries}] Error placing or managing order: {e}")
                time.sleep(retrydelay)

        logging.warning(f"❌ Retry group {retry_count} failed. Waiting {retrydelay}s before next group...")
        time.sleep(retrydelay)

    logging.error("❌ All retry attempts failed.")
    return None, 0, 0

            

# Close position
def close_position(symbol, side, quantity, current_price):
    try:
        order_response, fee_cost, filled = place_limit_order_with_retry(
            symbol=symbol,
            side=side,
            quantity=quantity,
            current_price=current_price,
            retries=3,
            retryorder=9,
            wait_timeout=180,   
            retrydelay=5,
            delay=10,
            reduce_only=True
        )
        if order_response is not None:
            logging.info(f"[LIMIT ORDER] Close position with {side.upper()} {quantity} {symbol}")
        else:
            logging.warning("Limit exit order failed after retries")

        return order_response, fee_cost
    except Exception as e:
        logging.error(f"Failed to close position: {e}")
        return None, 0
        

# Main trading logic
def trade():
    start_time = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
    log_data_df = pd.DataFrame(columns=["timestamp", "closing price", "EMA", "OI", "PnL"])
    if not check_api_connection():
        return

    set_leverage(SYMBOL, LEVERAGE)

    position = None
    entry_price = None
    entry_fee_cost = 0
    exit_fee_cost = 0
    funding_fee = 0
    entry_time = None
    #last_trend = None #long or short
    #trade_executed_in_trend = False
    #trend_cycle_id = 0
    #last_trend_cycle_traded = 1 #make sure first trade happens
    #last_trend_traded = None

    while True:
        df = fetch_ohlcv(SYMBOL, '1m')
        if df is None:
            continue

        df = calculate_emas(df)
        trend_text, ema_condition = check_trend(df)
        cleared_condition = price_cleared_emas(df, buffer=0.0015)
        current_price = df['close'].iloc[-1]

        

        df = calculate_rsi_wilder(df)
        # print(df)
        rsi_value = df['rsi'].iloc[-1] 
        # print(rsi_value)
        df = calculate_atr_rma(df, period=14)
        atr_value = df['ATR_RMA'].iloc[-1]

        oi_5m, _ = fetch_open_interest(SYMBOL, interval='5min')
        oi_15m, _ = fetch_open_interest(SYMBOL, interval='15min')
        if None in (oi_5m, oi_15m):
            continue

        open_interest_history['5m'].append(oi_5m)
        open_interest_history['15m'].append(oi_15m)
        oi_5m_list = list(open_interest_history['5m'])
        oi_15m_list = list(open_interest_history['15m'])

        oi_increasing = (
            len(oi_5m_list) == 2 and
            len(oi_15m_list) == 2 and
            oi_5m_list[1] > oi_5m_list[0] and
            oi_15m_list[1] > oi_15m_list[0]
        )

        oi_decreasing = (
            len(oi_5m_list) == 2 and 
            len(oi_15m_list) == 2 and
            oi_5m_list[1] < oi_5m_list[0] and
            oi_15m_list[1] < oi_15m_list[0]

        )



        if oi_increasing:
            oi_condition = "Increase (5m+15m)"
        elif oi_decreasing:
            oi_condition = "Decrease (5m+15)"
        else:
            oi_condition = "Open interest condition not fulfilled"
        
        market = exchange.market(SYMBOL)
        min_qty = market['limits']['amount']['min']
        precision = market['precision']['amount']
        
        #Calculate quantity before trade dynamically
        quantity = calculate_order_quantity(current_price, USDT_TRADE_AMOUNT, LEVERAGE, precision, min_qty)

        if quantity < 0.01:
            logging.warning(f"Calculated quantity {quantity} is below minimum allowed by Bybit. Skipping trade.")
            time.sleep(60)
            continue

        price = df['close'].iloc[-1]    
        ema50 = df['EMA50'].iloc[-1]
        ema100 = df['EMA100'].iloc[-1]
        ema200 = df['EMA200'].iloc[-1]
        buffer = 0.0015
        #pricecheck if its too close to the emas
        close_to_ema50 = abs(price - ema50) / ema50 < buffer
        close_to_ema100 = abs(price - ema100) / ema100 < buffer
        close_to_ema200 = abs(price - ema200) / ema200 < buffer

        pausetrend = close_to_ema50 or close_to_ema100 or close_to_ema200


        # log data into excel file
        log_data = {
            "timestamp": df['timestamp'].iloc[-1],
            "closing price": current_price,
            "EMA": ema_condition,
            "OI condition": oi_condition,
            "RSI": rsi_value
        }

        if len(log_data_df) == 0:
            print("Empty dataframe")
            log_data_df = pd.DataFrame([log_data])
        else:
            log_data_df = pd.concat([log_data_df, pd.DataFrame([log_data])], ignore_index=True)
        log_data_df.to_csv(f"logs-{start_time}.csv", index=False)

        if not position:
            if ema_condition == 'long' and pausetrend:
                logging.info("PAUSETREND: Weakening bullish trend detected. No trade executed.")
                time.sleep(60)
                continue
            if ema_condition == 'short' and pausetrend:
                logging.info("PAUSETREND: Weakening bearish trend detected. No trade executed.")
                time.sleep(60)
                continue
            #if ema_condition == last_trend_traded:
                #logging.info(f"Already traded this trend direction ({ema_condition.upper()})")
                #time.sleep(60)
                #continue
            #trend execution limiter
            current_trend = ema_condition if  ema_condition in ['long', 'short']else None
            #if current_trend and current_trend: != last_trend
                #trend_cycle_id += 1
                #last_trend = current_trend
                #logging.info(f"New trend detected: {current_trend.upper()} (Cycle ID: {trend_cycle_id})")
                #last_trend_traded = None
            
            #if trend_cycle_id == last_trend_cycle_traded:
                #logging.info('Trade already executed in this trend, waiting for trend change')
                #time.sleep(60)
                #continue

            if ema_condition == 'long' and cleared_condition == 'long_clear' and oi_increasing:
                if rsi_value >= 70:
                    logging.info(f"RSI too high for entry: {rsi_value:.2f}. Skipping Trade.")
                    time.sleep(60)
                    continue 
                #Guard
                if position:
                    logging.warning("Already in position, skipping new LONG entry.")
                    time.sleep(60)
                    continue


                logging.info(f"Opening LONG | EMA50: {ema50:.2f}, EMA100: {ema100:.2f}, EMA200: {df['EMA200'].iloc[-1]:.2f}")

                #Submit position before filled
                position = 'long'
                order_response, entry_fee_cost, filled = place_limit_order_with_retry(
                    SYMBOL, 'buy', quantity, current_price=current_price
                )
                if order_response is None or filled ==0:
                    logging.error("Order failed,skipping this cycle")
                    position = None
                    time.sleep(60)
                    continue
                
                entry_price = current_price
                entry_time = df['timestamp'].iloc[-1]
                #last_trend_cycle_traded = trend_cycle_id
                #last_trend_traded = 'long'
                take_profit_price = entry_price + TP_ATR_MULTIPLIER * atr_value
                try:
                    tp_order = exchange.create_limit_order(
                        symbol=SYMBOL,
                        side='sell',
                        amount=quantity,
                        price=take_profit_price,
                        params={
                            "reduceOnly": True

                        }
                    )
                    logging.info(
                        f"Set TP @ {take_profit_price:.2f} "
                        f"(ATR={atr_value:.2f}, R={TP_ATR_MULTIPLIER})"

                )
                except Exception as e:
                    logging.error(f"Failed to place TP order: {e}")
                    entry_time = df['timestamp'].iloc[-1]
                        
                else:
                    logging.warning("Long order not filled, skipping update")
                    time.sleep(60)
                    continue
                entry_time = df['timestamp'].iloc[-1]
            
            
            
            elif ema_condition == 'short' and cleared_condition == 'short_clear' and oi_increasing:
                if rsi_value <= 30:
                    logging.info(f"RSI too low for entry: {rsi_value:.2f}.Skipping Trade")
                    time.sleep(60)
                    continue

                #Guard
                if position:
                    logging.warning("Already in position skipping new short entry")
                    time.sleep(60)
                    continue
                logging.info(f"Opening SHORT | EMA50: {ema50:.2f}, EMA100: {ema100:.2f}, EMA200: {df['EMA200'].iloc[-1]:.2f}")

                #Set position Before Submit
                position = 'short'
                order_response, entry_fee_cost, filled = place_limit_order_with_retry(
                    SYMBOL, 'sell', quantity, current_price=current_price
                )
                if order_response is None or filled == 0:
                    logging.error("Order failed.Skipping this cycle.")
                    position = None
                    time.sleep(60)
                    continue
                
                entry_price = current_price
                entry_time = df['timestamp'].iloc[-1]
                #last_trend_cycle_traded = trend_cycle_id
                #last_trend_traded = 'short'
                take_profit_price = entry_price - TP_ATR_MULTIPLIER * atr_value
                try:
                    tp_order = exchange.create_limit_order(
                        symbol=SYMBOL,
                        side='buy',
                        amount=quantity,
                        price=take_profit_price,
                        params={
                            "reduceOnly": True

                        }
                    )
                    logging.info(
                        f"Set TP REDUCEONLY @ {take_profit_price:.2f} "
                        f"(ATR={atr_value:.2f}, R={TP_ATR_MULTIPLIER})"

                    )
                except Exception as e:
                    logging.error(f"Failed to place SHORT TP order: {e}")
                    entry_time = df['timestamp'].iloc[-1]
                    



                    
                else:
                    logging.warning("Short not filled, skipping position update")
                    time.sleep(60)
                    continue
        else:
            reason = None
            price_under_all_emas = current_price < ema50 and current_price < ema100 and current_price < df['EMA200'].iloc[-1]
            price_above_all_emas = current_price > ema50 and current_price > ema100 and current_price > df['EMA200'].iloc[-1]
            ema_reversal = (position == 'long' and ema_condition == 'short' and price_under_all_emas) or (position == 'short' and ema_condition == 'long' and price_above_all_emas)
            atr_tp = (
                (position == 'long' and current_price >= take_profit_price) or 
                (position == 'short' and current_price <= take_profit_price)
            )        

            
            if ema_reversal:
                reason = "EMA reversal against position with price under or overPrice: {current_price:.2f}, EMA50: {ema50:.2f}, EMA100: {ema100:.2f}, EMA200: {df['EMA200'].iloc[-1]:.2f}"
            elif atr_tp:
                reason = f"ATR×{TP_ATR_MULTIPLIER} TP hit at {current_price:.2f} (target {take_profit_price:.2f})"
            elif position == 'long':
                if rsi_value > 70:
                    reason = f"RSI: {rsi_value:.2f} overbough, take profit"
                elif rsi_value < 30 and price_under_all_emas:
                    reason = f"RSI: {rsi_value:.2f} oversold, stop loss + price under all emas"

            elif position == 'short':
                if rsi_value > 70 and price_above_all_emas:
                    reason = f"RSI: {rsi_value:.2f} overbought, stop loss + price over all emas"
                elif rsi_value < 30:
                    reason = f"RSI: {rsi_value:.2f} oversold, take profit"

            if reason:
                try:
                    open_orders = exchange.fetch_open_orders(SYMBOL)
                    for order in open_orders:
                        if order.get('reduceOnly', False):
                            exchange.cancel_order(order['id'], SYMBOL)
                            logging.info(f"Cancelled reduce only tp (ID: {order['id']})")
                except Exception as e:
                    logging.warning(f"Failed to cancel TP order: {e}")
                side = 'sell' if position == 'long' else 'buy'
                logging.info(f"Closing {position.upper()} position due to: {reason} | EMA50: {ema50:.2f}, EMA100: {ema100:.2f}, EMA200: {df['EMA200'].iloc[-1]:.2f}")
                order_response, exit_fee_cost = close_position(SYMBOL, side, quantity, current_price=current_price)
                if order_response is not None:
                    exit_price = current_price
                    exit_time = df['timestamp'].iloc[-1]
                    funding_fee = 0
                    # Calculate PnL
                    gross_pnl = (exit_price - entry_price) * quantity if position == 'long' else (entry_price - exit_price) * quantity
                    net_pnl = gross_pnl - entry_fee_cost - entry_fee_cost - funding_fee
                    logging.info(f"[TRADE RESULT] {position.upper()} trade PnL: {net_pnl:.4f} USDT (Gross: {gross_pnl:.4f}, Fees: {entry_fee_cost + exit_fee_cost:.4f})")
                    # Log to CSV
                    trade_result = {
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": quantity,
                        "gross_pnl": gross_pnl,
                        "entry_fee": entry_fee_cost,
                        "exit_fee": exit_fee_cost,
                        "funding_fee": funding_fee,
                        "net_pnl": net_pnl
                    }

                    pd.DataFrame([trade_result]).to_csv("trade_pnls.csv", mode='a', header=not pd.io.common.file_exists("trade_pnls.csv"), index=False)
                    position = None
                    entry_price = None
                else:
                    logging.error("❌ Could not close the position. Skipping trade logging to prevent crash.")
                    # Force reset
                    logging.warning("Resetting position to none")
                    position = None
                    entry_price = None
            else:
                logging.info(f"Holding {position.upper()} position. No exit condition met.")
        time.sleep(60)




if __name__ == "__main__":
    trade()
