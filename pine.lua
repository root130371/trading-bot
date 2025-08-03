//@version=5
strategy("My Strategy with 1m and 5m Price Checks", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// === INPUTS ===
ema_fast_len = input.int(20, title="EMA Fast")
ema_mid_len  = input.int(50, title="EMA Mid")
ema_slow_len = input.int(100, title="EMA Slow")
rsi_len      = input.int(14, title="RSI Length")
atr_len      = input.int(14, title="ATR Length")
atr_mult     = input.float(1.5, title="ATR TP Multiplier")

// === INDICATORS ===
ema_fast = ta.ema(close, ema_fast_len)
ema_mid  = ta.ema(close, ema_mid_len)
ema_slow = ta.ema(close, ema_slow_len)
rsi      = ta.rsi(close, rsi_len)
atr      = ta.atr(atr_len)

// === 5m Price Direction Check ===
close_5m_1 = request.security(syminfo.tickerid, "5", close[1])
close_5m_2 = request.security(syminfo.tickerid, "5", close[2])
price_up_5m = close_5m_1 > close_5m_2
price_down_5m = close_5m_1 < close_5m_2

// === 1m Price Direction Check ===
price_up_1m = close > close[1]
price_down_1m = close < close[1]

// === CONDITIONS ===
price_above_emas = close > ema_fast and close > ema_mid and close > ema_slow
price_below_emas = close < ema_fast and close < ema_mid and close < ema_slow

ema_condition_long  = ema_fast > ema_mid and ema_mid > ema_slow
ema_condition_short = ema_fast < ema_mid and ema_mid < ema_slow

// Entry conditions require both 1m and 5m price to be moving up/down respectively
long_entry  = ema_condition_long and price_above_emas and rsi < 70 and price_up_5m and price_up_1m
short_entry = ema_condition_short and price_below_emas and rsi > 30 and price_down_5m and price_down_1m

var float long_tp_price = na
var float short_tp_price = na

if (long_entry)
    long_tp_price := close + atr * atr_mult
    strategy.entry("Long", strategy.long)
if (short_entry)
    short_tp_price := close - atr * atr_mult
    strategy.entry("Short", strategy.short)

// === EXIT CONDITIONS ===
ema_reversal_long_exit  = ema_condition_short and price_below_emas
ema_reversal_short_exit = ema_condition_long and price_above_emas

take_profit_long_hit  = close >= long_tp_price
take_profit_short_hit = close <= short_tp_price

rsi_exit_long  = rsi > 70
rsi_exit_short = rsi < 30

// === APPLY EXITS ===
strategy.close("Long", when=ema_reversal_long_exit or take_profit_long_hit or rsi_exit_long)
strategy.close("Short", when=ema_reversal_short_exit or take_profit_short_hit or rsi_exit_short)

// === PLOTTING ===
plot(ema_fast, color=color.blue, title="EMA Fast")
plot(ema_mid, color=color.orange, title="EMA Mid")
plot(ema_slow, color=color.red, title="EMA Slow")

plotshape(long_entry, location=location.belowbar, color=color.green, style=shape.labelup, title="Buy Signal", text="BUY")
plotshape(short_entry, location=location.abovebar, color=color.red, style=shape.labeldown, title="Sell Signal", text="SELL")
