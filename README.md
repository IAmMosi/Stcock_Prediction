# Stcock Prediction

Hi guys! <br />
This is my code for [Stock Prediction Challenge](https://www.hackerrank.com/challenges/stockprediction/problem). In this code I have tried to implement approprite and clear code to be easy to read. The main problem is compeletly explained in [hakcerrank](https://www.hackerrank.com/challenges/stockprediction/problem) and I dont write execive explanation hear. So, let's go to the code and solution. <br />
In my code, there are some functions and some statistical formula and I will explain all of them. After that, I will explaine my logic in code. firstly, I explaine some statistical subjects. <br />
# Some Statistics
I have used some statistical formulas to have more accurate prediction and BUY and SELL more reliable. <br />
1. SMA (Simple Moving Avreage): <br />
It calculates the average of a selected range of prices by the number of periods in that range. The average is called "moving" because it is plotted on the chart bar by bar, forming a line that moves along the chart as the average value changes. You can find its formula [hear](https://www.investopedia.com/terms/s/sma.asp#:~:text=A%20simple%20moving%20average%20(SMA)%20calculates%20the%20average%20of%20a,of%20periods%20in%20that%20range).
2. EMA (Exponential Moving Avreage): <br />
Exponential Moving Average (EMA) is similar to Simple Moving Average (SMA), measuring trend direction over a period of time. However, whereas SMA simply calculates an average of price data, EMA applies more weight to data that is more current. Because of its unique calculation, EMA will follow prices more closely than a corresponding SMA. You can find its formula [hear](https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp#:~:text=The%20exponential%20moving%20average%20(EMA)%20is%20a%20technical%20chart%20indicator,importance%20to%20recent%20price%20data).
3. RSI (Relative Strength Index): <br />
The Relative Strength Index (RSI), developed by J. Welles Wilder, is a momentum oscillator that measures the speed and change of price movements. The RSI oscillates between zero and 100. Traditionally the RSI is considered overbought when above 70 and oversold when below 30. You can find its formula [hear](https://www.investopedia.com/terms/r/rsi.asp).
Now, I am going to explaine the functions i have wrote in my code.
# Functions
