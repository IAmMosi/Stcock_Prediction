# Stcock Prediction

Hi guys! <br />
This is my code for [Stock Prediction Challenge](https://www.hackerrank.com/challenges/stockprediction/problem). In this code I have tried to implement appropriate and clear code to be easy to read. The main problem is completely explained in [hakcerrank](https://www.hackerrank.com/challenges/stockprediction/problem), and I dont write excessive explanation hear. So, let's go to the code and solution. <br />
In my code, there are some functions and some statistical formula, and I will explain all of them. After that, I will explain my logic in code.Firstly, I explain some statistical subjects. <br />

# Some Statistics
I have used some statistical formulas to have more accurate prediction and BUY and SELL more reliable. <br />
1. SMA (Simple Moving Avreage): <br />
It calculates the average of a selected range of prices by the number of periods in that range. The average is called "moving" because it is plotted on the chart bar by bar, forming a line that moves along the chart as the average value changes. You can find its formula [hear](https://www.investopedia.com/terms/s/sma.asp#:~:text=A%20simple%20moving%20average%20(SMA)%20calculates%20the%20average%20of%20a,of%20periods%20in%20that%20range).
2. EMA (Exponential Moving Avreage): <br />
Exponential Moving Average (EMA) is similar to Simple Moving Average (SMA), measuring trend direction over a period of time. However, whereas SMA simply calculates an average of price data, EMA applies more weight to data that is more current. Because of its unique calculation, EMA will follow prices more closely than a corresponding SMA. You can find its formula [hear](https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp#:~:text=The%20exponential%20moving%20average%20(EMA)%20is%20a%20technical%20chart%20indicator,importance%20to%20recent%20price%20data).
3. RSI (Relative Strength Index): <br />
The Relative Strength Index (RSI), developed by J. Welles Wilder, is a momentum oscillator that measures the speed and change of price movements. The RSI oscillates between zero and 100. Traditionally the RSI is considered overbought when above 70 and oversold when below 30. You can find its formula [hear](https://www.investopedia.com/terms/r/rsi.asp).<br />
Now, I am going to explaine the functions i have wrote in my code.

# Functions
I have wrote functions in my code to solve the problem. Now, take a look on them: <br />
1. def load(): <br />
This function loads input file from website to start work. <br />
2. def save():<br />
This function saves each loaded data from website for future uses.<br />
3. def read_file():<br />
This functoin reads file that is loaded from website.<br />
4. def read_input():<br />
This function turns the input file to string.<br />
5. def process_input():<br />
This function processes input string and spits it to lists of stock names, prices, owned stocks and ... . <br />
6. def update_data():<br />
After each iteration, This function updates the lists of data.<br />
7. def calculate_ema():<br />
In this functoin, I calculate the EMA of each stock.<br />
8. def calculate_rsi():<br />
In this function, I calculate each stock RSI indicator.<br />
9. def calculate_ema_var(data):<br />
In this function, I calculated variance of EMA for each stock.<br />
10. def make_decision(data):<br />
This is the main and most important funtion of code. In this function, I impelemente my strategy to buy and sell stocks and make profit!<br />
11. def do_transactions():<br />
In this function, I do my transactions (buy and sell orders) and make output.<br />

# Logic and Strategy
In this challenge, we have price of stocks for today and last 4 days. So, I cannot use historical data to learn any model. Therefore, I use some statistical metrics to build my strategy. Because I do not have OHLC (open, high, low, close) prices and I just have one price  for each stock for each day, I cannot use much statistical metrics and financial indicators. I use SMA, EMA and RSI to build my strategy and make a profit. I implement a tree like model. The first root of tree is RSI, and it branches tree to two branches. After that, in one branch, EMA branches tree to two branches and in another branch, SMA branches tree to two branches. Finally, there are four leaves for decision-making. In two of them, I decide to buy and in other leaves, I decide to sell. In the end, I set orders for stocks that are qualified for buy or sell.
# Running Code
For running this code, please go [hear](https://www.hackerrank.com/challenges/stockprediction/problem) and copy code in editor, then run!
