import json
import numpy as np
import math
import sys
import statistics
import random

from typing import Dict, List

first_time = True


def load():
    data = {}
    try:
        with open("data.json", "r") as f:
            content = f.read()
            data = json.loads(content)
    except:
        pass

    return data


def read_input():
    content = ""
    for line in sys.stdin:
        content += line

    content = content.split("\n")
    return content


def read_file(n):
    with open("data/%s" % n) as f:
        c = f.read().split("\n")
    return c


def process_input(content):
    first_line = True

    data = {"own_data": {}, "stocks": {}}

    for line in content:
        if first_line:
            d = line.split(' ')
            data["own_data"]["remaining_money"] = float(d[0])
            data["own_data"]["remaining_stocks"] = int(d[1])
            data["own_data"]["remaining_days"] = int(d[2])
            first_line = False
        else:
            name, owned, last_5_days_price = line.split(" ", 2)
            data['stocks'][name] = {
                "owned": int(owned),
                "prices": [float(p) for p in last_5_days_price.split(" ")],
                "sma": [],
                "ema": [],
                "rsi": [],
                "sma_variance": [],
                "ema_variance": [],
            }
    return data


def update_data(data, new_data):
    if not data:
        data = new_data
    else:
        for stock_name, d in new_data['stocks'].items():
            data['stocks'][stock_name]['prices'].append(
                d['prices'][-1]
            )
            data['stocks'][stock_name]['owned'] = d['owned']

        data["own_data"] = new_data["own_data"]

    for stock_name, d in data['stocks'].items():
        d['sma'].append(statistics.mean(d['prices'][-5:]))
        d['ema'].append(calculate_ema(d))
        d['rsi'] = calculate_rsi(d)
        d['sma_variance'].append(np.std(d['prices'][-5:]))
        d['ema_variance'].append(calculate_ema_var(d))

    return data


def calculate_ema(stock_data, days=5):
    k = 2 / (days + 1)

    if len(stock_data['ema']) == 0:
        yema = statistics.mean(stock_data['prices'])
    else:
        yema = stock_data['ema'][-1]

    ema = (stock_data['prices'][-1] * k) + (yema * (1-k))
    return ema


def calculate_rsi(stock_data):
    gain = []
    loss = []
    for i in range(len(stock_data['prices'])):
        if i == 0:
            gain.append(stock_data['prices'][i])
            loss.append(stock_data['prices'][i])
            continue

        d = stock_data['prices'][i] - stock_data['prices'][i-1]
        if d > 0:
            gain.append(d)
        else:
            gain.append(0)

        if d < 0:
            loss.append(abs(d))
        else:
            loss.append(0)

    avg_gain = []
    avg_gain.append(statistics.mean(gain[:5]))
    for i in range(5, len(gain)):
        avg_gain.append(((4 * avg_gain[i-5]) + gain[i]) / 5)

    avg_loss = []
    avg_loss.append(statistics.mean(loss[:5]))
    for i in range(5, len(loss)):
        avg_loss.append(((4 * avg_loss[i-5]) + loss[i]) / 5)

    rs = np.divide(avg_gain, avg_loss)
    rsi = []
    for e in rs:
        rsi.append(100 - (100 / (1 + e)))

    return rsi


def calculate_ema_var(data):
    return math.sqrt(sum(
        [((x - data['ema'][-1])**2) for x in data['prices'][-5:]]
    )/4)
    

def make_decision(data):
    result = {"number_of_transactions": 0, "transactions": []}
    if data["own_data"]["remaining_days"] == 1:
        for stock_name, d in data['stocks'].items():
            if d['owned'] > 0:
                result['number_of_transactions'] += 1
                result['transactions'].append({"name": stock_name, "amount": -d['owned']})
    else:
        sells = []
        buys = []
        for stock_name, d in data['stocks'].items():
            if d['rsi'][-1] > 30:
                if d['prices'][-1] > d['ema'][-1] - (0.5 * d['ema_variance'][-1]):
                    # 25 % chance to decrease the price
                    if d['owned'] > 0 and random.random() < 0.25:
                        sells.append({"name": stock_name, "amount": d['owned']})
                else:
                    # 33 % chance to increase the price
                    if random.random() < 0.75:
                        buys.append({"name": stock_name, "amount": 0, "weight": 0.33, "price": d['prices'][-1]})
            else:
                # 63 % chance to increase the price
                if d['prices'][-1] > d['sma'][-1] - (0.5 * d['sma_variance'][-1]):
                    # 52 % chance to decrease the price
                    if d['owned'] > 0 and random.random() < 0.20:
                        sells.append({"name": stock_name, "amount": d['owned']})
                else:
                    # 85 % chance to increase the price
                    if random.random() < 0.80:
                        buys.append({"name": stock_name, "amount": 0, "weight": 0.85, "price": d['prices'][-1]})

        max_weight_stock = None
        for b in buys:
            if data['own_data']['remaining_money'] > b['price'] and max_weight_stock == None:
                max_weight_stock = b
            elif max_weight_stock is not None:
                if b['weight'] > max_weight_stock['weight'] and data['own_data']['remaining_money'] > b['price']:
                    max_weight_stock = b

        for s in sells:
            result['number_of_transactions'] += 1
            result['transactions'].append({"name": s['name'], "amount": -s['amount']})

        if max_weight_stock:
            result['number_of_transactions'] += 1
            result['transactions'].append({"name": max_weight_stock['name'], "amount": int(data['own_data']['remaining_money'] / max_weight_stock['price'])})

    return result


def save(data):
    content = json.dumps(data, indent=4, sort_keys=True)
    with open("data.json", "w") as f:
        f.write(content)

    return content


def do_transactions(number: int, transactions: List[Dict]):
    print(number)
    for t in transactions:
        type = None
        if t['amount'] > 0:
            type = "BUY"
        else:
            type = "SELL"

        print(t['name'], type, abs(t['amount']))


def main():
    data = load()
    content = read_input()
    new_data = process_input(content)
    data = update_data(data, new_data)
    content = save(data)
    
    decision = make_decision(data)
    do_transactions(decision['number_of_transactions'], decision['transactions'])


if __name__ == "__main__":
    main()
