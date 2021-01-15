import csv
import http.client
import json

def getMarketData():
    conn = http.client.HTTPSConnection("alpha-vantage.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "b8604d934emsh388e7aff14f4d7ep1eb9efjsn80019256d442",
        'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
    }

    conn.request("GET", "/query?function=GLOBAL_QUOTE&symbol=TSLA", headers=headers)

    res = conn.getresponse()
    data = res.read()
    return data


def updateCSV(date1, open1, high1, low1, close1, volume1):
    fields = [date1, open1, high1, low1, close1, volume1]
    with open(r'TSLA', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def update():
    data = getMarketData()
    jdata = json.loads(data)
    data = jdata["Global Quote"]
    updateCSV(data['07. latest trading day'], data['02. open'], data['03. high'], data['04. low'], data['05. price'], data['06. volume'])

update()

