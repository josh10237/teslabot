import csv
import http.client

data = {
    "Global Quote": {
        "01. symbol": "TSLA",
        "02. open": "852.7600",
        "03. high": "860.4700",
        "04. low": "832.0000",
        "05. price": "854.4100",
        "06. volume": "32692271",
        "07. latest trading day": "2021-01-13",
        "08. previous close": "849.4400",
        "09. change": "4.9700",
        "10. change percent": "0.5851%"
    }
}


# print(data['Global Quote'])


def getMarketData():
    conn = http.client.HTTPSConnection("alpha-vantage.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "b8604d934emsh388e7aff14f4d7ep1eb9efjsn80019256d442",
        'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
    }

    conn.request("GET", "/query?function=GLOBAL_QUOTE&symbol=TSLA", headers=headers)

    res = conn.getresponse()
    data = res.read()
    return data['Global Quote']


def updateCSV(date1, open1, high1, low1, close1, volume1):
    changes = [
        ['Date', date1],
        ['Open', open1],
        ['High', high1],
        ['Low', low1],
        ['Close', close1],
        ['Volume', volume1],
    ]

    with open('TSLA_DATA.csv', 'ab') as f:
        writer = csv.writer(f)
        writer.writerows(changes)



with open('TSLA_DATA.csv', 'rb') as f:
    reader = csv.reader(f)
    count = 0
    for row in reader:
        # row is a list of strings
        # use string.join to put them together
        print(', '.join(row))

print("before")
updateCSV("2021-01-13", 831.0000, 868.000, 827.340, 849.440, 849.440)
print("after")
with open('TSLA_DATA.csv', 'rb') as f:
    reader = csv.reader(f)
    count = 0
    for row in reader:
        # row is a list of strings
        # use string.join to put them together
        print(', '.join(row))