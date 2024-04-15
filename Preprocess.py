import csv
import datetime

import numpy as np

def ft_haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371 #km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def datetime_processing(datetime_str):
    datetime_split = datetime_str.split(" ")
    d = datetime_split[0]
    t = datetime_split[1]

    date_split = d.split("-")
    time_split = t.split(":")

    dt = datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2]),
                           int(time_split[0]), int(time_split[1]), int(time_split[2]))
    dayOfTheWeek = dt.weekday()

    isMon = 0
    isTue = 0
    isWed = 0
    isThu = 0
    isFri = 0
    isSat = 0

    if dayOfTheWeek == 0:
        isMon = 1
    elif dayOfTheWeek == 1:
        isTue = 1
    elif dayOfTheWeek == 2:
        isWed = 1
    elif dayOfTheWeek == 3:
        isThu = 1
    elif dayOfTheWeek == 4:
        isFri = 1
    elif dayOfTheWeek == 5:
        isSat = 1

    return [isMon, isTue, isWed, isThu, isFri, isSat, int(time_split[0]) * 60 + int(time_split[1])]


def remove_feature(d, col):
    return np.delete(d, col, axis=1)


def preprocess(d):
    # Remove header
    d = np.delete(d, 0, axis=0)

    # Remove id, store_and_fwd_flag
    d = remove_feature(d, 0) # id
    d = remove_feature(d, 8) # flag
    d = remove_feature(d, 2) # drop-off datetime

    return d


def read_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data


def write_csv(filename, d):
    features = ['vendor_id', 'passenger_count', 'trip_duration', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri',
                'isSat', 'time', 'distance']
    with open(filename, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(features)
        write.writerows(d)


data = read_csv("train.csv")
data = preprocess(data)

new_column = []
for i in range(len(data)):
    # dayOfTheWeek and time in minute
    c = datetime_processing(data[i][1])

    # Distance
    c.append(ft_haversine_distance(float(data[i][3]), float(data[i][4]), float(data[i][5]), float(data[i][6])))

    new_column.append(c)

data = np.hstack((data, np.array(new_column)))
data = remove_feature(data, 1)  # pickup time
data = remove_feature(data, 2)  # pickup long
data = remove_feature(data, 2)  # pickup lat
data = remove_feature(data, 2)  # dropoff long
data = remove_feature(data, 2)  # dropoff lat

write_csv("better_train.csv", data)
