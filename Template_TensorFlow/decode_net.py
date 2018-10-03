#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:37:16 2018

@author: yutong
"""
#def decode_int(data):
#    return list(int(data.split(',')[i].strip()) for i in range(len(data.split(','))))
#
#a  = '[1,2,3|4,5,6|12,11]'
#b = [i for i in a[1:-1].split('|')]
#print(b)
#
#r = []
#for c in b:
#    r.append(decode_int(c))
#
#print(r)

def decode_int(data):
    return list(int(data.split(',')[i].strip()) for i in range(len(data.split(','))))

def decode_float(data):
    return list(float(data.split(',')[i].strip()) for i in range(len(data.split(','))))


def decode_net(data):
    results = []
    for result in [i for i in data[1:-1].split('|')]:
        results.append(decode_int(result))
    return results

a = '[1,2,3 |4,5 , 6| 12,11]'
b = '[ 1,2,3 ]'
c = '1, 2,3, 4, 5,6'
d = '0.5, 0.1, 0.05'

print(decode_net(a))
print(decode_net(b))
print(decode_int(c))
print(decode_float(d))


