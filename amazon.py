#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 07:59:26 2021

https://github.com/sergioteula/python-amazon-paapi

Use Amazon data to test models.
- CNN on book images and categories.


@author: justin
"""

# AWS Developer Info
AMAZON_ACCESS_KEY = ''
AMAZON_SECRET_KEY = ''
# Business recipient of online traffic payments
AMAZON_ASSOC_TAG = 'findelworks-22'
AMAZON_COUNTRY = 'AU'

from amazon.paapi import AmazonAPI
amazon = AmazonAPI(AMAZON_ACCESS_KEY, AMAZON_ACCESS_KEY, AMAZON_ACCESS_KEY, AMAZON_COUNTRY)
product = amazon.get_product('B01N5IB20Q')
print(product.title)

help(AmazonAPI)

