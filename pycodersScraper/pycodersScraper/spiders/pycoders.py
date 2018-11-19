# -*- coding: utf-8 -*-
import scrapy


class PycodersSpider(scrapy.Spider):
    name = 'pycoders'
    allowed_domains = ['pycoders.com']
    start_urls = ['http://pycoders.com/']

    def parse(self, response):
        pass
