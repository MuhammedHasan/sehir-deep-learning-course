import scrapy


class Antoloji(scrapy.Spider):
    name = 'Antoloji'
    url = 'https://www.antoloji.com'
    poet = 'cemal-sureya'

    def start_requests(self):
        yield scrapy.Request('%s/%s/siirleri/' % (self.url, self.poet),
                             self.parse_all)

    def parse_all(self, response):
        for i in range(1, len(response.css('.pagination li')) - 1):
            yield scrapy.Request('%s/%s/siirleri/ara-/sirala-/sayfa-%s/' %
                                 (self.url, self.poet, i), self.parse_list)

    def parse_list(self, response):
        for i in response.css('.poemListBox a::attr(href)').extract():
            yield scrapy.Request(self.url + i, self.parse)

    def parse(self, response):
        with open('poets/%s.txt' % self.poet, 'a') as f:
            text = ''.join(response.css('.pd-text p::text').extract())
            f.write(text)
            yield {'poem': text}
