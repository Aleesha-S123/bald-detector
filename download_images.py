from icrawler.builtin import GoogleImageCrawler

# Create bald image folder
bald_crawler = GoogleImageCrawler(storage={'root_dir': 'dataset/bald'})
bald_crawler.crawl(keyword='bald man face', max_num=50)

# Create not bald image folder
hair_crawler = GoogleImageCrawler(storage={'root_dir': 'dataset/not_bald'})
hair_crawler.crawl(keyword='man with hair face', max_num=50)