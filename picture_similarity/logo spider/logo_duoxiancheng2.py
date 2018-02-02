# -*- coding:utf-8 -*-
import queue
import requests
import os
import threading
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np

picture_num=0

#多线程下载图片,递归下载，一次只启动一个程序
#另外一种思路，用另外一个python程序去控制启动其他python程序
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'private',
    'Connection': 'keep-alive',
    'Host': 'www.logodashi.com',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36',
    'Content-Type': 'text/html; charset=utf-8',
}
class Worker(threading.Thread):  # 处理工作请求
    def __init__(self, workQueue, resultQueue, **kwds):
        threading.Thread.__init__(self, **kwds)
        self.setDaemon(True)
        self.workQueue = workQueue
        self.resultQueue = resultQueue

    def run(self):
        while 1:
            try:
                callable, args, kwds = self.workQueue.get(False)  # get task
                res = callable(*args, **kwds)
                self.resultQueue.put(res)  # put result
            except queue.Empty:
                break


class WorkManager:  # 线程池管理,创建
    def __init__(self, num_of_workers=10):
        self.workQueue = queue.Queue()  # 请求队列
        self.resultQueue = queue.Queue()  # 输出结果的队列
        self.workers = []
        self._recruitThreads(num_of_workers)

    def _recruitThreads(self, num_of_workers):
        for i in range(num_of_workers):
            worker = Worker(self.workQueue, self.resultQueue)  # 创建工作线程
            self.workers.append(worker)  # 加入到线程队列

    def start(self):
        for w in self.workers:
            w.start()

    def wait_for_complete(self):
        while len(self.workers):
            worker = self.workers.pop()  # 从池中取出一个线程处理请求
            worker.join()
            if worker.isAlive() and not self.workQueue.empty():
                self.workers.append(worker)  # 重新加入线程池中
        print('All jobs were complete.')

    def add_job(self, callable, *args, **kwds):
        self.workQueue.put((callable, args, kwds))  # 向工作队列中加入请求

    def get_result(self, *args, **kwds):
        return self.resultQueue.get(*args, **kwds)


def get_start_links(url):
    html = requests.get(url, headers=headers,timeout=10)
    time.sleep(0.1)
    #html.encoding = 'utf-8'
    html = html.text
    return html

def download_file(item):
    global picture_num
    pic=None
    pic = requests.get(item,headers=headers,timeout=10)
    string = './picture_place/'+str(picture_num)+'.jpg'
    try:
        tmp=pic.content
        with open(string, 'wb') as fp:
            fp.write(pic.content)
    except Exception as e:
        print("picture with no data")
    picture_num += 1
    print(string, "downloading")

def main():
    if not os.path.exists('picture_place'):
        os.mkdir('picture_place')
    num_of_threads = 10
    _st = time.time()
    wm = WorkManager(num_of_threads)
    print(num_of_threads)
    num=0
    save_picture_list=[]
    for num in tqdm(range(1)):
        url="http://www.logodashi.com/Home/?type=1&CurrentPage="+str(num)+"&SortID=0&KeyWord="
        print(url)
        main_page = get_start_links(url)
        soup = BeautifulSoup(main_page, 'lxml')
        A = soup.find('div',class_='body').find_all('img')
        for item in A:
            save_picture_list.append('http://www.logodashi.com/'+item.attrs['src'])
    np.save('save_picture_list.npy',save_picture_list)
    print("load url complete! and begin download")
    for url in tqdm(save_picture_list):
        wm.add_job(download_file, url)
    wm.start()
    print("start")
    wm.wait_for_complete()
    print(time.time() - _st)
if __name__ == '__main__':
    main()

