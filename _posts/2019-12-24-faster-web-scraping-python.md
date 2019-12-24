---
title:  "Faster Web Scraping in Python"
date:   2019-12-24
tags: [web scraping, concurrency, multithreading, python]

header:
  image: "faster_web_scraping/web_scraping_image.png"
  caption: "Image credit: Towards Data Science"

excerpt: "Faster Web Scraping in Python with Multithreading"
---

Working on GPU-accelerated [data science libraries](http://rapids.ai/) at NVIDIA, I think about accelerating code through parallelism and concurrency pretty frequently. You might even say I think about it all the time.

In light of that, I recently took a look at some of my old web scraping code across various projects and realized I could have gotten results **much** faster if I had just made a small change and used Python's built-in [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) library. I wasn't as well versed in concurrency and asynchronous programming back in 2016, so this didn't even enter my mind. Luckily, times have changed.

In this post, I'll use `concurrent.futures` to make a simple web scraping task 20x faster on my 2015 Macbook Air. I'll briefly touch on how multithreading is possible here, but won't go into detail. This is really just about highlighting how you can do faster web scraping with almost no changes.


# A Simple Example

Let's say you wanted to download the HTML for a bunch of stories submitted to Hacker News. It's pretty easy to do this. I'll walk through a quick example below.

First, we need get the URLs of all the posts. Since there are 30 per page, we only need a few pages to demonstrate the power of multithreading. `requests` and `BeautifulSoup` make extracting the URLs easy. Let's also make sure to `sleep` for a bit between calls, to be nice to the Hacker News server. Even though we're only making 10 requests, it's good to be nice.


```python
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://news.ycombinator.com/"
STORY_LINKS = []

for i in range(10):
    resp = requests.get(f"{BASE_URL}news?p={i}")
    soup = BeautifulSoup(resp.content, "html.parser")
    stories = soup.find_all("a", attrs={"class":"storylink"})
    links = [x["href"] for x in stories if "http" in x["href"]]
    STORY_LINKS += links
    time.sleep(0.25)

print(len(STORY_LINKS))

for url in STORY_LINKS[:3]:
    print(url)
```

    289
    https://www.thirtythreeforty.net/posts/2019/12/my-business-card-runs-linux/
    https://github.com/Hack-with-Github/Awesome-Hacking
    https://techcrunch.com/2019/12/24/uber-founder-travis-kalanick-is-leaving-the-companys-board-of-directors/


So, we've got 289 URLs. That [first one](https://www.thirtythreeforty.net/posts/2019/12/my-business-card-runs-linux/) sounds pretty cool, actually. A business card that runs Linux?

Let's download the HTML content for each of them. We can do this by stringing together a couple of simple functions. We'll start by defining a function to download the HTML from a single URL. Then, we'll run the download function on a test URL, to see how long it takes to make a `GET` request and receive the HTML content.


```python
import time

def download_url(url):
    t0 = time.time()
    resp = requests.get(url)
    t1 = time.time()
    print(f"Request took {round(t1-t0, 2)} seconds.")
    
    title = "".join(x for x in url if x.isalpha()) + "html"
    
    with open(title, "wb") as fh:
        fh.write(resp.content)

download_url("https://beckernick.github.io/what-blogging-taught-me-about-software/")
```

    Request took 0.53 seconds.


# The Problem

Right away, there's a problem. Making the `GET` request and receiving the response took about 500 ms, which is pretty concerning if we need to make thousands of these requests. Multiprocessing can't really solve this for me, as I only have two physical cores on my machine. Scraping thousands of files will still take thousands of seconds.

We'll solve this problem in a minute. For now, let's redefine our `download_url` function (without the timers) and another function to execute `download_url` once per URL. I'll wrap these into a `main` function, which is just standard practice. These functions should be pretty self-explanatory for those familiar with Python. Note that I'm still calling `sleep` in between `GET` requests even though we're not hitting the same server on each iteration.


```python
def download_url(url):
    print(url)
    resp = requests.get(url)
    title = "".join(x for x in url if x.isalpha()) + "html"
    
    with open(title, "wb") as fh:
        fh.write(resp.content)
        
    time.sleep(0.25)
        
def download_stories(story_urls):
    for url in story_urls:
        download_url(url)

def main(story_urls):
    t0 = time.time()
    download_stories(story_urls)
    t1 = time.time()
    print(f"{t1-t0} seconds to download {len(story_urls)} stories.")
```


```python
main(STORY_LINKS[:5])
```

    https://www.thirtythreeforty.net/posts/2019/12/my-business-card-runs-linux/
    https://github.com/Hack-with-Github/Awesome-Hacking
    https://techcrunch.com/2019/12/24/uber-founder-travis-kalanick-is-leaving-the-companys-board-of-directors/
    https://volument.com/blog/minimalism-the-most-undervalued-development-skill
    https://blog.jonlu.ca/posts/aa-tracker
    5.173792123794556 seconds to download 5 stories on the page

And, now on the full data.

```python
main(STORY_LINKS)
```

    https://www.thirtythreeforty.net/posts/2019/12/my-business-card-runs-linux/
    https://github.com/Hack-with-Github/Awesome-Hacking
    https://techcrunch.com/2019/12/24/uber-founder-travis-kalanick-is-leaving-the-companys-board-of-directors/
    ...
    https://www.isi.edu/~johnh/SOFTWARE/LAVAPS/
    https://www.theverge.com/2019/12/23/21035567/t-mobile-merger-documents-sprint-comcast-merger-assessment
    https://archive.org/details/KodakReferenceHandbook
    319.86593675613403 seconds to download 289 stories on the page


As expected, this scales pretty poorly. On the full 289 files, this scraper took 319.86 seconds. That's about one file per second. At this point, we're definitely screwed if we need to scale up and we don't change our approach. 

# Why Multiprocessing Isn't Enough

So, what do we do next? Google "fast web scraping in python", probably. Unfortunately, the top results are primarily about speeding up web scraping in Python using the built-in `multiprocessing` library. This isn't surprising, as multiprocessing is easy to understand conceptually. But, it's a shame.

The benefits of multiprocessing are basically capped by the number of cores in the machine, and multiple Python processes come with more overhead than simply using multiple threads. If I were to use multiprocessing on my 2015 Macbook Air, it would at best make my web scraping task just less than 2x faster on my machine (two physical cores, minus the overhead of mulitprocessing).

# Multithreaded Web Scraping

Luckily, there's a solution. In Python, I/O functionality is implemented in C and releases the [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) (GIL). This means I/O tasks can be executed concurrently across multiple threads **in the same process**, and that these tasks can happen while other Python bytecode is being interpreted.

Oh, and it's not just I/O that can release the GIL. You can release the GIL in your own library code, too. This is how data science libraries like cuDF and CuPy can be so fast. You can wrap Python code _around_ blazing fast CUDA code (to take advantage of the GPU) that isn't bound by the GIL!

While it's slightly more complicated to understand, multithreading with `concurrent.futures` can give us a significant boost here. We can take advantage of multithreading by making a tiny change to our scraper.


```python
import concurrent.futures

MAX_THREADS = 30

def download_url(url):
    print(url)
    resp = requests.get(url)
    title = ''.join(x for x in url if x.isalpha()) + "html"
    
    with open(title, "wb") as fh:
        fh.write(resp.content)
        
    time.sleep(0.25)
    
def download_stories(story_urls):
    threads = min(MAX_THREADS, len(story_urls))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(download_url, story_urls)

def main(story_urls):
    t0 = time.time()
    download_stories(story_urls)
    t1 = time.time()
    print(f"{t1-t0} seconds to download {len(story_urls)} stories.")
```

Notice how little changed. Instead of looping through `stories` and calling `download_url`, I use the `ThreadPoolExecutor` from `concurrent.futures` to execute the function across many independent threads. I also don't want to launch 30 threads for two URLs, so I set `threads` to be the smaller of `MAX_THREADS` and the number of URLs. These threads operate **asynchronously**.

That's all there is to it. Let's see how big of an impact this tiny change can make. It took about four seconds to download five links before.


```python
main(STORY_LINKS[:5])
```

    https://www.thirtythreeforty.net/posts/2019/12/my-business-card-runs-linux/
    https://github.com/Hack-with-Github/Awesome-Hacking
    https://techcrunch.com/2019/12/24/uber-founder-travis-kalanick-is-leaving-the-companys-board-of-directors/
    https://volument.com/blog/minimalism-the-most-undervalued-development-skill
    https://blog.jonlu.ca/posts/aa-tracker
    0.761509895324707 seconds to download 5 stories.


Six times faster! And, we're still sleeping for 0.25 seconds between calls in each thread. Python releases the GIL while sleeping, too.

What about if we scale up to the full 289 stories?


```python
main(STORY_LINKS)
```

    https://www.thirtythreeforty.net/posts/2019/12/my-business-card-runs-linux/
    ...
    https://www.theverge.com/2019/12/23/21035567/t-mobile-merger-documents-sprint-comcast-merger-assessment
    https://archive.org/details/KodakReferenceHandbook
    17.836607933044434 seconds to download 289 stories.


17.8 seconds for 289 stories! That's **way** faster. With almost no code changes, we got a roughly 18x speedup. At larger scale, we'd likely see even more potential benefit from multithreading.

# Conclusion

Basic web scraping in Python is pretty easy, but it can be time consuming. Multiprocessing looks like the easiest solution if you Google things like "fast web scraping in python", but it can only do so much. Multithreading with [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) can speed up web scraping just as easily and usually far more effectively.

<sub><sub>*Note: This post also syndicated on my [Medium page](https://medium.com/@beckernick).*</sub></sub>