---
title:  "What 200,000 Readers Taught Me About Building Software"
date:   2019-11-03
tags: [machine learning, mathematics]

header:
  image: "blogging_product_development/writing_image.jpg"
  caption: "Photo Credit: None"

excerpt: "More point estimates"
---

Since my first post about three years ago, more than 200,000 different people have read this blog. Two hundred thousand people. The size of that number blows my mind. 200,000 people is nearly twice the size of the largest stadiums in the United States. I don’t know what it is, but something about seeing a number like that just humbles you.

What began as a place for me to document my fooling around with the technology and modeling techniques I was using at the Federal Reserve Board grew into a site that thousands of people use to learn more about data science. [Andy Grove](https://en.wikipedia.org/wiki/Andrew_Grove) (former CEO and employee #3 of Intel) wrote that the value of writing reports “stems from the discipline and the thinking the writer is forced to impose upon himself as he identifies and deals with trouble spots in his presentation.” I now believe this holds true for blogging, too.

To celebrate 200,000 readers, I spent some time reflecting on how writing this blog has helped clarify my thinking about building developer tools (software products for software developers). I’ve cut down my thinking into four lessons from blogging that apply to product development:

- Make Documentation Focused, Clear, and Independent.
- Listen to Your Users
- Example-Based Documentation Builds Champions
- Market Size and Timing Matter

I’ll dig deeper on each one.

## Make Documentation Focused, Clear, and Independent
Multiple short, clear posts have served me better than single large, exhaustive posts. My three shortest, most focused posts are also in my top five most viewed posts. It’s tempting to write a long, detailed post, but the utility of that content seems to fade faster. In my experience, a table of contents combined with independent, discoverable pages that each address one topic of interest leads to more engaged and a higher volume of readers.

Analogously, product documentation that describes every feature in a single page becomes hard to maintain as the product’s breadth increases, and provides fewer user entry points for search engine discoverability. Instead, serve product documentation in a single location composed of independent pages, all cleanly accessible from the primary entry point.

[Twilio](https://www.twilio.com/docs/sms) and [Stripe](https://stripe.com/docs/api) do developer documentation perhaps better than any other companies. In Stripe’s case, navigating from the homepage to the “Create a Charge” section of their documentation site is trivial, and Googling the obvious query “Stripe API create charge” brings me directly to that same page. As a user, when I’m looking for a specific feature in the documentation, I want to get the answer to my question as quickly as possible. Immediately finding what I need is a better user experience than spending more time searching through long articles covering multiple topics. At the end of the day, reading documentation is only incidentally related to the real goal of solving my problem.


## Listen to Your Users
Listening to readers has been my most effective way of identifying content that led to successful posts. A junior data scientist at my former employer mentioned that he was studying neural networks, but couldn’t find a good example-based explanation of the sigmoid function derivative he saw in all of the readings. That discussion turned into [Deriving the Sigmoid Derivative for Neural Networks](https://beckernick.github.io/sigmoid-derivative-neural-network/), which has been one of the most successful posts on this site. 

When we shift from product user to product builder, we step farther away from the user’s day-to-day problems. It’s easy to forget that. Communicating with users is crucial to closing that gap. As an example, let’s look at the popular Python libraries Pandas and Dask. Pandas and Dask are some of the most extensively documented libraries in the entire PyData ecosystem.

Yet, user surveys by [Pandas](https://dev.pandas.io/pandas-blog/2019-pandas-user-survey.html) and [Dask](https://blog.dask.org/2019/08/05/user-survey) found that users want more example-based documentation. In fact, “even among those using Dask everyday more people thought that “More examples” is more valuable than “New features” or “Performance improvements”.” Both novice and power users benefit from example-based documentation.


## Example-Based Documentation Builds Champions
Despite those survey results, some people think that APIs should be self-documenting, and that any additional documentation is irrelevant. While APIs should always aim to be clear and minimally surprise users, I think building a strong community around a product requires more than just a good API.

My experience with this blog has been that people not only respond well to example-based documentation, but they learn the content faster. The most viewed posts are those that walk through a technical topic by example, explaining each component as it’s used. Speed of learning matters. For products I’ve worked on, users who have had faster [time to utility](https://tomtunguz.com/time-to-utility/) have been more likely to continue and expand their use of the product.

Highlighting Twilio again, if I Google “[twilio send text message python](https://www.google.com/search?q=twilio+send+sms+python&oq=twilio+send+sms+python&aqs=chrome..69i57j69i60.356j0j1&sourceid=chrome&ie=UTF-8),” the first result provides me with a copy/pastable code example I can use to get up and running. It doesn’t get much faster than that. Documentation like this builds internal champions while reinforcing a low-touch (and even zero-touch) initial sales model that can lead to internal expansion.


## Market Size and Timing Matter
Fred Wilson is right: [market size matters most](https://avc.com/2019/03/market-team-product/). My posts on larger topics have been more successful in terms of viewership than posts about more niche topics. But, readers have still engaged with and commented on my niche posts. Posts about niche topics and products with small addressable markets often have different growth trajectories than ones targeting larger markets, which should affect your strategy. Products that need a large user base to succeed often need to grow differently than those that can survive before achieving scale. To paraphrase Joel Spoelsky, [are you Ben and Jerry’s or are you Amazon?](https://www.joelonsoftware.com/2000/05/12/strategy-letter-i-ben-and-jerrys-vs-amazon/)

But, timing matters, too.

First, markets are inherently dynamic. From 2016-2019, the number of people seeking to learn more about data science and machine learning [doubled](https://trends.google.com/trends/explore?date=2016-10-04%202019-11-04&geo=US&q=machine%20learning,data%20science). Posts about topics for which there was latent demand ([Logistic Regression From Scratch in Python](https://beckernick.github.io/logistic-regression-from-scratch/)) did better than posts for which there was more saturation ([Building a Neural Network from Scratch in Python and in TensorFlow](https://beckernick.github.io/neural-network-scratch/)), even though both posts did well. It’s part of the reason why some people describe product-market fit as when “the market pulls product out of the startup”.

Second, posts generally did better after other posts started doing better. When credibility or recognition is a factor in discoverability, success can engender success. In other words, user base and network effects matter in many software mediated markets. This is likely part of the reason why we see large technology companies leverage their existing installed base to expand into related markets (Spotify moving into podcasts, Facebook into online dating and commercial marketplaces, etc.).


## Writing to Interrogate Your Ideas
Over the past three years, this blog transitioned from a fun way to kill time into a vehicle for better understanding software company dynamics. I’ve seen how focused, discoverable documentation accelerates users’ time to utility, I’ve experienced firsthand how talking with users can change your roadmap and lead to more successful content, and I’ve lived the importance of market size and timing.

Writing has helped me clarify my strategy by forcing me to interrogate my ideas. Not all ideas withstand scrutiny, but they help lead you to ones that do. Turns out, those end up being the better blog posts.


<sub><sub>*Thanks to Bryan Silverman, Jonathan Reshef, and Jordan Laney for reviewing this post.*</sub></sub>