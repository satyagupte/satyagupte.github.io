---
date: 2023-03-22 03:58:03+00:00
excerpt: None
layout: post
slug: ads-auction
title: How Ads Work - Part 4,  Auction
wordpress_id: 199
categories:
- ML for Ads
tags:
- auction
- GSP
---

Welcome to the fourth post in the [How Ads work](https://nofreehunch.org/category/ml-for-ads/) series. Now it’s time to earn some money and sell ads ! This will be a short post and my goal in this post is to outline the basic mechanics of how the ads auction runs and complete our end to end understanding of the entire ads funnel.

#### Table of contents
- [Why do we need an Auction](#why-do-we-need-an-auction)
- [First pass at a solution](#first-pass-at-a-solution)
- [Generalized Second Price Auctions](#generalized-second-price-auctions)
  - [What about slot position](#what-about-slot-position)
- [Closing thoughts](#closing-thoughts)
- [Further Reading](#further-reading)

## Why do we need an Auction

After [Ranking](https://nofreehunch.org/2023/03/10/ads-ranking/) we have a list of ads ranked by probability of being clicked. From our list of ranked ads, we use a cutoff (on Utility or pClick)  to select the ads most likely to be clicked. For a traditional recommender system we would just show these ads in the same order and maximize user clicks. But for ads our goal is to maximize revenue, while keeping users happy.

Remember from our [introductory post](https://nofreehunch.org/2023/02/17/how-ads-work/) that advertisers pay the platform only when a user clicks on their ad (this is called CPC, or cost per click). The question is how much should we (the platform) charge the advertisers.

A crude first attempt would be to simply have a fixed price per position and charge the advertisers accordingly. This is roughly how advertising in a printed magazine works and in the early days of internet advertising the platform rented out space in similar style(remember ugly banner ads ?). This inflexible scheme has several drawbacks chief amongst which is the problem of determining the "right" price. Ads for a user searching for “apartments for rent” are worth much more than for a search like “quotes about sunrise”. 

The auction is the solution to this problem of finding the right price for each ad. 

## First pass at a solution

We can design an auction very much like how auctions run in the physical world of bids and hammers. Advertisers bid for each query and the highest bidder wins. We usually have multiple slots, so the winner gets the first slot, the second highest bid gets the second slot and so on. 

We can further improve on this by not ranking only on bid, but by the expected revenue for the platform. For each ad , we know the probability of a user clicking it  (pClick). We multiply this by the bid and get the expected revenue 

_Expected revenue  = pClick * bid_

Now we place ads in the slots ranked by expected revenue. This scheme looks like it should maximize the platform revenue while also keeping users happy. It does have one subtle kink though.

An advertiser can monitor the platform in an automated manner to figure out if its ad was shown for a particular query. Starting off with a really low bid, the advertiser could keep incrementing the bid to figure out just the minimum bid required to win over its competitors. Other advertisers could of course employ the same strategy and there would be a never ending race to outbid each other. In a pathological scenario, advertisers could even collude to drive down prices. There is no equilibrium of prices in this auction scheme .

## Generalized Second Price Auctions

Google solved this problem by introducing a scheme called the Generalized Second Price (GSP) auction in 2002. 

The ads are ranked by expected revenue (pClick * bid) and placed in slots. But instead of each advertiser paying its own bid, it pays what the advertiser one slot below it had bid plus a small increment. The figure below should help understand this clearly.

![](/assets/img/post_images/2023_03_auction-1.jpeg)Figure - GSP with 3 slots 

Now advertisers are no longer able to figure out the minimum bid required to win an auction, since they don’t know what the second highest bid is. This keeps the auction efficient and prevents attempts to game the system. There are additional rules like minimum floor prices and if bids are below that no ads get shown.

### What about slot position

If you read the post on [Ads Ranking](https://nofreehunch.org/2023/03/10/ads-ranking/), you should have noticed that our pClick model outputs probability of an ad getting clicked for the first slot position. We set up our training to use slot position as an explanatory feature but during serving we set this feature to 1. We did this to reduce the position bias inherent in our training data.

Now the question is if its legit to use pClick (which is for slot 1) when we calculate expected revenue (pClick * bid) and place ads in all available slots. The answer is that it’s okay because even though click probability actually decreases as we go to lower positions, since it’s a constant scaling factor it does not change the actual allocation of ads to lower slots.

Another way to think of this is that after the 1st slot has been allocated using pClick for slot 1, we run another auction with the remaining ads but now using pClick for slot 2. Since pClick for slot 2 is simply some scaled down pClick for slot 1, it does not change the order of ad allocation.

This argument works only if there are **no advertiser-position effects**. That is for different ads with the same pClick for slot 1, they get scaled down for lower positions equally. In concrete terms for a query like “sugary fizzy drink”, coke and pepsi ads have the same relevance and get equal clicks for the same position.

If we really want to model advertiser position effects, we could in theory run the Ranking model for different slot positions but this gets very computationally inefficient. One idea would be to “factorize” the Ranking model into two, one complex model with all the features except position and another simpler model with position features. The complex model needs to be run only once and the simpler position model can be run several times for each position. 

## Closing thoughts

Our big picture goal is to maximize revenue while keeping users happy. We keep users happy by using ML for [Retrieval](https://nofreehunch.org/2023/02/25/retrieval/) and [Ranking](https://nofreehunch.org/2023/03/10/ads-ranking/) and select only the most relevant ads that enter the auction. The auction then maximizes revenue by using  rules for allocation and pricing like the GSP mechanism.  For a healthy and efficient auction we need to have a large pool of advertisers competing for a few slots. A large pool of advertisers only show up when the platform provides true value to users. This needs relevant ads and even more relevant organic content. 

This concludes the How Ads work series of posts. I had a great time writing these  and we covered a lot of ground. I hope you enjoyed reading about the machine learning behind ads too. Thank you for reading!

## Further Reading 

  * Classic [Paper](https://www.nber.org/system/files/working_papers/w11765/w11765.pdf) by Edelman and others that explains the evolution and rules behind GSP. 

  * Introductory [pape](https://www.di.ens.fr/~lelarge/soc/varian2.pdf)r by Hal Varian on Auctions 

  * Another [paper](https://people.ischool.berkeley.edu/~hal/Papers/2009/online-ad-auctions.pdf) by Hal Varian on Auctions 


