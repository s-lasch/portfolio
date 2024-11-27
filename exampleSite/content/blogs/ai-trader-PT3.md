---
title: "AI-Assisted Trading - Trading (Part 3)"
date: 2024-11-27T08:01:00-05:00
draft: true
author: "Steven Lasch"
tags:
  - applied math
  - reinforcement learning
image: /images/photo-1651340981821-b519ad14da7c.png
description: "A continuation of my series on an AI-based trading strategy using reinforcement learning. This section assesses the role of transaction costst for the learner."
toc: true
mathjax: true
---

## TLDR
This article is the third and final installment of creating an AI-Based trading strategy using reinforcement learning. In the last article, the learner proved to make sensible trades during the testing phase, assuming no transaction costs. In this article, we will explore the effects of one type of transaction cost, **market impact**, and how this number affects trade volume and returns. 

## What is market impact?
Market impact measures how the actions of a buyer or seller affect the price of an asset in the market. It is one type of a broader category called transaction costs. Each transaction has a small effect on the market value of an asset. However, when scaled up to trading with millions of dollars, these tiny fractions magnify to large sums. 

This is bad for our learner, since it was trained on $0 in transaction costs, which is not very realistic. For reference, here is the trade volume chart from the last article:

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/s-lasch/portfolio/refs/heads/master/exampleSite/content/blogs/ai-trader/strategy_learner_trades_out_sample.png" />
</div>

By including market impact as detrimental in our reward function, we can reduce the reward for high market impact levels. This would discourage the learner from making a large number of trades, as each trade would result in decreasing returns. 

## Experimental Results
We should expect our learner to decrease the number of trades it makes over the 2-year period for higher values of market impact. We should also expect cumulative returns to drop for higher levels of market impact.

Running the learner three separate times and plotting the results confirms that as market impact increases, cumulative returns decrease.

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/s-lasch/portfolio/refs/heads/master/exampleSite/content/blogs/ai-trader/experiment2_impact_testing.png" />
</div>

We can test this by querying our optimal policy for all times in which a trade was made.

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/s-lasch/portfolio/refs/heads/master/exampleSite/content/blogs/ai-trader/experiment_2_trade_counts.png" />
</div>

Since the learner is finding an optimal policy given no penalty for frequent transactions, we see that as impact increases, trade volume decreases. 

## Conclusions
There are dozens of theories about market impact, as well as countless examples to account for them. It should be noted that I did not modify this strategy for these factors—this is just an educational example to highlight its importance.

[<< PART 1](https://slasch-portfolio.netlify.app/blogs/ai-trader-pt1/) - [<< PART 2](https://slasch-portfolio.netlify.app/blogs/ai-trader-pt2/)
