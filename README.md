# Weird Deep Learning Metrics
📚 A collection of all the Deep Learning Metrics that I came across which are not accuracy/loss.

| Metric | Description | Notes |
|--------|:-----------|:-----|
|[Churn](metrics/churn.py)|Shows the disagreement between two model's prediction. Higher the churn, more the disagreement. Bounds: [0,1]. Takes two prediction arrays as inputs and retuns churn percentage.| From [this paper](https://arxiv.org/abs/2106.11872)|
