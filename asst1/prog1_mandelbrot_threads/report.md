## speedup
The largest speedup is 6.88 in my test with 8 threads

Averagely, the speedup is about 6.5 for both which is impacted by my laptop performance

New idea: we should sort this pixel computing cost to different queue, maybe 256 queues, then distribute these computing pixel to different thread queue from the lowest cost queues to the highest queues which we have sorted