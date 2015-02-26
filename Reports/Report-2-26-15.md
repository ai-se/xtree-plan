## Summary 


## Changes to the Contrast Set Learner

### In the Planning Phase

+ I use DE to tune the data to find the best settings for building the Contrast Set.

+ Next I build 4 Islands of 10 potential row values and use CART (trained on the SMOTED, pruned training dataset) to estimate the number of bugs in each island. If the mean of the estimated bugs is less than the original number of bugs then I choose a row from that island.
 - I use a distance/depth policy to create these 4 islands (nearest, near, far, farthest). And pick the best contrast set from that.

+ No SMOTING while planning (SMOTING before planning, for some reason, wreaks havoc).

+ The DATA is discretized into Defective == {True, False} with a threshold 1. ```Defect = True if bugs > 1 else 0```

### Prediction Phase
+ CART is now trained on Binary Data.
  - A threshold of Bugs > 1 is used to determine if a row is defective (or not).
  - CART is trained on a the other half of the pruned dataset. **NOTE: _**We do not use the same training data for planning and prediction.**_**



**Prediction Accuracy**

```
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,          ivy ,    0.00  ,  0.00 (*              |              ), 0,  0,  0
   1 ,      synapse ,    0.00  ,  0.00 (*              |              ), 0,  0,  0
   1 ,        log4j ,    14.00  ,  0.00 (     *         |              ), 14,  14,  14
   1 ,       pbeans ,    20.00  ,  0.00 (       *       |              ), 20,  20,  20
   1 ,       xerces ,    36.00  ,  0.00 (             * |              ), 36,  36,  36
   2 ,        xalan ,    52.00  ,  0.00 (               |   *          ), 52,  52,  52
   2 ,       lucene ,    58.00  ,  0.00 (               |      *       ), 58,  58,  58
   2 ,          poi ,    60.00  ,  0.00 (               |      *       ), 60,  60,  60
   2 ,        camel ,    61.00  ,  0.00 (               |       *      ), 61,  61,  61
   3 ,     velocity ,    65.00  ,  0.00 (               |        *     ), 65,  65,  65
   3 ,          ant ,    73.00  ,  0.00 (               |           *  ), 73,  73,  73
   3 ,      forrest ,    78.00  ,  0.00 (               |             *), 78,  78,  78
   3 ,        jedit ,    79.00  ,  0.00 (               |             *), 79,  79,  79
```
