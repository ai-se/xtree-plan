## Summary 


## Changes to the Contrast Set Learner

### In the Planning Phase

+ I use DE to tune the data to find the best settings for building the Contrast Set. As expected, tuning helps improve where2's accuracy. My DE tunes for:
  
![](_img/params.png)

+ Instead of standard DE, i used DE/best/2, with:
  - ```xNew = Xbest + F * (A + B - Y - Z)```
  - Here, A,B,Y,Z are candidates selected at random
  
+ Next I build 4 Islands of 10 potential row values and use CART (trained on the SMOTED, pruned training dataset) to estimate the number of bugs in each island. If the mean of the estimated bugs is less than the original number of bugs then I choose a row from that island.
 - I use a distance/depth policy to create these 4 islands (nearest, near, far, farthest). And pick the best contrast set from that.

+ No SMOTING while planning (SMOTING whilst planning, for some reason, wreaks havoc).

+ The DATA is discretized into Defective == {True, False} with a threshold 1. ```Defect = True if bugs > 1 else 0```

### Prediction Phase
+ CART is now trained on Binary Data. This is actually [better than original results](https://github.com/rahlk/Research/wiki/SMOTE-Pt-I), there smoting didn't happen on binary data.
  - A threshold of Bugs > 1 is used to determine if a row is defective (or not).
  - CART is trained on a the other half of the pruned dataset. **NOTE: I do not use the same rows of the training data for planning and prediction.**



**Prediction Accuracy of SMOTED CART with DEFECTS = {TRUE, FALSE} and with threshold of 1**
```

rank ,         name ,    med   ,  iqr 
----------------------------------------------------
   1 ,      synapse ,    0.00  ,  0.00   (*              |              ), 0,  0,  0
   2 ,        log4j ,    9.00  ,  9.00   (---*           |              ), 2,  11,  14
   2 ,          ivy ,    15.00  ,  15.00 (-----*         |              ), 0,  15,  15
   3 ,       xerces ,    35.00  ,  5.00  (          --*  |              ), 31,  35,  37
   4 ,        xalan ,    50.00  ,  5.00  (               |-*            ), 46,  50,  53
   4 ,       pbeans ,    55.00  ,  31.00 (        -------|---*          ), 25,  56,  62
   5 ,        camel ,    60.00  ,  2.00  (               |    -*        ), 58,  60,  60
   5 ,       lucene ,    61.50  ,  2.00  (               |     *        ), 60,  62,  62
   5 ,          poi ,    61.50  ,  4.00  (               |    -*        ), 59,  62,  64
   6 ,     velocity ,    66.00  ,  1.00  (               |      -*      ), 65,  66,  66
   6 ,      forrest ,    69.50  ,  21.00 (               |    -------*  ), 59,  78,  80
   7 ,          ant ,    73.00  ,  1.00  (               |         *    ), 72,  73,  74
   8 ,        jedit ,    79.00  ,  2.00  (               |           *  ), 78,  79,  80
```

**GAIN**
```

rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,          ant ,        6  ,      0 (*              |              ), 6,  6,  6
   1 ,        camel ,        6  ,      0 (*              |              ), 6,  6,  6
   1 ,        jedit ,    -1.23  ,   5.32 (        -------|-----*        ), -4,  0,  2
   1 ,      forrest ,     0.00  ,  42.22 (   -------*    |              ), -20,  0,  30
   1 ,     velocity ,    -1  ,  0 (*              |              ), -1,  -1,  -1
```
## So... What next?
 - Create a New Contrast Set Rig with just CART, see if that helps increasing the gain.
 - I noticed my Tuning Rig [reported on Jan 29th](https://github.com/rahlk/Research/wiki/SMOTE) was not parsing parameters to my CART with SMOTE as I wanted it to; thus, I am now re-evaluating the effect of SMOTE+tuning just to be sure.. 
