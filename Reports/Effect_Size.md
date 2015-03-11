## Results of Feature weighting and Cliffs Delta tests

- A variance based feature weighting was performed. Each feature was mutated by only a fraction as recommended by __sdiv__. Here's an example of which (and by how much each) feature gets mutated.
 ```
 
 ```
- 
- Cliffs Delta was applied to perform an effect size test on the dataset.

<img src="https://github.com/ai-se/Transfer-Learning/blob/master/Reports/_img/cliffsdelta.png" aligh="middle" alt="alt text" width="300" height="60">

```
Legend:
=======
List1 - Bugs Before Contrast Set; m = No. of elements in List1
List2 - Bugs After Contrast Set; n = No. of elements in List2
```
## Results
```

rank ,         name ,    med   ,  iqr 
----------------------------------------------------
   1 ,          ant ,    0.09  ,  0.03 (  *            |              ), 0.09,  0.09,  0.12
   1 ,       lucene ,    0.10  ,  0.04 (  *            |              ), 0.08,  0.10,  0.12
   1 ,          poi ,    0.11  ,  0.01 (   *           |              ), 0.10,  0.11,  0.12
   1 ,        camel ,    0.11  ,  0.01 (   *           |              ), 0.11,  0.11,  0.11
   2 ,        jedit ,    0.22  ,  0.04 (     -*        |              ), 0.18,  0.22,  0.22
   3 ,        log4j ,    0.35  ,  1.00 (---------------|----*         ), 0.00,  0.70,  1.00
   3 ,       pbeans ,    0.39  ,  0.61 (   --------*   |              ), 0.12,  0.39,  0.74
   3 ,      forrest ,    0.58  ,  0.38 (       --------|-*            ), 0.25,  0.58,  0.62
   4 ,          ivy ,    0.82  ,  0.21 (               |       --*    ), 0.79,  0.86,  1.00
```
