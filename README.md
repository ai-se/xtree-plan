# BELLTREE: **BELL**WETHER + **XTREE**
<img width="200" alt="portfolio_view" src="https://s3.amazonaws.com/images.static.steveweissmusic.com/products/images/uploads/popup/SW-450.jpg">

## Submission 

Published in Empirical Software Engineering Journal. Article: [https://link.springer.com/content/pdf/10.1007/s10664-020-09843-6.pdf](https://link.springer.com/content/pdf/10.1007/s10664-020-09843-6.pdf)

## Abstract
The current generation of software analytics tools are mostly prediction algorithms (e.g.
support vector machines, naive bayes, logistic regression, etc). While prediction is useful,
after prediction comes planning about what actions to take in order to improve quality. This
research seeks methods that generate demonstrably useful guidance on “what to do” within
the context of a specific software project. Specifically, we propose XTREE (for withinproject planning) 
and BELLTREE (for cross-project planning) to generating plans that can
improve software quality. Each such plan has the property that, if followed, it reduces the
expected number of future defect reports. To find this expected number, planning was first
applied to data from release x. Next, we looked for change in release x + 1 that conformed
to our plans. This procedure was applied using a range of planners from the literature, as
well as XTREE. In 10 open-source JAVA systems, several hundreds of defects were reduced
in sections of the code that conformed to XTREE’s plans. Further, when compared to other
planners, XTREE’s plans were found to be easier to implement (since they were shorter)
and more effective at reducing the expected number of defects.


## Cite As

```
@article{krishna2020learning,
  title={Learning actionable analytics from multiple software projects},
  author={Krishna, Rahul and Menzies, Tim},
  journal={Empirical Software Engineering},
  pages={1--33},
  year={2020},
  publisher={Springer}
}
```

## Authors

+ Rahul Krishna
  + Columbia University, USA
  + i.m.ralk@gmail.com
+ Tim Menzies
  + North Carolina State University, USA
  + tim@ieee.org  

## Data

+ [Defect Data](/src/data)

## Source Code

+ [BELLTREE](/src/)

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

(BTW, it would be great to hear from you if you are using this material. But that is optional.)

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to http://unlicense.org
