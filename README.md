# DecorRater
A rater of house internal decoration based on images.
A original thought of this object is based on object oeirented programming based on python.
之前用SVM的想法行不通，因为SVM只能输出+-，现在改用洛吉斯特回归作为多分类框架的底层分类器，因为洛吉斯特回归输出的是概率，能够逐个进行比较选出最概然标签
