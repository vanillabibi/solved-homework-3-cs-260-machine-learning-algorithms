Download Link: https://assignmentchef.com/product/solved-homework-3-cs-260-machine-learning-algorithms
<br>
<h1>Gradient Descent and Newton’s Method</h1>

In this problem, you will implement (unregularized/regularized) logistic regression for binary classification problems using two di↵erent types of optimization approaches, namely, <strong>batch gradient descent </strong>and <strong>Newton’s method</strong>. Two data sets are given; one of which is text data from which you will learn to construct features. For each of the problems, you need to <em>report your results on both of the datasets</em>. Please note that there are 9 problems in total. Also, <em>cross validation is NOT needed </em>for this programming assignment. <em>For any plot you make, you MUST at least provide title, x-label, y-label and legend (if there is more than one curves)</em>. Please read submission instructions carefully before submitting your report and source codes.

<h1>1           Data</h1>

<strong>Ionosphere </strong>This dataset contains 351 instances and each instance has 34 attributes (features). All feature values are continuous and the last column is the class label (“bad” = b = 1, “good” = g = 0). Your goal is to predict the correct label, which is either “b” or “g”. We already divided the dataset into training and test sets (iono train.dat and iono test.dat). Please use the training set to build your model and use the test set to evaluate your model. For more details about Ionosphere dataset, please refer to UCI website: <a href="https://archive.ics.uci.edu/ml/datasets/Ionosphere">https://archive.ics.uci.edu/ml/datasets/Ionosphere</a><a href="https://archive.ics.uci.edu/ml/datasets/Ionosphere">.</a>

<strong>EmailSpam </strong>This dataset contains 943 instances and each of them is labeled as either a spam or ham (not spam) email. Each data instance is the text which is the content of the email (subject and body), and your goal is to classify each email as either a spam or a ham. We have already divided this dataset into training and test datasets, and each of these datasets is stored in two distinct folders (one corresponding to ham and the other corresponding the spam). In other words, to build your model, you need to iterate through all the text files within /train/spam and /train/ham, and to test your model you need to iterate through /test/spam and /test/ham.

<h1>2           Feature Representation</h1>

An essential part of machine learning is to build the feature representation for raw input data, which is often unstructured. Once we construct the features, each data instance <strong>x</strong><em><sub>i </sub></em>can be represented as:

<strong>x</strong><em><sub>i </sub></em>= (<em>x<sub>i</sub></em><sub>1</sub><em>,…,x<sub>id</sub></em>)

where <em>x<sub>ij </sub></em>denotes the <em>j</em>th feature for <em>i</em>th instance.

<strong>Algorithm 1 </strong>Pseudocode for generating bag-of-word features from text

1: Initialize feature vector bg feature = [0,0,…,0]

2: <strong>for </strong>token in text.tokenize() <strong>do</strong>

3:                <strong>if </strong>token in dict <strong>then</strong>

4:                       token idx = getIndex(dict, token)

5:                      bg feature[token idx]


6:              <strong>else</strong>

7:                    continue

8:               <strong>end if</strong>

9: <strong>end for</strong>

10: return bg feature

<strong>Ionosphere </strong>The dataset is well-formatted, so you can directly use the raw feature values to build your model. Please do not do any normalization. Also, since there is no categorical attribute, converting to binary features (which you did for HW #1) is not needed.

<table>

 <tbody>

  <tr>

   <td width="95"></td>

  </tr>

  <tr>

   <td></td>

   <td></td>

  </tr>

 </tbody>

</table>

<strong>EmailSpam </strong>Since each data instance is text, you need to find a way converting it into a feature vector. In this homework, you will use the “Bag-of-Words” representation. More specifically, you will convert the text into a feature vector in which each entry of the vector is the count of words occur in that text. You will be provided with the predefined dictionary (dic.dat), and you should only consider words that appear in that dictionary and ignore all others.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> Below is the pseudocode for generating bag-of-word features from text. For tokenization, please tokenize the string only using <strong>whitespace and these three delimiters: ’.,?’</strong>. See below for an example:<sup>2</sup>

<u>Email</u>: <em>hey, i have a better o↵er for you, o↵er. better than all other spam filters. Do you like accepting o↵er?</em>

Pre-defined Dictionary: <em>[add, better, email, filters, hey, o↵er,like, spam,special]</em>

Bag-of-words feature vector: [0<em>,</em>2<em>,</em>0<em>,</em>1<em>,</em>1<em>,</em>3<em>,</em>1<em>,</em>1<em>,</em>0]

<strong>(Q1)</strong>. After converting all training data into bag-of-words feature vectors, what are the 3 words that occur most frequently? Report the results using this format:

{(word1: # of occurrences), (word2: # of occurrences), (word3: # of occurrences)}

<h1>3           Implementation</h1>

The regularized cross-entropy function can be written as:

where <strong>x</strong><em><sub>i </sub></em>2 R<em><sup>d</sup></em>, <strong>w </strong>2 R<em><sup>d</sup></em>, and (·) is the sigmoid function. is regularization coe cient and <em>w</em><sub>0 </sub>is bias parameter. Note that we don’t regularize bias term <em>b</em>.

<em>Stopping Criteria</em>. For both algorithms, run for 50 iterations.

<em>Step size</em>. For gradient method, you will use a fixed step size. Recall that Newton’s method does not require a step size.

<em>Initialization</em>. For batch gradient descent, initialize the weight <strong>w </strong>to 0, and <em>b </em>to 0.1. For Newton’s method, set initial weights to the ones we got from batch gradient descent after 5 iterations (when = 0<em>.</em>05<em>,⌘ </em>= 0<em>.</em>01). (Please note that Newton’s method may diverge if initialization is not proper).

<em>Extreme Condition</em>. It is possible that when (<em>b</em>+ <strong>w</strong><em><sup>T</sup></em><strong>x</strong>) approaches 0, log( (<em>b</em>+ <strong>w</strong><em><sup>T</sup></em><strong>x</strong>)) goes to -infinity. In order to prevent such case, please bound the value (<em>b</em>+<strong>w</strong><em><sup>T</sup></em><strong>x</strong>) using small constant value, 1<em>e </em>16. You can use the following logic in your code:

<em>tmp </em>= (<em>b </em>+ <strong>w</strong><em><sup>T</sup></em><strong>x</strong>) if <em>tmp &lt; </em>1<em>e </em>16 then <em>tmp </em>= 1<em>e </em>16

<h2>3.1         Batch Gradient Descent</h2>

<strong>(Q2) </strong>Please write down the updating equation for <strong>w </strong>and <em>b</em>, for both unregularized logistic regression and regularized logistic regression. In particular, at iteration <em>t </em>using data points <em>X </em>= [<strong>x</strong><sub>1</sub><em>,</em><strong>x</strong><sub>2</sub><em>,…,</em><strong>x</strong><em><sub>n</sub></em>], where <strong>x</strong><em><sub>i </sub></em>= [<em>x<sub>i</sub></em><sub>1</sub><em>,…,x<sub>id</sub></em>], and <em>y<sub>i </sub></em>2 {0<em>,</em>1} is the label, how do we compute <strong>w</strong><em><sup>t</sup></em><sup>+1 </sup>and <em>b<sup>t</sup></em><sup>+1 </sup>from <strong>w</strong><em><sup>t </sup></em>and <em>b<sup>t</sup></em>?

<strong>(Q3) </strong>For step sizes <em>⌘ </em>= {0<em>.</em>001<em>,</em>0<em>.</em>01<em>,</em>0<em>.</em>05<em>,</em>0<em>.</em>1<em>,</em>0<em>.</em>5} and <strong>without regularization</strong>, implement Batch gradient descent (without cross-validation, use the whole training data for the gradient calculation).

<ul>

 <li>Plot the cross-entropy function value with respect to the number of steps (<em>T </em>= [1<em>,…,</em>50]) for the training data for each step size. Note: you need to make two plots, one for each dataset.</li>

 <li>Report the <em>L</em><sub>2 </sub>norm of vector <strong>w </strong>after 50 iterations for each step size <em>⌘<sub>i </sub></em>(fill in the table below)</li>

</ul>

<table width="401">

 <tbody>

  <tr>

   <td width="218"><em>L</em><sub>2 </sub>norm (without regularization)</td>

   <td width="46">0.001</td>

   <td width="40">0.01</td>

   <td width="40">0.05</td>

   <td width="33">0.1</td>

   <td width="25">0.5</td>

  </tr>

  <tr>

   <td width="218">Ionosphere EmailSpam</td>

   <td width="46"> </td>

   <td width="40"> </td>

   <td width="40"> </td>

   <td width="33"> </td>

   <td width="25"> </td>

  </tr>

 </tbody>

</table>

<strong>(Q4) </strong>For step sizes <em>⌘ </em>= {0<em>.</em>001<em>,</em>0<em>.</em>01<em>,</em>0<em>.</em>05<em>,</em>0<em>.</em>1<em>,</em>0<em>.</em>5} and with regularization coe cients = {0<em>,</em>0<em>.</em>05<em>,</em>0<em>.</em>1<em>,</em>0<em>.</em>15<em>,…,</em>0<em>.</em>5}, do the following:

<ul>

 <li>Given = 0<em>.</em>1, plot the cross-entropy function value with respect to the number of steps (<em>T </em>= [1<em>,…,</em>50]) for the training data for each step size using di↵erent step sizes. Note: you need to make two plots, one for each dataset.</li>

 <li>Given <em>⌘ </em>= 0<em>.</em>01, report the <em>L</em><sub>2 </sub>norm of vector <strong>w </strong>after 50 iterations for each regularization coe cient <em>i </em>(fill in the table below).</li>

 <li>Plot the cross entropy function value at <em>T </em>= 50 for di↵erent regularization coe cients, for both the training and test data. The x-axis will be the regularization coe               cient and y-axis will be the cross entropy function value after 50 iterations. Each plot should contain two curves, and you should make 2 (two data sets) ⇤⇥ 5 (five di↵erent step sizes) = 10 plots.</li>

</ul>

<table width="632">

 <tbody>

  <tr>

   <td width="255"><em>L</em><sub>2 </sub>norm (with regularization, <em>⌘ </em>= 0<em>.</em>01)</td>

   <td width="23">0</td>

   <td width="40">0.05</td>

   <td width="33">0.1</td>

   <td width="40">0.15</td>

   <td width="33">0.2</td>

   <td width="40">0.25</td>

   <td width="33">0.3</td>

   <td width="40">0.35</td>

   <td width="33">0.4</td>

   <td width="40">0.45</td>

   <td width="25">0.5</td>

  </tr>

  <tr>

   <td width="255">Ionosphere EmailSpam</td>

   <td width="23"> </td>

   <td width="40"> </td>

   <td width="33"> </td>

   <td width="40"> </td>

   <td width="33"> </td>

   <td width="40"> </td>

   <td width="33"> </td>

   <td width="40"> </td>

   <td width="33"> </td>

   <td width="40"> </td>

   <td width="25"> </td>

  </tr>

 </tbody>

</table>

<h2>3.2         Newton’s method</h2>

Optimization also can be done using the 2nd order technique called “Newton’s method”. We define H as the Hessian matrix and r<em>“<sub>t </sub></em>as the gradient of objective function at iteration <em>t</em>.

<strong>(Q5) </strong>Using the notation above, write down the updating equation for <strong>w </strong>and <em>b </em>at time <em>t</em>, for both unregularized logistic regression and regularized logistic regression.

<strong>(Q6) </strong>Implement Newton’s method for logistic regression without regularization, and run it for 50 iterations on both of datasets.

<ul>

 <li>Plot the cross-entropy function value with respect to the number of steps (<em>T </em>= [1<em>,…,</em>50]) for the training data. Note: you need to make two plots, one for each dataset.</li>

 <li>Report the <em>L</em><sub>2 </sub>norm of vector <strong>w </strong>after 50 iterations.</li>

 <li>Report the cross-entropy function value for the test data.</li>

</ul>

<strong>(Q7) </strong>Repeat (Q6) for the regularized case using      = {0<em>,</em>0<em>.</em>05<em>,</em>0<em>.</em>1<em>,</em>0<em>.</em>15<em>,…,</em>0<em>.</em>5} (report values for each setting).

<h2>3.3         Analysis and Comparison of Gradient Descent and Newton’s Method</h2>

<strong>(Q8) </strong>Briefly (in no more than 4 sentences) explain your results from (Q3) and (Q4). You should discuss the rate of convergence as a function <em>⌘</em>, the change in magnitude of <strong>w </strong>as a function of    , the value of the cross entropy function for di↵erent values of              and <em>⌘</em>, and any other interesting trends you have observed.

<strong>(Q9) </strong>Briefly (in no more than 4 sentences) discuss the di↵erences between gradient descent and Newton’s method based on the results from (Q4) and (Q7). In particular, discuss the di↵erence between gradient descent and Newton’s method in terms of convergence and computation time.


