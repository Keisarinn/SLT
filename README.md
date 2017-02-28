# Statistical Learning Theory: Coding Exercises #

This repository serves two purposes:
* Distribute the coding exercises, including the templates.
* Collect the code written by students for the SLT coding exercises.

The coding exercises concern the implementation of algorithms taught in the lecture/exercise class.

The final grade for the lecture is max(exam, 0.7 exam + 0.3 project). 

Students who wish to get the advantage of the project bonus need to submit reports about their coding excercises.

There will be eight coding ecercises, and each report will be graded either as good, normal or not accepted/not submitted. 

With no submitted/accepted reports, the project grade is 4.0. Each good report increases the project grade by 0.25, while each normal report increases it by 0.125.

This means eight good reports result in a project grade of 6.0, while eight normal reports will give you a project grade of 5.0. 

## How to find exercises ##
The exercises templates are published on the master branch. The latex templates contain all the information about the exercises.

## How to do an exercise ##
First you need to setup the repository on your machine, see initial setup below.

Once an exercise template is online, derive your working branch for the exercise:

```git checkout master```

```git checkout -b xx-xxx-xxx/exercise_name```

Make sure you provide your correct ETH legi number in the format xx-xxx-xxx and the exercise name identical to the name of the latex template.

The legi has to be correct, so that we can match it!

Now you can go ahead and do some work, like compiling the template, adding and running code, etc.

Use the template also for your report and delete the instructions which are in the document.

Do not forget to set your name, legi number and git branch at the beginning of each template.

You can always commit and push to your branch and we will take your final report from your branch.

If you are uncertain if everything is where it should be, just look at your branch in the browser.

## Initial setup ##

(1) Request and get access to the repository.

(2) Clone the repository to your machine, i.e.

```git clone https://gitlab.vis.ethz.ch/vwegmayr/slt-coding-exercises.git```

(3) Change into the repository directory:
    
```cd slt-coding-exercises```