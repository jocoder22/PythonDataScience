#!/usr/bin/env python

# jupyter nbconvert will convert jupyter .ipynb to html or pdf
# from the console type

jupyter nbconvert "C:\Users\Jose\Example 1.ipynb" --to pdf
jupyter nbconvert "C:\Users\Jose\Example 1.ipynb" --to html



# Example: creating slides w/ speaker notes
# Let’s suppose you have a notebook your_talk.ipynb that you want to convert to slides. 
# For this example, we’ll assume that you are working in the same directory as the notebook 
# you want to convert (i.e., when you run ls ., your_talk.ipynb shows up amongst the list of files).
# First, we need a copy of reveal.js in the same directory as your slides. One way to do this is 
# to use the following commands in your terminal:

git clone https://github.com/hakimel/reveal.js.git
cd reveal.js
git checkout 3.5.0
cd ..


# Then we need to tell nbconvert to point to this local copy. To do that 
# we use the --reveal-prefix command line flag to point to the local copy.

jupyter nbconvert "Example 1.ipynb" --to slides --reveal-prefix reveal.js

# This will create file your_talk.slides.html, which you should be able to 
# access with open your_talk.slides.html. To access the speaker notes, 
# press s after the slides load and they should open in a new window.