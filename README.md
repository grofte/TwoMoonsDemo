# TwoMoonsDemo

To install packages run `poetry install`

Starting point is in main.py file.

This is a just a quick change of off something Artur made - hence all the Laplace stuff.

Just run it once with `root_n_outliers` = whatever and = 0 and compare the two.
Notice that if `root_n_outliers` is big enough then the classifier stops working.
And keep in mind that adding the same outliers every epoch is pretty bad - it would be much better with a new sampled set of outliers each epoch.
And it's only because the data here is very simple that it works. You can't add outliers to the input when the data is text or images (not efficiently anyways).
But you could add them to a deep layer after they have been transformed to something simpler.

I still haven't read the Virtual Outlier Synthesis paper but I am guessing that this is what they have studied?
But this is close to Entropic Open-Set Loss from a paper called Reducing Network Agnostophobia. 
