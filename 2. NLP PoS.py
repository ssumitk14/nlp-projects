# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 17:45:25 2021

@author: Sumit
"""

import nltk

paragraph = """It has finally happened, you guys. We have witnessed a historic moment. Well, we witnessed many historic moments, but Leonardo DiCaprio winning an Oscar for the first time just felt like a victory for all of us, you know? DiCaprio has been nominated for an Oscar so many times, only to go home empty-handed, that it has reached meme levels. So to see him finally stand on that stage with a golden statue in his hand was just so heartwarming. Unsurprisingly, when you look at the transcript of DiCaprio's Oscar acceptance speech (which he must surely have been planning for the last 10 years), you see how it's not just thanking people and then breezing off the stage. No, the 41-year-old actor took a moment to speak about a cause that's very close to his heart: environmentalism and climate change.I mean, don't get me wrong. DiCaprio did thank a lot of people, some of them collectively. (Like his friends. "You know who you are.") But working on The Revenant highlighted a cause that DiCaprio supports when the cameras stop rolling, and he wasn't going to let that go unmentioned in his speech. Not when he finally had the eyes of the world upon him for more than just making Sad Leo at the Oscars GIFs. Reading his speech should be enough to inspire fans to give environmentalism the same attention that we've given to DiCaprio's numerous losses over the years. However, even if it doesn't, the fact that he would draw attention away from himself and toward a good cause is just so DiCaprio that it really makes me feel like he deserved this award for more than one reason.You can watch the video of the speech here. Check out his full speech below:Thank you all so very much. Thank you to the Academy. Thank you to all of you in this room. I have to congratulate the other incredible nominees this year. The Revenant was the product of the tireless efforts of an unbelievable cast and crew. First off, to my brother in this endeavor, Mr. Tom Hardy. Tom, your talent on screen can only be surpassed by your friendship off screen … thank you for creating a transcendent cinematic experience. Thank you to everybody at Fox and New Regency … my entire team. I have to thank everyone from the very onset of my career … To my parents; none of this would be possible without you. And to my friends, I love you dearly; you know who you are.And lastly, I just want to say this: Making The Revenant was about man's relationship to the natural world. A world that we collectively felt in 2015 as the hottest year in recorded history. Our production needed to move to the southern tip of this planet just to be able to find snow. Climate change is real, it is happening right now. It is the most urgent threat facing our entire species, and we need to work collectively together and stop procrastinating. We need to support leaders around the world who do not speak for the big polluters, but who speak for all of humanity, for the indigenous people of the world, for the billions and billions of underprivileged people out there who would be most affected by this. For our children’s children, and for those people out there whose voices have been drowned out by the politics of greed. I thank you all for this amazing award tonight. Let us not take this planet for granted. I do not take tonight for granted. Thank you so very much."""

words = nltk.word_tokenize(paragraph)

#POS tagging returns a tuple of two element.
tagged_words = nltk.pos_tag(words)


# Since we cannot directly use the pos tuple
# So, we'll create a new pragraph with the words and pos words

words_pos = []
for tw in tagged_words:
    words_pos.append(tw[0] + "_" + tw[1])

tagged_paragraph = " ".join(words_pos) 













