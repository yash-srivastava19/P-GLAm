## Inspiration and the Problem Statement
It was a random Sunday night, and I was wondering about language modelling when something clicked. There is this famous experiment in probability called the Infinite Monkey Theorem, which states that a monkey hitting keys at random on a typewriter keyboard for an infinite amount of time will almost surely type complete works of Shakespeare. In fact, the monkey would type every possible finite text an infinite number of time. The probability of happening such events is exteremly low but technically not zero. Let's use this fact together with the capabilities of Language Modelling we have today. We'll twist the rules a little bit to better asses the experiment.

Imagine yourself as the monkey in this scenario, but instead of having to type the entire Shakespeare, we'll be typing arithmetic equations. The domain is a little restricted, with little characters(0-9 and binary arithmetic operators) which is ideal given the scope of the experiment. We'll feed this arithmetic data to a GPT-2 inspired model, and check for the validity of the arithmetic equation. **If this experiment is repeated for several iterations - What is the probability the random Language Model will spit a correct arithmetic equation(with no domain knowledge whatsoever) ?**

## How will be the experiemt conducted ?
We'll be implementing GPT-2 from scratch - inspired by the excellent tutorial by @karparthy. But instead of providing some normal data, we'll be creating a dataset of arithmetic equations and fedd it to the Language Model we'll be calling P-GLAm(Pascal GPT Language Arithmetic model - named after the founder of this idea, Blaise Pascal) from now on. Then, we'll allow P-GLAm yo spit out random arithmetic expressions and we'll check the validity of the statements using sympy evaluator and analyze the results. **Can we really have valid arithmetic expressions being generated by a GPT-2 type model ?** We'll find out for sure.

## Parameters of the experiment.

## What kind of data we'll be evaluate for results ?

## Limits of the experiment.

## Results of the experiment.
